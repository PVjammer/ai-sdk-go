// Package httprequest provides an HTTP client tool with optional OpenAPI spec support.
//
// SSRF protection is enabled by default: requests to private/loopback IP ranges
// are blocked unless allow_private_ips=true. Map-type inputs (headers,
// query_params, path_params) are accepted as JSON strings so the CLI builder
// and LLM tools both work without schema changes.
package httprequest

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"time"

	"github.com/pvjammer/ai-sdk-go/internal/schema"
	"github.com/pvjammer/ai-sdk-go/internal/validation"
	"github.com/pvjammer/ai-sdk-go/pkg/function"
)

// privateIPBlocks lists the CIDR ranges blocked by default SSRF protection.
var privateIPBlocks []*net.IPNet

func init() {
	for _, cidr := range []string{
		"127.0.0.0/8",    // loopback
		"10.0.0.0/8",     // RFC 1918
		"172.16.0.0/12",  // RFC 1918
		"192.168.0.0/16", // RFC 1918
		"169.254.0.0/16", // link-local
		"::1/128",        // IPv6 loopback
		"fc00::/7",       // IPv6 unique-local
	} {
		_, block, _ := net.ParseCIDR(cidr)
		if block != nil {
			privateIPBlocks = append(privateIPBlocks, block)
		}
	}
}

// isPrivateIP returns true when ip falls within any of the blocked ranges.
func isPrivateIP(ip net.IP) bool {
	for _, block := range privateIPBlocks {
		if block.Contains(ip) {
			return true
		}
	}
	return false
}

// HTTPRequestFunction executes HTTP requests.
type HTTPRequestFunction struct {
	function.Base
}

// Input defines the input schema for the http_request tool.
//
// Map-typed fields (headers, query_params, path_params) are represented as
// JSON strings so the auto-generated CLI and LLM tool schemas stay compatible.
type Input struct {
	// Core HTTP fields
	URL              string `json:"url" description:"Target URL (required unless using spec + operation_id)"`
	Method           string `json:"method" enum:"GET,POST,PUT,PATCH,DELETE,HEAD,OPTIONS" default:"GET" description:"HTTP method"`
	Headers          string `json:"headers" description:"Request headers as a JSON object (e.g., {\"Content-Type\":\"application/json\"})"`
	QueryParams      string `json:"query_params" description:"URL query parameters as a JSON object (e.g., {\"key\":\"value\"})"`
	Body             string `json:"body" description:"Request body (for POST/PUT/PATCH requests)"`
	TimeoutSeconds   int    `json:"timeout_seconds" validate:"min=0,max=120" default:"30" description:"Request timeout in seconds (1–120)"`
	MaxResponseBytes int    `json:"max_response_bytes" default:"102400" description:"Maximum response body bytes to return (default 100 KB)"`
	AllowPrivateIPs  bool   `json:"allow_private_ips" default:"false" description:"Allow requests to private/local IP ranges (disabled by default for SSRF protection)"`

	// OpenAPI fields (all optional)
	SpecPath    string `json:"spec_path" description:"Path to a local OpenAPI 3.x spec file (JSON or YAML)"`
	SpecURL     string `json:"spec_url" description:"URL of a remote OpenAPI 3.x spec"`
	OperationID string `json:"operation_id" description:"operationId from the spec; resolves URL template and HTTP method"`
	PathParams  string `json:"path_params" description:"Path template variables as a JSON object (e.g., {\"id\":\"123\"} for /users/{id})"`
}

// Output is the result of an HTTP request.
type Output struct {
	StatusCode  int               `json:"status_code"`
	Headers     map[string]string `json:"response_headers"`
	Body        string            `json:"body"`
	BodyBytes   int               `json:"body_bytes"`
	Truncated   bool              `json:"truncated"`
	ContentType string            `json:"content_type"`
	FinalURL    string            `json:"final_url"`
}

// New creates a new HTTPRequestFunction with the given config.
func New(cfg function.Config) function.AIFunction {
	return &HTTPRequestFunction{Base: function.NewBase(cfg)}
}

// Execute implements AIFunction.
func (f *HTTPRequestFunction) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	var in *Input

	switch v := input.(type) {
	case *Input:
		in = v
	case map[string]interface{}:
		in = parseMapInput(v, f.Config())
	default:
		return nil, function.NewExecutionError(f, "input validation", "invalid input type")
	}

	// Apply defaults.
	if in.Method == "" {
		in.Method = "GET"
	}
	if in.TimeoutSeconds <= 0 {
		in.TimeoutSeconds = 30
	}
	if in.MaxResponseBytes <= 0 {
		in.MaxResponseBytes = 102400
	}

	if err := validation.Validate(in); err != nil {
		return nil, function.WrapValidationError(f, err)
	}

	// Validate HTTP method manually (enum tag not validated by go-playground for strings without oneof).
	validMethods := map[string]bool{
		"GET": true, "POST": true, "PUT": true, "PATCH": true,
		"DELETE": true, "HEAD": true, "OPTIONS": true,
	}
	if !validMethods[strings.ToUpper(in.Method)] {
		return nil, function.NewExecutionError(f, "validation", "invalid HTTP method: "+in.Method)
	}
	in.Method = strings.ToUpper(in.Method)

	// Parse JSON map fields.
	headers, err := parseJSONStringMap(in.Headers, "headers")
	if err != nil {
		return nil, function.WrapExecutionError(f, "parse headers", err)
	}
	queryParams, err := parseJSONStringMap(in.QueryParams, "query_params")
	if err != nil {
		return nil, function.WrapExecutionError(f, "parse query_params", err)
	}
	pathParams, err := parseJSONStringMap(in.PathParams, "path_params")
	if err != nil {
		return nil, function.WrapExecutionError(f, "parse path_params", err)
	}

	// Resolve URL from OpenAPI spec when operation_id is provided and URL is missing.
	targetURL := in.URL
	method := in.Method
	if in.OperationID != "" && targetURL == "" {
		resolvedURL, resolvedMethod, resolveErr := resolveOperation(in.SpecPath, in.SpecURL, in.OperationID, pathParams)
		if resolveErr != nil {
			return nil, function.WrapExecutionError(f, "resolve operation", resolveErr)
		}
		targetURL = resolvedURL
		method = resolvedMethod
	}

	if targetURL == "" {
		return nil, function.NewExecutionError(f, "validation", "url is required (or provide spec + operation_id)")
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Build request URL with query params.
	reqURL, err := url.Parse(targetURL)
	if err != nil {
		return nil, function.WrapExecutionError(f, "parse url", err)
	}
	if len(queryParams) > 0 {
		q := reqURL.Query()
		for k, v := range queryParams {
			q.Set(k, v)
		}
		reqURL.RawQuery = q.Encode()
	}

	// Build request.
	var bodyReader io.Reader
	if in.Body != "" {
		bodyReader = strings.NewReader(in.Body)
	}

	req, err := http.NewRequestWithContext(ctx, method, reqURL.String(), bodyReader)
	if err != nil {
		return nil, function.WrapExecutionError(f, "build request", err)
	}

	for k, v := range headers {
		req.Header.Set(k, v)
	}

	// Execute with SSRF-protected client.
	client := createHTTPClient(in.TimeoutSeconds, in.AllowPrivateIPs)
	resp, err := client.Do(req)
	if err != nil {
		return nil, function.WrapExecutionError(f, "execute request", err)
	}
	defer resp.Body.Close()

	// Read response body, capped at MaxResponseBytes.
	limitedReader := io.LimitReader(resp.Body, int64(in.MaxResponseBytes+1))
	rawBody, err := io.ReadAll(limitedReader)
	if err != nil {
		return nil, function.WrapExecutionError(f, "read response", err)
	}

	truncated := false
	if len(rawBody) > in.MaxResponseBytes {
		rawBody = rawBody[:in.MaxResponseBytes]
		truncated = true
	}

	// Flatten response headers (last value wins for duplicates).
	respHeaders := make(map[string]string, len(resp.Header))
	for k, vals := range resp.Header {
		if len(vals) > 0 {
			respHeaders[k] = vals[len(vals)-1]
		}
	}

	finalURL := resp.Request.URL.String()

	return &Output{
		StatusCode:  resp.StatusCode,
		Headers:     respHeaders,
		Body:        string(rawBody),
		BodyBytes:   len(rawBody),
		Truncated:   truncated,
		ContentType: resp.Header.Get("Content-Type"),
		FinalURL:    finalURL,
	}, nil
}

// createHTTPClient builds an http.Client with optional SSRF protection.
func createHTTPClient(timeoutSeconds int, allowPrivateIPs bool) *http.Client {
	baseDialer := &net.Dialer{
		Timeout:   30 * time.Second,
		KeepAlive: 30 * time.Second,
	}

	transport := &http.Transport{
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			host, port, err := net.SplitHostPort(addr)
			if err != nil {
				return nil, err
			}

			if !allowPrivateIPs {
				// If the host is already an IP literal, check it directly.
				if ip := net.ParseIP(host); ip != nil {
					if isPrivateIP(ip) {
						return nil, fmt.Errorf("private IP address not allowed: %s", host)
					}
					return baseDialer.DialContext(ctx, network, net.JoinHostPort(host, port))
				}

				// Otherwise resolve and check every resolved IP.
				ips, lookupErr := net.LookupHost(host)
				if lookupErr != nil {
					return nil, lookupErr
				}
				for _, ipStr := range ips {
					if ip := net.ParseIP(ipStr); ip != nil && isPrivateIP(ip) {
						return nil, fmt.Errorf("private IP address not allowed: %s resolves to %s", host, ipStr)
					}
				}
			}

			return baseDialer.DialContext(ctx, network, net.JoinHostPort(host, port))
		},
	}

	return &http.Client{
		Timeout:   time.Duration(timeoutSeconds) * time.Second,
		Transport: transport,
	}
}

// parseJSONStringMap parses a JSON string into map[string]string.
// Empty string → nil (no error). Invalid JSON → error.
func parseJSONStringMap(s, fieldName string) (map[string]string, error) {
	if s == "" {
		return nil, nil
	}
	var m map[string]string
	if err := json.Unmarshal([]byte(s), &m); err != nil {
		return nil, fmt.Errorf("field %s must be a JSON object string: %w", fieldName, err)
	}
	return m, nil
}

// parseMapInput converts a map[string]interface{} (CLI/LLM) into Input.
func parseMapInput(m map[string]interface{}, cfg function.Config) *Input {
	timeoutSecs := 30
	maxBytes := 102400
	if cfg != nil {
		if v := cfg.GetInt("http_request.timeout_seconds"); v > 0 {
			timeoutSecs = v
		}
		if v := cfg.GetInt("http_request.max_response_bytes"); v > 0 {
			maxBytes = v
		}
	}

	in := &Input{
		Method:           "GET",
		TimeoutSeconds:   timeoutSecs,
		MaxResponseBytes: maxBytes,
		AllowPrivateIPs:  false,
	}

	if cfg != nil && cfg.IsSet("http_request.allow_private_ips") {
		in.AllowPrivateIPs = cfg.GetBool("http_request.allow_private_ips")
	}

	if v, ok := m["url"].(string); ok {
		in.URL = v
	}
	if v, ok := m["method"].(string); ok {
		in.Method = v
	}
	if v, ok := m["headers"].(string); ok {
		in.Headers = v
	}
	if v, ok := m["query_params"].(string); ok {
		in.QueryParams = v
	}
	if v, ok := m["body"].(string); ok {
		in.Body = v
	}
	if v, ok := toHTTPInt(m["timeout_seconds"]); ok {
		in.TimeoutSeconds = v
	}
	if v, ok := toHTTPInt(m["max_response_bytes"]); ok {
		in.MaxResponseBytes = v
	}
	if v, ok := m["allow_private_ips"].(bool); ok {
		in.AllowPrivateIPs = v
	}
	if v, ok := m["spec_path"].(string); ok {
		in.SpecPath = v
	}
	if v, ok := m["spec_url"].(string); ok {
		in.SpecURL = v
	}
	if v, ok := m["operation_id"].(string); ok {
		in.OperationID = v
	}
	if v, ok := m["path_params"].(string); ok {
		in.PathParams = v
	}

	return in
}

func toHTTPInt(v interface{}) (int, bool) {
	switch n := v.(type) {
	case float64:
		return int(n), true
	case int:
		return n, true
	case int64:
		return int(n), true
	default:
		return 0, false
	}
}

// InputSchema implements AIFunction.
func (f *HTTPRequestFunction) InputSchema() *function.Schema {
	return schema.MustGenerateSchema(reflect.TypeOf(Input{}))
}

// Metadata implements AIFunction.
func (f *HTTPRequestFunction) Metadata() *function.Metadata {
	return function.NewMetadata("HTTPRequestFunction").
		WithName("http_request").
		WithDescription("Execute HTTP requests").
		WithToolName("http_request").
		WithToolDescription("Executes HTTP requests (GET, POST, PUT, PATCH, DELETE, etc.) and returns the response. Includes SSRF protection by default. Supports optional OpenAPI spec resolution via spec_path/spec_url + operation_id. Pass headers and query_params as JSON strings.").
		WithCLIName("http").
		WithCLIDescription("Execute an HTTP request").
		WithCategory("web").
		WithTags("http", "api", "request", "openapi").
		WithExamples(
			function.Example{
				Input:       map[string]interface{}{"url": "https://httpbin.org/get"},
				Description: "Simple GET request",
			},
			function.Example{
				Input: map[string]interface{}{
					"url":     "https://httpbin.org/post",
					"method":  "POST",
					"headers": `{"Content-Type":"application/json"}`,
					"body":    `{"key":"value"}`,
				},
				Description: "POST with JSON body",
			},
		)
}
