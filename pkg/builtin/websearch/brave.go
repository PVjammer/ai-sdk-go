package websearch

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const (
	// BraveSearchAPIEndpoint is the Brave Search API endpoint
	BraveSearchAPIEndpoint = "https://api.search.brave.com/res/v1/web/search"
	// BraveDefaultTimeout is the default timeout for Brave API requests
	BraveDefaultTimeout = 10 // seconds
)

// BraveSearchProvider implements SearchProvider for Brave Search API.
type BraveSearchProvider struct {
	apiKey     string
	httpClient *http.Client
	endpoint   string
}

// NewBraveSearchProvider creates a Brave Search provider.
func NewBraveSearchProvider(cfg *ProviderConfig) (*BraveSearchProvider, error) {
	if cfg.APIKey == "" {
		return nil, ErrAPIKeyRequired
	}

	timeout := BraveDefaultTimeout
	if cfg.Timeout > 0 {
		timeout = cfg.Timeout
	}

	endpoint := BraveSearchAPIEndpoint
	if cfg.Endpoint != "" {
		endpoint = cfg.Endpoint
	}

	return &BraveSearchProvider{
		apiKey:   cfg.APIKey,
		endpoint: endpoint,
		httpClient: &http.Client{
			Timeout: time.Duration(timeout) * time.Second,
		},
	}, nil
}

// Name returns the provider identifier.
func (p *BraveSearchProvider) Name() string {
	return "brave"
}

// RequiresAPIKey returns true since Brave requires an API key.
func (p *BraveSearchProvider) RequiresAPIKey() bool {
	return true
}

// Search performs a search using the Brave Search API.
func (p *BraveSearchProvider) Search(ctx context.Context, params *SearchParams) ([]SearchResult, error) {
	// Build query parameters
	queryParams := url.Values{}
	queryParams.Set("q", params.Query)
	queryParams.Set("count", fmt.Sprintf("%d", params.MaxResults))

	if params.SafeSearch {
		queryParams.Set("safesearch", "strict")
	}

	if params.Language != "" {
		queryParams.Set("search_lang", params.Language)
	}

	if params.Region != "" {
		queryParams.Set("country", strings.ToUpper(params.Region))
	}

	if params.Freshness != "" && params.Freshness != "all" {
		queryParams.Set("freshness", params.Freshness)
	}

	// Build request
	requestURL := fmt.Sprintf("%s?%s", p.endpoint, queryParams.Encode())
	req, err := http.NewRequestWithContext(ctx, "GET", requestURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	req.Header.Set("Accept", "application/json")
	req.Header.Set("X-Subscription-Token", p.apiKey)

	// Execute request
	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	// Handle response codes
	switch resp.StatusCode {
	case 429:
		return nil, ErrRateLimitExceeded
	case 401, 403:
		return nil, ErrAPIKeyRequired
	case 200:
		// Success, continue processing
	default:
		return nil, fmt.Errorf("API returned status %d", resp.StatusCode)
	}

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Parse response
	var apiResp braveAPIResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Convert to SearchResult
	results := p.convertResults(&apiResp)

	if len(results) == 0 {
		return []SearchResult{}, nil // Return empty array, not an error
	}

	return results, nil
}

// braveAPIResponse matches Brave Search API response structure.
type braveAPIResponse struct {
	Web struct {
		Results []struct {
			Title       string `json:"title"`
			URL         string `json:"url"`
			Description string `json:"description"`
			Age         string `json:"age,omitempty"`
		} `json:"results"`
	} `json:"web"`
}

// convertResults converts Brave API response to SearchResult slice.
func (p *BraveSearchProvider) convertResults(apiResp *braveAPIResponse) []SearchResult {
	results := make([]SearchResult, 0, len(apiResp.Web.Results))

	for i, res := range apiResp.Web.Results {
		result := SearchResult{
			Rank:        i + 1,
			Title:       res.Title,
			URL:         res.URL,
			Snippet:     res.Description,
			Source:      extractDomain(res.URL),
			PublishedAt: res.Age,
		}
		results = append(results, result)
	}

	return results
}

// extractDomain extracts the domain from a URL.
func extractDomain(urlStr string) string {
	u, err := url.Parse(urlStr)
	if err != nil {
		return ""
	}
	return u.Host
}
