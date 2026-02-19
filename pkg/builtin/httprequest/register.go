package httprequest

import (
	"github.com/pvjammer/ai-sdk-go/pkg/registry"
)

func init() {
	reg := registry.Global()

	reg.RegisterWithParams("http_request", New, []registry.ConfigParam{
		{Key: "timeout_seconds", Type: "int", Default: 30, Description: "Default request timeout in seconds"},
		{Key: "max_response_bytes", Type: "int", Default: 102400, Description: "Default maximum response body bytes (100 KB)"},
		{Key: "allow_private_ips", Type: "bool", Default: false, Description: "Allow requests to private/local IP ranges (false = SSRF protection on)"},
	})

	// Network requests require user approval before the agent fires them.
	reg.SetRequiresApproval("http_request", true)
}
