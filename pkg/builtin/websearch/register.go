package websearch

import (
	"github.com/pvjammer/ai-sdk-go/pkg/registry"
)

func init() {
	registry.Global().RegisterWithParams("websearch", New, []registry.ConfigParam{
		{Key: "backend", Type: "string", Default: "auto", Description: "Search backend", Options: []string{"auto", "brave", "duckduckgo"}},
		{Key: "max_results", Type: "int", Default: 5, Description: "Maximum number of results"},
		{Key: "fetch_content", Type: "bool", Default: false, Description: "Fetch and extract content from results"},
		{Key: "safe_search", Type: "bool", Default: true, Description: "Enable safe search filtering"},
	})
}
