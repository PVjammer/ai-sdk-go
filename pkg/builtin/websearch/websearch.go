// Package websearch provides web search functionality using various backend providers.
//
// This package implements web search using multiple backends (Brave Search API, DuckDuckGo)
// with optional content fetching from result pages. It supports configuration-based provider
// selection with automatic fallback.
package websearch

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"strings"
	"time"

	"github.com/pvjammer/ai-sdk-go/internal/schema"
	"github.com/pvjammer/ai-sdk-go/internal/validation"
	"github.com/pvjammer/ai-sdk-go/pkg/function"
)

// WebSearchFunction performs web searches using various backends.
type WebSearchFunction struct {
	function.Base
	provider SearchProvider
	fetcher  *ContentFetcher
}

// Input defines the input schema for web search.
type Input struct {
	Query        string `json:"query" validate:"required,min=1" description:"Search query"`
	MaxResults   int    `json:"max_results" validate:"min=1,max=20" description:"Maximum number of results to return" default:"5"`
	Backend      string `json:"backend" validate:"oneof=brave duckduckgo auto" enum:"brave,duckduckgo,auto" description:"Search backend to use" default:"auto"`
	FetchContent bool   `json:"fetch_content" description:"Fetch and extract content from result pages" default:"false"`
	ContentCount int    `json:"content_count" validate:"min=0,max=10" description:"Number of top results to fetch content for (requires fetch_content=true)" default:"3"`
	SafeSearch   bool   `json:"safe_search" description:"Enable safe search filtering" default:"true"`
	Language     string `json:"language" description:"Search language preference (e.g., en, es, fr)" default:"en"`
	Region       string `json:"region" description:"Search region/country code (e.g., US, GB, DE)" default:"US"`
	Freshness    string `json:"freshness" validate:"oneof=day week month year all" enum:"day,week,month,year,all" description:"Time range for results" default:"all"`
}

// Output defines the output structure for web search results.
type Output struct {
	Results      []SearchResult `json:"results"`
	Query        string         `json:"query"`
	Backend      string         `json:"backend_used"`
	TotalResults int            `json:"total_results"`
	Timestamp    string         `json:"timestamp"`
	FetchErrors  []string       `json:"fetch_errors,omitempty"`
}

// SearchResult represents a single search result.
type SearchResult struct {
	Rank        int             `json:"rank"`
	Title       string          `json:"title"`
	URL         string          `json:"url"`
	Snippet     string          `json:"snippet"`
	Content     *FetchedContent `json:"content,omitempty"`
	Source      string          `json:"source,omitempty"`
	PublishedAt string          `json:"published_at,omitempty"`
}

// FetchedContent represents content fetched from a result page.
type FetchedContent struct {
	Text      string `json:"text"`
	Title     string `json:"title"`
	WordCount int    `json:"word_count"`
	FetchedAt string `json:"fetched_at"`
	Error     string `json:"error,omitempty"`
}

// New creates a new WebSearchFunction with configured provider.
func New(cfg function.Config) function.AIFunction {
	provider := createProvider(cfg)

	return &WebSearchFunction{
		Base:     function.NewBase(cfg),
		provider: provider,
		fetcher:  NewContentFetcher(),
	}
}

// NewDefault creates a new WebSearchFunction with default provider.
func NewDefault() *WebSearchFunction {
	provider := createDefaultProvider()

	return &WebSearchFunction{
		Base:     function.NewBase(function.NilConfig{}),
		provider: provider,
		fetcher:  NewContentFetcher(),
	}
}

// NewWithProvider creates a new WebSearchFunction with a specific provider.
// Useful for testing with mock providers.
func NewWithProvider(provider SearchProvider) *WebSearchFunction {
	return &WebSearchFunction{
		Base:     function.NewBase(function.NilConfig{}),
		provider: provider,
		fetcher:  NewContentFetcher(),
	}
}

// createProvider creates a search provider based on configuration.
func createProvider(cfg function.Config) SearchProvider {
	backend := cfg.GetString("websearch.backend")

	// Try Brave first if API key available
	if backend == "" || backend == "brave" || backend == "auto" {
		apiKey := cfg.GetString("websearch.brave.api_key")
		if apiKey == "" {
			// Try environment variable
			apiKey = os.Getenv("BRAVE_API_KEY")
		}

		if apiKey != "" {
			timeout := cfg.GetInt("websearch.brave.timeout")
			provider, err := NewBraveSearchProvider(&ProviderConfig{
				APIKey:  apiKey,
				Timeout: timeout,
			})
			if err == nil {
				return provider
			}
		}
	}

	// Fall back to DuckDuckGo (no API key required)
	if backend == "duckduckgo" || backend == "auto" || backend == "" {
		timeout := cfg.GetInt("websearch.duckduckgo.timeout")
		provider, err := NewDuckDuckGoProvider(&ProviderConfig{
			Timeout: timeout,
		})
		if err == nil {
			return provider
		}
	}

	// Last resort: return DuckDuckGo with default config
	provider, _ := NewDuckDuckGoProvider(&ProviderConfig{})
	return provider
}

// createDefaultProvider creates a provider when no config is available.
func createDefaultProvider() SearchProvider {
	// Try environment variable for Brave API key
	apiKey := os.Getenv("BRAVE_API_KEY")
	if apiKey != "" {
		provider, err := NewBraveSearchProvider(&ProviderConfig{
			APIKey: apiKey,
		})
		if err == nil {
			return provider
		}
	}

	// Default to DuckDuckGo
	provider, _ := NewDuckDuckGoProvider(&ProviderConfig{})
	return provider
}

// Execute implements the AIFunction interface.
func (f *WebSearchFunction) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	// Type assert and validate
	var in *Input

	switch v := input.(type) {
	case *Input:
		in = v
	case map[string]interface{}:
		in = f.parseMapInput(v)
	default:
		return nil, function.NewExecutionError(f, "input validation", "invalid input type")
	}

	// Validate input
	if err := validation.Validate(in); err != nil {
		return nil, function.WrapValidationError(f, err)
	}

	// Additional validation: ContentCount should not exceed MaxResults
	if in.FetchContent && in.ContentCount > in.MaxResults {
		in.ContentCount = in.MaxResults
	}

	// Check context for cancellation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Perform search
	params := &SearchParams{
		Query:      in.Query,
		MaxResults: in.MaxResults,
		SafeSearch: in.SafeSearch,
		Language:   in.Language,
		Region:     in.Region,
		Freshness:  in.Freshness,
	}

	results, err := f.provider.Search(ctx, params)
	if err != nil {
		return nil, function.WrapExecutionError(f, "search", err)
	}

	// Build output
	output := &Output{
		Results:      results,
		Query:        in.Query,
		Backend:      f.provider.Name(),
		TotalResults: len(results),
		Timestamp:    time.Now().Format(time.RFC3339),
		FetchErrors:  []string{},
	}

	// Fetch content if requested
	if in.FetchContent && in.ContentCount > 0 {
		count := in.ContentCount
		if count > len(results) {
			count = len(results)
		}

		urls := make([]string, count)
		for i := 0; i < count; i++ {
			urls[i] = results[i].URL
		}

		contents := f.fetcher.FetchMultiple(ctx, urls)

		for i, content := range contents {
			output.Results[i].Content = content
			if content.Error != "" {
				output.FetchErrors = append(output.FetchErrors,
					fmt.Sprintf("Failed to fetch %s: %s", urls[i], content.Error))
			}
		}
	}

	return output, nil
}

// toInt converts a value to int, handling float64 (from JSON), int, and int64.
// Returns the converted int and true on success, or 0 and false on failure.
func toInt(v interface{}) (int, bool) {
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

// parseMapInput parses map[string]interface{} input from CLI or JSON-decoded tool calls.
// Priority: hardcoded defaults < config (Viper) < explicitly-set per-call args.
// The "_changed_flags" key (set by buildInputFromFlags) distinguishes user-set
// flags from Cobra defaults so config values are not masked by schema defaults.
func (f *WebSearchFunction) parseMapInput(m map[string]interface{}) *Input {
	// Determine which keys were explicitly set by the caller
	changed, _ := m["_changed_flags"].(map[string]bool)

	// Start with hardcoded defaults
	in := &Input{
		MaxResults:   5,
		Backend:      "auto",
		FetchContent: false,
		ContentCount: 3,
		SafeSearch:   true,
		Language:     "en",
		Region:       "US",
		Freshness:    "all",
	}

	// Layer config-driven defaults (overrides hardcoded)
	cfg := f.Config()
	if cfg != nil {
		if v := cfg.GetInt("websearch.max_results"); v > 0 {
			in.MaxResults = v
		}
		if v := cfg.GetString("websearch.backend"); v != "" {
			in.Backend = v
		}
		// Use IsSet for booleans to distinguish "unset" from "set to false"
		if cfg.IsSet("websearch.fetch_content") {
			in.FetchContent = cfg.GetBool("websearch.fetch_content")
		}
		if cfg.IsSet("websearch.safe_search") {
			in.SafeSearch = cfg.GetBool("websearch.safe_search")
		}
	}

	// Layer per-call values — only override if the key was explicitly set
	// (or if there's no _changed_flags metadata, e.g. from tool calls)
	isSet := func(key string) bool {
		return changed == nil || changed[key]
	}

	if query, ok := m["query"].(string); ok && isSet("query") {
		in.Query = query
	}
	if maxResults, ok := toInt(m["max_results"]); ok && isSet("max_results") {
		in.MaxResults = maxResults
	}
	if backend, ok := m["backend"].(string); ok && isSet("backend") {
		in.Backend = backend
	}
	if fetchContent, ok := m["fetch_content"].(bool); ok && isSet("fetch_content") {
		in.FetchContent = fetchContent
	}
	if contentCount, ok := toInt(m["content_count"]); ok && isSet("content_count") {
		in.ContentCount = contentCount
	}
	if safeSearch, ok := m["safe_search"].(bool); ok && isSet("safe_search") {
		in.SafeSearch = safeSearch
	}
	if language, ok := m["language"].(string); ok && isSet("language") {
		in.Language = language
	}
	if region, ok := m["region"].(string); ok && isSet("region") {
		in.Region = region
	}
	if freshness, ok := m["freshness"].(string); ok && isSet("freshness") {
		in.Freshness = freshness
	}

	return in
}

// InputSchema implements the AIFunction interface.
func (f *WebSearchFunction) InputSchema() *function.Schema {
	return schema.MustGenerateSchema(reflect.TypeOf(Input{}))
}

// Metadata implements the AIFunction interface.
func (f *WebSearchFunction) Metadata() *function.Metadata {
	return function.NewMetadata("WebSearchFunction").
		WithName("websearch").
		WithDescription("Search the web using Brave or DuckDuckGo").
		WithToolName("web_search").
		WithToolDescription("Searches the web for information using Brave Search API or DuckDuckGo. Returns structured results with titles, URLs, and snippets. Can optionally fetch and extract content from result pages.").
		WithCLIName("search").
		WithCLIDescription("Search the web for information").
		WithCategory("web").
		WithTags("web", "search", "internet", "information", "research").
		WithExamples(
			function.Example{
				Input: map[string]interface{}{
					"query":       "Go programming language",
					"max_results": 5,
				},
				Description: "Basic web search",
			},
			function.Example{
				Input: map[string]interface{}{
					"query":         "artificial intelligence news",
					"max_results":   10,
					"backend":       "brave",
					"fetch_content": true,
					"content_count": 3,
				},
				Description: "Search with content fetching",
			},
		)
}

// FormatMarkdown returns results formatted as markdown.
func (o *Output) FormatMarkdown() string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("# Search Results for: %s\n\n", o.Query))
	sb.WriteString(fmt.Sprintf("**Backend:** %s  \n", o.Backend))
	sb.WriteString(fmt.Sprintf("**Results:** %d  \n", o.TotalResults))
	sb.WriteString(fmt.Sprintf("**Timestamp:** %s\n\n", o.Timestamp))

	for _, result := range o.Results {
		sb.WriteString(fmt.Sprintf("## %d. %s\n\n", result.Rank, result.Title))
		sb.WriteString(fmt.Sprintf("**URL:** [%s](%s)  \n", result.URL, result.URL))
		if result.Source != "" {
			sb.WriteString(fmt.Sprintf("**Source:** %s  \n", result.Source))
		}
		if result.PublishedAt != "" {
			sb.WriteString(fmt.Sprintf("**Published:** %s  \n", result.PublishedAt))
		}
		sb.WriteString(fmt.Sprintf("\n%s\n\n", result.Snippet))

		if result.Content != nil && result.Content.Error == "" {
			sb.WriteString(fmt.Sprintf("**Content Preview** (%d words):\n\n", result.Content.WordCount))
			// Show first 500 characters of content
			preview := result.Content.Text
			if len(preview) > 500 {
				preview = preview[:500] + "..."
			}
			sb.WriteString(fmt.Sprintf("%s\n\n", preview))
		}

		sb.WriteString("---\n\n")
	}

	if len(o.FetchErrors) > 0 {
		sb.WriteString("## Fetch Errors\n\n")
		for _, err := range o.FetchErrors {
			sb.WriteString(fmt.Sprintf("- %s\n", err))
		}
	}

	return sb.String()
}

// FormatJSON returns results as pretty-printed JSON.
func (o *Output) FormatJSON() (string, error) {
	data, err := json.MarshalIndent(o, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}
