package websearch

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"
)

const (
	// DuckDuckGoEndpoint is the DuckDuckGo HTML search endpoint
	DuckDuckGoEndpoint = "https://html.duckduckgo.com/html/"
	// DDGDefaultTimeout is the default timeout for DuckDuckGo requests
	DDGDefaultTimeout = 10
	// DDGUserAgent is the user agent string to mimic a browser
	DDGUserAgent = "Mozilla/5.0 (compatible; ai-sdk-go/1.0)"
)

// DuckDuckGoProvider implements SearchProvider using DuckDuckGo HTML scraping.
// Note: This is fragile and may break if DuckDuckGo changes their HTML structure.
type DuckDuckGoProvider struct {
	httpClient *http.Client
	endpoint   string
}

// NewDuckDuckGoProvider creates a DuckDuckGo search provider.
// DuckDuckGo doesn't require an API key but uses HTML scraping.
func NewDuckDuckGoProvider(cfg *ProviderConfig) (*DuckDuckGoProvider, error) {
	timeout := DDGDefaultTimeout
	if cfg.Timeout > 0 {
		timeout = cfg.Timeout
	}

	endpoint := DuckDuckGoEndpoint
	if cfg.Endpoint != "" {
		endpoint = cfg.Endpoint
	}

	return &DuckDuckGoProvider{
		endpoint: endpoint,
		httpClient: &http.Client{
			Timeout: time.Duration(timeout) * time.Second,
		},
	}, nil
}

// Name returns the provider identifier.
func (p *DuckDuckGoProvider) Name() string {
	return "duckduckgo"
}

// RequiresAPIKey returns false since DuckDuckGo doesn't require an API key.
func (p *DuckDuckGoProvider) RequiresAPIKey() bool {
	return false
}

// Search performs a search using DuckDuckGo HTML scraping.
func (p *DuckDuckGoProvider) Search(ctx context.Context, params *SearchParams) ([]SearchResult, error) {
	// Build form data (DDG HTML search uses POST)
	formData := url.Values{}
	formData.Set("q", params.Query)

	if params.Region != "" {
		// DDG uses format like "us-en" for US English
		regionCode := strings.ToLower(params.Region)
		formData.Set("kl", regionCode+"-"+regionCode)
	}

	// Create request
	req, err := http.NewRequestWithContext(ctx, "POST", p.endpoint, strings.NewReader(formData.Encode()))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers to mimic browser
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("User-Agent", DDGUserAgent)

	// Execute request
	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("DDG returned status %d", resp.StatusCode)
	}

	// Read HTML response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Parse HTML to extract results
	results, err := p.parseHTMLResults(string(body), params.MaxResults)
	if err != nil {
		return nil, fmt.Errorf("failed to parse results: %w", err)
	}

	return results, nil
}

// parseHTMLResults extracts search results from DDG HTML.
// This is a simplified parser using regex patterns.
func (p *DuckDuckGoProvider) parseHTMLResults(html string, maxResults int) ([]SearchResult, error) {
	results := make([]SearchResult, 0)

	// Regex patterns for extracting result elements
	// DDG uses class="result__body" for result containers
	resultPattern := regexp.MustCompile(`(?s)<div[^>]*class="[^"]*result__body[^"]*"[^>]*>(.*?)</div>`)

	// Pattern for extracting links and titles
	// Looking for <a class="result__a" href="...">title</a>
	titlePattern := regexp.MustCompile(`(?s)<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>`)

	// Pattern for extracting snippets
	// Looking for <a class="result__snippet">snippet</a>
	snippetPattern := regexp.MustCompile(`(?s)<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>`)

	// Find all result blocks
	resultMatches := resultPattern.FindAllStringSubmatch(html, -1)

	for i, match := range resultMatches {
		if i >= maxResults {
			break
		}

		if len(match) < 2 {
			continue
		}

		block := match[1]

		// Extract title and URL
		titleMatch := titlePattern.FindStringSubmatch(block)
		if len(titleMatch) < 3 {
			continue
		}

		resultURL := titleMatch[1]
		title := titleMatch[2]

		// Extract snippet
		snippet := ""
		snippetMatch := snippetPattern.FindStringSubmatch(block)
		if len(snippetMatch) >= 2 {
			snippet = snippetMatch[1]
		}

		result := SearchResult{
			Rank:    i + 1,
			Title:   cleanHTML(title),
			URL:     cleanHTML(resultURL),
			Snippet: cleanHTML(snippet),
			Source:  extractDomain(resultURL),
		}

		results = append(results, result)
	}

	if len(results) == 0 {
		return []SearchResult{}, nil // Return empty array, not an error
	}

	return results, nil
}

// cleanHTML removes HTML entities and tags.
func cleanHTML(s string) string {
	// Remove HTML tags
	s = regexp.MustCompile(`<[^>]*>`).ReplaceAllString(s, "")

	// Decode common HTML entities
	replacements := map[string]string{
		"&amp;":  "&",
		"&lt;":   "<",
		"&gt;":   ">",
		"&quot;": "\"",
		"&#39;":  "'",
		"&nbsp;": " ",
		"&#x27;": "'",
		"&apos;": "'",
	}

	for entity, char := range replacements {
		s = strings.ReplaceAll(s, entity, char)
	}

	return strings.TrimSpace(s)
}
