package websearch

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"
)

const (
	// FetchTimeout is the timeout for fetching a single page
	FetchTimeout = 10 // seconds
	// MaxContentBytes is the maximum size to read from a single page
	MaxContentBytes = 1024 * 1024 // 1MB
	// UserAgent is the user agent string for content fetching
	UserAgent = "Mozilla/5.0 (compatible; ai-sdk-go/1.0)"
)

// ContentFetcher handles fetching and extracting content from web pages.
type ContentFetcher struct {
	httpClient *http.Client
}

// NewContentFetcher creates a new content fetcher.
func NewContentFetcher() *ContentFetcher {
	return &ContentFetcher{
		httpClient: &http.Client{
			Timeout: FetchTimeout * time.Second,
		},
	}
}

// FetchContent fetches and extracts text content from a URL.
func (f *ContentFetcher) FetchContent(ctx context.Context, urlStr string) (*FetchedContent, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", urlStr, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("User-Agent", UserAgent)
	req.Header.Set("Accept", "text/html,application/xhtml+xml")

	resp, err := f.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetch failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	// Read with size limit
	limitedReader := io.LimitReader(resp.Body, MaxContentBytes)
	body, err := io.ReadAll(limitedReader)
	if err != nil {
		return nil, fmt.Errorf("failed to read content: %w", err)
	}

	html := string(body)

	// Extract text and title
	title := extractTitle(html)
	text := extractText(html)

	content := &FetchedContent{
		Title:     title,
		Text:      text,
		WordCount: countWords(text),
		FetchedAt: time.Now().Format(time.RFC3339),
	}

	return content, nil
}

// FetchMultiple fetches content for multiple URLs concurrently.
// Errors for individual fetches are stored in FetchedContent.Error.
func (f *ContentFetcher) FetchMultiple(ctx context.Context, urls []string) []*FetchedContent {
	results := make([]*FetchedContent, len(urls))

	type fetchResult struct {
		index   int
		content *FetchedContent
		err     error
	}

	resultCh := make(chan fetchResult, len(urls))

	// Launch goroutines for concurrent fetching
	for i, urlStr := range urls {
		go func(idx int, url string) {
			content, err := f.FetchContent(ctx, url)
			resultCh <- fetchResult{index: idx, content: content, err: err}
		}(i, urlStr)
	}

	// Collect results
	for range urls {
		res := <-resultCh
		if res.err != nil {
			// Store error in content
			results[res.index] = &FetchedContent{
				Error: res.err.Error(),
			}
		} else {
			results[res.index] = res.content
		}
	}

	return results
}

// extractTitle extracts the page title from HTML.
func extractTitle(html string) string {
	titlePattern := regexp.MustCompile(`(?i)<title[^>]*>([^<]+)</title>`)
	matches := titlePattern.FindStringSubmatch(html)
	if len(matches) >= 2 {
		return strings.TrimSpace(cleanHTML(matches[1]))
	}
	return ""
}

// extractText extracts plain text from HTML (simplified).
func extractText(html string) string {
	// Remove script and style tags (case-insensitive)
	html = regexp.MustCompile(`(?is)<script[^>]*>.*?</script>`).ReplaceAllString(html, "")
	html = regexp.MustCompile(`(?is)<style[^>]*>.*?</style>`).ReplaceAllString(html, "")

	// Remove HTML comments
	html = regexp.MustCompile(`(?s)<!--.*?-->`).ReplaceAllString(html, "")

	// Remove HTML tags
	text := regexp.MustCompile(`<[^>]*>`).ReplaceAllString(html, " ")

	// Decode HTML entities
	text = cleanHTML(text)

	// Normalize whitespace
	text = regexp.MustCompile(`\s+`).ReplaceAllString(text, " ")
	text = strings.TrimSpace(text)

	// Limit length (first 5000 characters)
	if len(text) > 5000 {
		text = text[:5000] + "..."
	}

	return text
}

// countWords counts words in text.
func countWords(text string) int {
	words := strings.Fields(text)
	return len(words)
}
