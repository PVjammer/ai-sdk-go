// Package websearch provides web search functionality using various backend providers.
package websearch

import (
	"context"
	"errors"
)

// SearchProvider defines the interface for web search backends.
type SearchProvider interface {
	// Search performs a web search with the given parameters.
	// Returns a slice of SearchResult or an error.
	Search(ctx context.Context, params *SearchParams) ([]SearchResult, error)

	// Name returns the provider identifier (e.g., "brave", "duckduckgo")
	Name() string

	// RequiresAPIKey returns true if this provider requires an API key
	RequiresAPIKey() bool
}

// SearchParams encapsulates search parameters for providers.
type SearchParams struct {
	Query      string
	MaxResults int
	SafeSearch bool
	Language   string
	Region     string
	Freshness  string
}

// ProviderConfig holds provider-specific configuration.
type ProviderConfig struct {
	APIKey   string
	Endpoint string // Optional custom endpoint
	Timeout  int    // Timeout in seconds
}

// Common errors for search operations
var (
	ErrProviderUnavailable = errors.New("search provider unavailable")
	ErrAPIKeyRequired      = errors.New("API key required but not provided")
	ErrRateLimitExceeded   = errors.New("rate limit exceeded")
	ErrInvalidQuery        = errors.New("invalid search query")
	ErrNoResults           = errors.New("no results found")
)
