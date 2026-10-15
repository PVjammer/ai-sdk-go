package llm

// Usage reports token accounting for one LLM call. A distinct type from StreamChunk/Response so
// other APIs (like hooks.CompactionResult) can report usage without depending on StreamChunk's
// full streaming-specific shape.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

// UsageFromChunk extracts Usage from a StreamChunk's token fields.
func UsageFromChunk(c StreamChunk) Usage {
	return Usage{PromptTokens: c.PromptTokens, CompletionTokens: c.CompletionTokens, TotalTokens: c.TokensUsed}
}

// UsageFromResponse extracts Usage from a Response's token fields. A nil Response returns the
// zero Usage.
func UsageFromResponse(r *Response) Usage {
	if r == nil {
		return Usage{}
	}
	return Usage{PromptTokens: r.PromptTokens, CompletionTokens: r.CompletionTokens, TotalTokens: r.TokensUsed}
}
