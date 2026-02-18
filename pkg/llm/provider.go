// Package llm provides abstractions for Large Language Model providers.
//
// This package defines interfaces and implementations for interacting with
// various LLM backends including local models (Ollama) and API-based services
// (Anthropic, OpenAI).
package llm

import (
	"context"
	"fmt"
)

// StreamChunk represents a chunk of streaming LLM response.
//
// Chunks are sent over a channel as the LLM generates text.
// The Done field indicates the final chunk.
type StreamChunk struct {
	// Text is the generated text in this chunk.
	Text string

	// Done indicates this is the final chunk.
	Done bool

	// TokensUsed is the total tokens consumed (only set on final chunk).
	TokensUsed int

	// PromptTokens is the number of input tokens (only set on final chunk).
	PromptTokens int

	// CompletionTokens is the number of output tokens (only set on final chunk).
	CompletionTokens int

	// Model is the actual model used (only set on final chunk).
	Model string

	// FinishReason indicates why generation stopped (only set on final chunk).
	FinishReason string

	// Error contains any error that occurred during generation.
	// If set, this is typically the final chunk.
	Error error
}

// Provider defines the interface for LLM backends.
//
// Implementations should support text completion with configurable parameters.
// The interface is designed to be simple and provider-agnostic, focusing on
// text-in, text-out completion.
type Provider interface {
	// Complete generates a text completion based on the prompt and options.
	//
	// The context parameter should be respected for cancellation and timeouts.
	// Returns the completion response or an error if generation fails.
	//
	// DEPRECATED: Use ainvoke() for new code. This method will be removed in a future version.
	Complete(ctx context.Context, prompt string, opts *CompletionOptions) (*Response, error)

	// Ainvoke generates a text completion and streams the response over a channel.
	//
	// The returned channel will receive StreamChunks as the LLM generates text.
	// The channel will be closed when generation is complete or an error occurs.
	//
	// If opts.Stream is false, a single chunk with the complete response will be sent.
	// If opts.Stream is true, incremental chunks will be sent as they're generated.
	//
	// The context parameter should be respected for cancellation and timeouts.
	// If the context is cancelled, the channel will be closed and the final chunk
	// will contain a context.Canceled error.
	Ainvoke(ctx context.Context, prompt string, opts *CompletionOptions) <-chan StreamChunk

	// Name returns a human-readable identifier for this provider.
	// Format: "provider:model" (e.g., "ollama:llama3.2", "anthropic:claude-3-5-haiku")
	Name() string

	// MaxTokens returns the maximum context length supported by this provider.
	// This includes both input and output tokens.
	MaxTokens() int
}

// ChatMessage represents a message in a multi-turn conversation.
//
// Unlike the single-prompt approach, this preserves role information
// for each message, allowing the LLM to properly interpret conversation context.
type ChatMessage struct {
	// Role identifies the message sender: "system", "user", "assistant", or "tool"
	Role string

	// Content is the message text
	Content string
}

// ConversationProvider extends Provider with multi-message conversation support.
//
// When a provider implements this interface, planners can send full conversation
// history with proper role attribution instead of packing everything into a
// single user prompt. This is critical for tool-calling flows where the LLM
// needs to see tool results as separate messages.
type ConversationProvider interface {
	Provider

	// ChatMessages sends a multi-turn conversation to the LLM and streams the response.
	//
	// Messages should include proper role attribution (system, user, assistant, tool).
	// The opts parameter configures generation parameters (temperature, max tokens, etc.).
	// The SystemPrompt field in opts is ignored when using this method — include system
	// messages directly in the messages slice.
	ChatMessages(ctx context.Context, messages []ChatMessage, opts *CompletionOptions) <-chan StreamChunk
}

// CompletionOptions configures the LLM completion request.
type CompletionOptions struct {
	// MaxTokens limits the number of tokens in the response.
	// If 0, uses provider default.
	MaxTokens int

	// Temperature controls randomness in generation.
	// Range: 0.0 (deterministic) to 2.0 (very random)
	// Typical: 0.0-1.0
	Temperature float64

	// StopWords are sequences that will stop generation when encountered.
	StopWords []string

	// SystemPrompt is prepended to the conversation as a system message.
	// Used to set context, tone, and instructions.
	SystemPrompt string

	// TopP controls nucleus sampling (alternative to temperature).
	// Range: 0.0-1.0. If 0, not used.
	TopP float64

	// TopK limits the number of tokens to consider at each step.
	// If 0, not used.
	TopK int

	// Stream controls whether to stream incremental responses (true) or
	// return a single complete response (false).
	// Default: false
	Stream bool
}

// Response contains the result of an LLM completion.
type Response struct {
	// Text is the generated completion text.
	Text string

	// TokensUsed is the total number of tokens consumed (input + output).
	TokensUsed int

	// PromptTokens is the number of input tokens.
	PromptTokens int

	// CompletionTokens is the number of output tokens.
	CompletionTokens int

	// Model is the actual model used (may differ from requested).
	Model string

	// FinishReason indicates why generation stopped.
	// Common values: "stop", "length", "error"
	FinishReason string

	// Provider is the provider that generated this response.
	Provider string
}

// Error types for LLM operations.
var (
	// ErrProviderUnavailable indicates the LLM provider cannot be reached.
	ErrProviderUnavailable = fmt.Errorf("llm provider unavailable")

	// ErrContextLengthExceeded indicates the input exceeds model context limit.
	ErrContextLengthExceeded = fmt.Errorf("context length exceeded")

	// ErrInvalidResponse indicates the provider returned an unexpected response.
	ErrInvalidResponse = fmt.Errorf("invalid provider response")

	// ErrRateLimitExceeded indicates rate limiting by the provider.
	ErrRateLimitExceeded = fmt.Errorf("rate limit exceeded")

	// ErrAuthenticationFailed indicates invalid API credentials.
	ErrAuthenticationFailed = fmt.Errorf("authentication failed")
)

// DefaultOptions returns sensible defaults for completion options.
func DefaultOptions() *CompletionOptions {
	return &CompletionOptions{
		MaxTokens:   2048,
		Temperature: 0.7,
		StopWords:   []string{},
		TopP:        0.0, // Not used by default
		TopK:        0,   // Not used by default
	}
}

// WithMaxTokens returns a new CompletionOptions with MaxTokens set.
func (o *CompletionOptions) WithMaxTokens(tokens int) *CompletionOptions {
	o.MaxTokens = tokens
	return o
}

// WithTemperature returns a new CompletionOptions with Temperature set.
func (o *CompletionOptions) WithTemperature(temp float64) *CompletionOptions {
	o.Temperature = temp
	return o
}

// WithSystemPrompt returns a new CompletionOptions with SystemPrompt set.
func (o *CompletionOptions) WithSystemPrompt(prompt string) *CompletionOptions {
	o.SystemPrompt = prompt
	return o
}

// WithStopWords returns a new CompletionOptions with StopWords set.
func (o *CompletionOptions) WithStopWords(words ...string) *CompletionOptions {
	o.StopWords = words
	return o
}
