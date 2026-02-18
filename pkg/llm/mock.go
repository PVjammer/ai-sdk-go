package llm

import (
	"context"
	"fmt"
	"strings"
)

// MockProvider is a mock LLM provider for testing.
//
// It returns predefined responses in sequence and tracks calls for verification.
// Useful for testing code that depends on LLM providers without making actual
// API calls or running local models.
type MockProvider struct {
	responses  []string
	callCount  int
	lastPrompt string
	lastOpts   *CompletionOptions
	failAfter  int // Fail after N calls (0 = never fail)
}

// NewMockProvider creates a mock provider with predefined responses.
//
// Each call to Complete() returns the next response in sequence.
// If all responses are exhausted, returns an error.
func NewMockProvider(responses ...string) *MockProvider {
	return &MockProvider{
		responses: responses,
		callCount: 0,
		failAfter: 0,
	}
}

// NewMockProviderWithFailure creates a mock that fails after N calls.
func NewMockProviderWithFailure(failAfter int, responses ...string) *MockProvider {
	return &MockProvider{
		responses: responses,
		callCount: 0,
		failAfter: failAfter,
	}
}

// Complete returns the next predefined response.
//
// DEPRECATED: Use ainvoke() for new code.
func (m *MockProvider) Complete(ctx context.Context, prompt string, opts *CompletionOptions) (*Response, error) {
	// Check context
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Track call
	m.lastPrompt = prompt
	m.lastOpts = opts
	m.callCount++

	// Check if should fail
	if m.failAfter > 0 && m.callCount > m.failAfter {
		return nil, fmt.Errorf("mock provider: simulated failure after %d calls", m.failAfter)
	}

	// Check if responses exhausted
	if m.callCount > len(m.responses) {
		return nil, fmt.Errorf("mock provider: no more responses (called %d times, have %d responses)", m.callCount, len(m.responses))
	}

	// Return next response
	response := m.responses[m.callCount-1]

	// Estimate tokens (rough: 1 token ≈ 4 characters)
	promptTokens := len(prompt) / 4
	completionTokens := len(response) / 4

	return &Response{
		Text:             response,
		TokensUsed:       promptTokens + completionTokens,
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		Model:            "mock",
		FinishReason:     "stop",
		Provider:         "mock",
	}, nil
}

// Ainvoke generates a streaming response with predefined text.
//
// If opts.Stream is true, the response is split into simulated chunks.
// If opts.Stream is false, a single chunk with the complete response is sent.
func (m *MockProvider) Ainvoke(ctx context.Context, prompt string, opts *CompletionOptions) <-chan StreamChunk {
	ch := make(chan StreamChunk, 10)

	go func() {
		defer close(ch)

		// Track call
		m.lastPrompt = prompt
		m.lastOpts = opts
		m.callCount++

		// Check if should fail
		if m.failAfter > 0 && m.callCount > m.failAfter {
			ch <- StreamChunk{
				Error: fmt.Errorf("mock provider: simulated failure after %d calls", m.failAfter),
				Done:  true,
			}
			return
		}

		// Check if responses exhausted
		if m.callCount > len(m.responses) {
			ch <- StreamChunk{
				Error: fmt.Errorf("mock provider: no more responses (called %d times, have %d responses)", m.callCount, len(m.responses)),
				Done:  true,
			}
			return
		}

		// Get next response
		response := m.responses[m.callCount-1]

		// Estimate tokens (rough: 1 token ≈ 4 characters)
		promptTokens := len(prompt) / 4
		completionTokens := len(response) / 4
		totalTokens := promptTokens + completionTokens

		// Check context before sending
		select {
		case <-ctx.Done():
			ch <- StreamChunk{
				Error: ctx.Err(),
				Done:  true,
			}
			return
		default:
		}

		if opts != nil && opts.Stream {
			// Stream mode: split response into chunks (word by word)
			words := strings.Fields(response)
			for i, word := range words {
				// Check context before each chunk
				select {
				case <-ctx.Done():
					ch <- StreamChunk{
						Error: ctx.Err(),
						Done:  true,
					}
					return
				default:
				}

				// Add space before word (except first)
				text := word
				if i > 0 {
					text = " " + word
				}

				// Send chunk
				isLast := i == len(words)-1
				chunk := StreamChunk{
					Text: text,
					Done: isLast,
				}

				// Add metadata on final chunk
				if isLast {
					chunk.TokensUsed = totalTokens
					chunk.PromptTokens = promptTokens
					chunk.CompletionTokens = completionTokens
					chunk.Model = "mock"
					chunk.FinishReason = "stop"
				}

				ch <- chunk
			}
		} else {
			// Non-streaming mode: send complete response in one chunk
			ch <- StreamChunk{
				Text:             response,
				Done:             true,
				TokensUsed:       totalTokens,
				PromptTokens:     promptTokens,
				CompletionTokens: completionTokens,
				Model:            "mock",
				FinishReason:     "stop",
			}
		}
	}()

	return ch
}

// Name returns the provider identifier.
func (m *MockProvider) Name() string {
	return "mock"
}

// MaxTokens returns the mock maximum context size.
func (m *MockProvider) MaxTokens() int {
	return 4096
}

// CallCount returns the number of times Complete was called.
func (m *MockProvider) CallCount() int {
	return m.callCount
}

// LastPrompt returns the prompt from the most recent call.
func (m *MockProvider) LastPrompt() string {
	return m.lastPrompt
}

// LastOptions returns the options from the most recent call.
func (m *MockProvider) LastOptions() *CompletionOptions {
	return m.lastOpts
}

// Reset resets the call counter and clears tracked data.
func (m *MockProvider) Reset() {
	m.callCount = 0
	m.lastPrompt = ""
	m.lastOpts = nil
}

// VerifyPromptContains checks if the last prompt contains the given substring.
func (m *MockProvider) VerifyPromptContains(substr string) bool {
	return strings.Contains(m.lastPrompt, substr)
}

// VerifySystemPrompt checks if the last system prompt matches.
func (m *MockProvider) VerifySystemPrompt(expected string) bool {
	if m.lastOpts == nil {
		return expected == ""
	}
	return m.lastOpts.SystemPrompt == expected
}

// VerifyTemperature checks if the last temperature matches.
func (m *MockProvider) VerifyTemperature(expected float64) bool {
	if m.lastOpts == nil {
		return false
	}
	return m.lastOpts.Temperature == expected
}
