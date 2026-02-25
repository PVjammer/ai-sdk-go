package llm

import (
	"context"
	"fmt"
)

// MockToolCallingResponse configures one response from MockToolCallingProvider.
// Set ToolCalls to simulate a tool-calling response; set Text for a final answer.
type MockToolCallingResponse struct {
	Text      string
	ToolCalls []ToolCall
	Error     error
}

// MockToolCallingProvider is a mock ToolCallingProvider for testing agent loops.
//
// It returns pre-configured responses in sequence via ChatWithTools, allowing
// tests to simulate multi-turn tool-calling conversations without a real LLM.
//
// Use LastMessages to inspect the exact message slice the agent passed on the
// most recent call — useful for asserting that PreModel hooks trimmed history.
type MockToolCallingProvider struct {
	responses    []MockToolCallingResponse
	callCount    int
	LastMessages []ChatMessage // messages from the most recent ChatWithTools call
}

// NewMockToolCallingProvider creates a provider that returns responses in sequence.
func NewMockToolCallingProvider(responses ...MockToolCallingResponse) *MockToolCallingProvider {
	return &MockToolCallingProvider{responses: responses}
}

// ChatWithTools returns the next pre-configured response.
// It records the messages it received in LastMessages for test assertions.
func (m *MockToolCallingProvider) ChatWithTools(
	ctx context.Context,
	messages []ChatMessage,
	tools []ToolDef,
	opts *CompletionOptions,
) <-chan StreamChunk {
	ch := make(chan StreamChunk, 1)

	// Record what was sent so tests can assert on PreModel effects.
	m.LastMessages = make([]ChatMessage, len(messages))
	copy(m.LastMessages, messages)

	go func() {
		defer close(ch)

		if m.callCount >= len(m.responses) {
			ch <- StreamChunk{
				Error: fmt.Errorf("mock: no more responses (call %d, have %d)", m.callCount+1, len(m.responses)),
				Done:  true,
			}
			return
		}

		resp := m.responses[m.callCount]
		m.callCount++

		if resp.Error != nil {
			ch <- StreamChunk{Error: resp.Error, Done: true}
			return
		}

		ch <- StreamChunk{Text: resp.Text, ToolCalls: resp.ToolCalls, Done: true}
	}()

	return ch
}

// CallCount returns the number of ChatWithTools calls made.
func (m *MockToolCallingProvider) CallCount() int { return m.callCount }

// Reset resets call counter and clears LastMessages.
func (m *MockToolCallingProvider) Reset() {
	m.callCount = 0
	m.LastMessages = nil
}

// ── Provider / ConversationProvider stubs (required by the interface) ─────────

func (m *MockToolCallingProvider) Complete(_ context.Context, _ string, _ *CompletionOptions) (*Response, error) {
	return &Response{Text: "mock", FinishReason: "stop", Model: "mock"}, nil
}

func (m *MockToolCallingProvider) Ainvoke(_ context.Context, _ string, _ *CompletionOptions) <-chan StreamChunk {
	ch := make(chan StreamChunk, 1)
	ch <- StreamChunk{Text: "mock", Done: true}
	close(ch)
	return ch
}

func (m *MockToolCallingProvider) ChatMessages(_ context.Context, _ []ChatMessage, _ *CompletionOptions) <-chan StreamChunk {
	ch := make(chan StreamChunk, 1)
	ch <- StreamChunk{Text: "mock", Done: true}
	close(ch)
	return ch
}

func (m *MockToolCallingProvider) Name() string { return "mock-tools" }
func (m *MockToolCallingProvider) MaxTokens() int { return 8192 }
