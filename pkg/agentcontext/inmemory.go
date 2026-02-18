package agentcontext

import (
	"fmt"
	"time"
)

// InMemory is the simplest AgentContext implementation.
//
// It stores messages in memory with no persistence, personalization,
// context switching, or compression. Perfect for simple use cases
// and as a building block for more advanced implementations.
type InMemory struct {
	messages []Message
}

// NewInMemory creates a new in-memory context with no advanced features.
func NewInMemory() *InMemory {
	return &InMemory{
		messages: make([]Message, 0),
	}
}

// AddMessage adds a message to the conversation history.
func (m *InMemory) AddMessage(msg Message) {
	if msg.Timestamp.IsZero() {
		msg.Timestamp = time.Now()
	}
	m.messages = append(m.messages, msg)
}

// GetMessages returns the last N messages, or all messages if limit <= 0.
// Returns a defensive copy to prevent external modification of internal state.
func (m *InMemory) GetMessages(limit int) []Message {
	var toReturn []Message
	if limit <= 0 || limit > len(m.messages) {
		toReturn = m.messages
	} else {
		toReturn = m.messages[len(m.messages)-limit:]
	}

	// Return a defensive copy
	result := make([]Message, len(toReturn))
	copy(result, toReturn)
	return result
}

// Clear removes all messages from the context.
func (m *InMemory) Clear() {
	m.messages = make([]Message, 0)
}

// SetPersonalization is a no-op in InMemory (use Extended for personalization).
func (m *InMemory) SetPersonalization(key string, value interface{}) {
	// No-op: InMemory doesn't support personalization
}

// GetPersonalization always returns nil in InMemory.
func (m *InMemory) GetPersonalization(key string) interface{} {
	return nil
}

// SwitchContext is a no-op in InMemory (use Extended for context switching).
func (m *InMemory) SwitchContext(contextID string) error {
	return fmt.Errorf("InMemory does not support context switching")
}

// GetCurrentContextID returns empty string in InMemory.
func (m *InMemory) GetCurrentContextID() string {
	return ""
}

// EstimateTokens provides a rough token count using the 4-char-per-token heuristic.
//
// This is intentionally simple and fast. More accurate counting can be added
// in Extended or via a custom implementation.
func (m *InMemory) EstimateTokens() int {
	totalChars := 0
	for _, msg := range m.messages {
		totalChars += len(msg.Content)
		// Add tool call content
		for _, tc := range msg.ToolCalls {
			totalChars += len(tc.Name)
			// Rough estimate for input/output
			totalChars += len(fmt.Sprintf("%v", tc.Input))
			totalChars += len(fmt.Sprintf("%v", tc.Output))
		}
	}
	// 1 token ≈ 4 characters (rough estimate)
	return totalChars / 4
}

// NeedsCompression always returns false in InMemory (no compression support).
func (m *InMemory) NeedsCompression() bool {
	return false
}

// RegisterCompressionHook is a no-op in InMemory (use Extended for compression).
func (m *InMemory) RegisterCompressionHook(hook CompressionHook) {
	// No-op: InMemory doesn't support compression hooks
}
