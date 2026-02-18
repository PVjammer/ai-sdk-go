package agentcontext

import (
	"fmt"
	"testing"
)

func TestExtended_BasicOperations(t *testing.T) {
	ext := NewExtended()

	// Test adding messages
	ext.AddMessage(Message{Role: "user", Content: "Hello"})
	ext.AddMessage(Message{Role: "assistant", Content: "Hi there"})

	messages := ext.GetMessages(0)
	if len(messages) != 2 {
		t.Errorf("Expected 2 messages, got %d", len(messages))
	}

	// Test personalization
	ext.SetPersonalization("name", "Test")
	if name := ext.GetPersonalization("name"); name != "Test" {
		t.Errorf("Expected personalization 'Test', got %v", name)
	}

	// Test context switching
	ext.SwitchContext("work")
	if ext.GetCurrentContextID() != "work" {
		t.Errorf("Expected context 'work', got %s", ext.GetCurrentContextID())
	}

	// Messages in new context should be empty
	messages = ext.GetMessages(0)
	if len(messages) != 0 {
		t.Errorf("Expected 0 messages in new context, got %d", len(messages))
	}

	// Switch back
	ext.SwitchContext("default")
	messages = ext.GetMessages(0)
	if len(messages) != 2 {
		t.Errorf("Expected 2 messages after switching back, got %d", len(messages))
	}
}

func TestExtended_TokenEstimation(t *testing.T) {
	ext := NewExtended()

	ext.AddMessage(Message{Role: "user", Content: "1234"}) // 4 chars = ~1 token

	tokens := ext.EstimateTokens()
	if tokens != 1 {
		t.Errorf("Expected ~1 token, got %d", tokens)
	}

	ext.AddMessage(Message{Role: "assistant", Content: "12345678"}) // 8 chars = ~2 tokens

	tokens = ext.EstimateTokens()
	if tokens != 3 {
		t.Errorf("Expected ~3 tokens, got %d", tokens)
	}
}

func TestExtended_ClearAndRemove(t *testing.T) {
	ext := NewExtended()

	ext.AddMessage(Message{Role: "user", Content: "Hello"})
	ext.SwitchContext("work")
	ext.AddMessage(Message{Role: "user", Content: "Work message"})

	// Clear current context (work)
	ext.Clear()
	messages := ext.GetMessages(0)
	if len(messages) != 0 {
		t.Errorf("Expected 0 messages after clear, got %d", len(messages))
	}

	// Default context should still have messages
	ext.SwitchContext("default")
	messages = ext.GetMessages(0)
	if len(messages) != 1 {
		t.Errorf("Expected 1 message in default context, got %d", len(messages))
	}

	// Test RemoveContext
	err := ext.RemoveContext("work")
	if err != nil {
		t.Errorf("Failed to remove context: %v", err)
	}

	// Can't remove current context
	err = ext.RemoveContext("default")
	if err == nil {
		t.Error("Should not be able to remove current context")
	}
}

func TestExtended_DefensiveCopy(t *testing.T) {
	ext := NewExtended()

	// Add original message
	ext.AddMessage(Message{Role: "user", Content: "Original"})

	// Get messages
	messages := ext.GetMessages(0)
	if len(messages) != 1 {
		t.Fatalf("Expected 1 message, got %d", len(messages))
	}

	// Try to modify the returned slice
	messages[0].Content = "Modified"

	// Get messages again - should still be original
	messages2 := ext.GetMessages(0)
	if messages2[0].Content != "Original" {
		t.Errorf("GetMessages should return defensive copy, but internal state was modified: got %s", messages2[0].Content)
	}
}

func TestExtended_CompressionErrorHandling(t *testing.T) {
	config := &ExtendedConfig{
		DefaultContextID: "test",
		MaxTokens:        10, // Very low to trigger compression
		AutoCompress:     true,
	}
	ext := NewExtendedWithConfig(config)

	// Create a hook that always fails
	failingHook := &FailingHook{}
	ext.RegisterCompressionHook(failingHook)

	// Add messages to trigger compression
	for i := 0; i < 10; i++ {
		ext.AddMessage(Message{
			Role:    "user",
			Content: "Test message",
		})
	}

	// Check that the error was recorded
	err := ext.GetLastCompressionError()
	if err == nil {
		t.Error("Expected compression error to be recorded, but got nil")
	}
}

// FailingHook is a test hook that always fails
type FailingHook struct{}

func (h *FailingHook) ShouldCompress(agentCtx AgentContext) bool {
	return true
}

func (h *FailingHook) Compress(agentCtx AgentContext) (AgentContext, error) {
	return agentCtx, fmt.Errorf("intentional test failure")
}

func (h *FailingHook) Priority() int {
	return 100
}
