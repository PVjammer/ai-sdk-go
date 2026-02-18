package hooks

import (
	"testing"

	"github.com/pvjammer/ai-sdk-go/pkg/agentcontext"
)

func TestSlidingWindowHook_Compress(t *testing.T) {
	// Create Extended context with low token limit
	config := &agentcontext.ExtendedConfig{
		DefaultContextID: "test",
		MaxTokens:        100, // Low limit
		AutoCompress:     false, // Disable auto-compress for manual testing
	}
	ext := agentcontext.NewExtendedWithConfig(config)

	// Add many messages
	for i := 0; i < 30; i++ {
		ext.AddMessage(agentcontext.Message{
			Role:    "user",
			Content: "Test message",
		})
	}

	// Register sliding window hook
	hook := NewSlidingWindow(10)
	ext.RegisterCompressionHook(hook)

	// Verify we have 30 messages
	messages := ext.GetMessages(0)
	if len(messages) != 30 {
		t.Errorf("Expected 30 messages before compression, got %d", len(messages))
	}

	// Manually trigger compression
	_, err := hook.Compress(ext)
	if err != nil {
		t.Errorf("Compression failed: %v", err)
	}

	// Should now have only 10 messages
	messages = ext.GetMessages(0)
	if len(messages) != 10 {
		t.Errorf("Expected 10 messages after compression, got %d", len(messages))
	}

	// Check that we kept the last 10
	firstMsg := messages[0]
	if firstMsg.Content != "Test message" {
		t.Errorf("Message content changed after compression")
	}
}

func TestSlidingWindowHook_ShouldCompress(t *testing.T) {
	ext := agentcontext.NewExtended()
	hook := NewSlidingWindow(5)

	// Add 3 messages - should not compress
	for i := 0; i < 3; i++ {
		ext.AddMessage(agentcontext.Message{
			Role:    "user",
			Content: "Test",
		})
	}

	if hook.ShouldCompress(ext) {
		t.Error("Should not compress with 3 messages (limit=5)")
	}

	// Add 3 more - should compress
	for i := 0; i < 3; i++ {
		ext.AddMessage(agentcontext.Message{
			Role:    "user",
			Content: "Test",
		})
	}

	if !hook.ShouldCompress(ext) {
		t.Error("Should compress with 6 messages (limit=5)")
	}
}

func TestSlidingWindowHook_AutoCompress(t *testing.T) {
	// This test verifies that auto-compression works without deadlocks
	config := &agentcontext.ExtendedConfig{
		DefaultContextID: "test",
		MaxTokens:        40, // Very low limit
		AutoCompress:     true, // Enable auto-compress
	}
	ext := agentcontext.NewExtendedWithConfig(config)

	hook := NewSlidingWindow(5)
	ext.RegisterCompressionHook(hook)

	// Add messages one by one - auto-compression should trigger
	// Each message is ~12 chars = ~3 tokens
	// After 14 messages (~42 tokens), should trigger compression to 5 messages
	// Then more messages can be added until hitting the token limit again
	for i := 0; i < 20; i++ {
		ext.AddMessage(agentcontext.Message{
			Role:    "user",
			Content: "Test message", // 12 chars = ~3 tokens
		})
	}

	// Should have been compressed at least once (no deadlock!)
	// Final count will be <= 13 messages (13 * 3 = 39 tokens < 40)
	messages := ext.GetMessages(0)
	if len(messages) > 13 {
		t.Errorf("Auto-compression should have kept message count <= 13, got %d", len(messages))
	}

	// Verify no compression errors occurred
	if err := ext.GetLastCompressionError(); err != nil {
		t.Errorf("Compression error: %v", err)
	}
}
