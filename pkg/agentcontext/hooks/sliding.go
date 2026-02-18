// Package hooks provides compression strategies for AgentContext.
package hooks

import (
	"fmt"

	"github.com/pvjammer/ai-sdk-go/pkg/agentcontext"
)

// SlidingWindowHook keeps only the last N messages.
//
// This is the simplest compression strategy - it just truncates old messages.
// Good for maintaining recent context without unbounded growth.
//
// Priority: 50 (medium - runs after summarization but before other strategies)
type SlidingWindowHook struct {
	maxMessages int
}

// NewSlidingWindow creates a new sliding window hook.
//
// maxMessages is the maximum number of messages to keep.
func NewSlidingWindow(maxMessages int) *SlidingWindowHook {
	if maxMessages < 1 {
		maxMessages = 20 // Default
	}

	return &SlidingWindowHook{
		maxMessages: maxMessages,
	}
}

// ShouldCompress returns true if there are more messages than the limit.
func (h *SlidingWindowHook) ShouldCompress(agentCtx agentcontext.AgentContext) bool {
	messages := agentCtx.GetMessages(0) // Get all messages
	return len(messages) > h.maxMessages
}

// Compress keeps only the last N messages.
func (h *SlidingWindowHook) Compress(agentCtx agentcontext.AgentContext) (agentcontext.AgentContext, error) {
	// Try to cast to Extended for direct manipulation
	if ext, ok := agentCtx.(*agentcontext.Extended); ok {
		return h.compressExtended(ext)
	}

	// For other types, we can't modify in place
	// Return error indicating incompatibility
	return agentCtx, fmt.Errorf("SlidingWindowHook only works with Extended AgentContext")
}

// compressExtended compresses an Extended context by keeping only last N messages.
func (h *SlidingWindowHook) compressExtended(ext *agentcontext.Extended) (agentcontext.AgentContext, error) {
	messages := ext.GetMessages(0)

	if len(messages) <= h.maxMessages {
		return ext, nil // No compression needed
	}

	// Keep only the last maxMessages
	keepMessages := messages[len(messages)-h.maxMessages:]

	// Clear and re-add
	ext.Clear()
	for _, msg := range keepMessages {
		ext.AddMessage(msg)
	}

	return ext, nil
}

// Priority returns the hook's execution priority.
//
// Higher values run first. Sliding window has medium priority (50)
// so it runs after more sophisticated strategies like summarization.
func (h *SlidingWindowHook) Priority() int {
	return 50
}
