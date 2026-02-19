package hooks

import (
	"context"

	"github.com/pvjammer/ai-sdk-go/pkg/agent"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// SlidingWindow returns an AgentHooks whose PreModel hook keeps only the last
// maxMessages non-system messages before each LLM call.
//
// The system message (role == "system") at index 0 is always preserved.
// When the history is within the limit, messages pass through unchanged.
//
// This is the simplest context management strategy: fast, predictable, and
// requires no LLM calls. Use TokenLimit if you need token-accurate trimming.
func SlidingWindow(maxMessages int) agent.AgentHooks {
	if maxMessages < 2 {
		maxMessages = 20
	}
	return agent.AgentHooks{
		PreModel: func(ctx context.Context, messages []llm.ChatMessage) ([]llm.ChatMessage, error) {
			sys, conv := splitSystem(messages)
			if len(conv) <= maxMessages {
				return messages, nil
			}
			// Keep only the last maxMessages conversation messages.
			// Align to a safe boundary so we never start mid-exchange.
			trimmed := alignToUserBoundary(conv[len(conv)-maxMessages:])
			return append(sys, trimmed...), nil
		},
	}
}

// TokenLimit returns an AgentHooks whose PreModel hook drops the oldest
// conversation exchanges when the estimated token count exceeds maxTokens.
//
// Token count is estimated at 4 characters per token (fast, no model call).
// The system message is always preserved. Messages are dropped in whole
// exchange units (from one user-message boundary to the next) so the model
// never receives orphaned tool-result or assistant messages.
//
// When the history is already within the limit, messages pass through unchanged.
func TokenLimit(maxTokens int) agent.AgentHooks {
	if maxTokens < 256 {
		maxTokens = 8000
	}
	return agent.AgentHooks{
		PreModel: func(ctx context.Context, messages []llm.ChatMessage) ([]llm.ChatMessage, error) {
			if estimateMsgTokens(messages) <= maxTokens {
				return messages, nil
			}

			sys, conv := splitSystem(messages)

			// Drop oldest complete exchanges until we are within the limit
			// or there is nothing left to drop.
			for len(conv) > 0 {
				if estimateMsgTokens(append(sys, conv...)) <= maxTokens {
					break
				}
				next := dropOldestExchange(conv)
				if len(next) == len(conv) {
					// Nothing could be dropped (single exchange) — stop to
					// avoid an infinite loop.
					break
				}
				conv = next
			}

			return append(sys, conv...), nil
		},
	}
}

// splitSystem separates the leading system message (if any) from the rest.
// Returns (systemMessages, conversationMessages).
// The systemMessages slice has 0 or 1 elements.
func splitSystem(messages []llm.ChatMessage) ([]llm.ChatMessage, []llm.ChatMessage) {
	if len(messages) > 0 && messages[0].Role == "system" {
		return messages[:1], messages[1:]
	}
	return nil, messages
}

// dropOldestExchange removes the first complete exchange from conv.
// An exchange starts at a user message and extends to (but not including)
// the next user message. If there is only one exchange, conv is returned
// unchanged so callers can detect the no-op and stop looping.
func dropOldestExchange(conv []llm.ChatMessage) []llm.ChatMessage {
	if len(conv) == 0 {
		return conv
	}
	// Find the second user message — that marks the start of the next exchange.
	for i := 1; i < len(conv); i++ {
		if conv[i].Role == "user" {
			return conv[i:]
		}
	}
	// Only one exchange; nothing safe to drop.
	return conv
}

// alignToUserBoundary trims the front of conv to start at the first user message.
// This prevents the model from receiving orphaned tool/assistant messages at the
// start of its context (which can cause tool-format errors in some models).
func alignToUserBoundary(conv []llm.ChatMessage) []llm.ChatMessage {
	for i, m := range conv {
		if m.Role == "user" {
			return conv[i:]
		}
	}
	return conv
}

// estimateMsgTokens estimates the total token count for a message slice.
// Uses the 4-chars-per-token heuristic: fast and good enough for trimming.
func estimateMsgTokens(messages []llm.ChatMessage) int {
	total := 0
	for _, m := range messages {
		total += len(m.Content)
		for _, tc := range m.ToolCalls {
			total += len(tc.Name) + len(tc.ID)
			// Args is an interface{}; use a conservative fixed overhead.
			total += 64
		}
	}
	return total / 4
}
