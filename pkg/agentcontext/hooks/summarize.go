package hooks

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/pvjammer/ai-sdk-go/pkg/agentcontext"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// SummarizationHook uses an LLM to summarize old messages.
//
// This preserves more information than sliding window while still compressing.
// It summarizes old messages and replaces them with a single summary message,
// keeping recent messages intact.
//
// Priority: 100 (high - runs before simpler strategies)
type SummarizationHook struct {
	llm               llm.Provider
	maxTokens         int // Trigger summarization above this
	keepRecentCount   int // Keep this many recent messages unsummarized
	summaryPrompt     string
	ctx               context.Context
}

// SummarizationConfig configures the summarization hook.
type SummarizationConfig struct {
	MaxTokens       int    // Trigger above this token count (default: 8000)
	KeepRecentCount int    // Keep N recent messages (default: 10)
	SummaryPrompt   string // Custom prompt (optional)
}

// DefaultSummarizationConfig returns sensible defaults.
func DefaultSummarizationConfig() *SummarizationConfig {
	return &SummarizationConfig{
		MaxTokens:       8000,
		KeepRecentCount: 10,
		SummaryPrompt:   "",
	}
}

// NewSummarization creates a new summarization hook.
func NewSummarization(provider llm.Provider, maxTokens int) *SummarizationHook {
	return NewSummarizationWithConfig(provider, &SummarizationConfig{
		MaxTokens:       maxTokens,
		KeepRecentCount: 10,
	})
}

// NewSummarizationWithConfig creates a summarization hook with custom config.
func NewSummarizationWithConfig(provider llm.Provider, config *SummarizationConfig) *SummarizationHook {
	if config == nil {
		config = DefaultSummarizationConfig()
	}

	prompt := config.SummaryPrompt
	if prompt == "" {
		prompt = `Summarize the following conversation history concisely, preserving key information, decisions, and context.
Focus on what's important for understanding the conversation flow.

Conversation:
%s

Provide a clear, concise summary:`
	}

	return &SummarizationHook{
		llm:             provider,
		maxTokens:       config.MaxTokens,
		keepRecentCount: config.KeepRecentCount,
		summaryPrompt:   prompt,
		ctx:             context.Background(),
	}
}

// ShouldCompress returns true if token count exceeds the limit.
func (h *SummarizationHook) ShouldCompress(agentCtx agentcontext.AgentContext) bool {
	return agentCtx.EstimateTokens() > h.maxTokens
}

// Compress summarizes old messages using the LLM.
func (h *SummarizationHook) Compress(agentCtx agentcontext.AgentContext) (agentcontext.AgentContext, error) {
	// Try to cast to Extended for direct manipulation
	if ext, ok := agentCtx.(*agentcontext.Extended); ok {
		return h.compressExtended(ext)
	}

	return agentCtx, fmt.Errorf("SummarizationHook only works with Extended AgentContext")
}

// compressExtended compresses an Extended context via summarization.
func (h *SummarizationHook) compressExtended(ext *agentcontext.Extended) (agentcontext.AgentContext, error) {
	messages := ext.GetMessages(0)

	// Need at least keepRecentCount + 2 messages to summarize
	if len(messages) <= h.keepRecentCount+1 {
		return ext, nil // Not enough to summarize
	}

	// Split into old (to summarize) and recent (to keep)
	splitPoint := len(messages) - h.keepRecentCount
	oldMessages := messages[:splitPoint]
	recentMessages := messages[splitPoint:]

	// Format old messages for summarization
	conversationText := h.formatMessages(oldMessages)

	// Call LLM to summarize
	summary, err := h.summarize(conversationText)
	if err != nil {
		return ext, fmt.Errorf("summarization failed: %w", err)
	}

	// Create summary message
	summaryMsg := agentcontext.Message{
		Role:      "system",
		Content:   fmt.Sprintf("[SUMMARY OF PREVIOUS CONVERSATION]\n%s", summary),
		Timestamp: time.Now(),
		Metadata: map[string]interface{}{
			"type":              "summary",
			"messages_count":    len(oldMessages),
			"summarized_at":     time.Now(),
			"original_tokens":   estimateTokens(conversationText),
			"compressed_tokens": estimateTokens(summary),
		},
	}

	// Rebuild context: summary + recent messages
	ext.Clear()
	ext.AddMessage(summaryMsg)
	for _, msg := range recentMessages {
		ext.AddMessage(msg)
	}

	return ext, nil
}

// formatMessages formats messages for the LLM prompt.
func (h *SummarizationHook) formatMessages(messages []agentcontext.Message) string {
	var sb strings.Builder

	for _, msg := range messages {
		role := strings.Title(msg.Role)
		sb.WriteString(fmt.Sprintf("%s: %s\n", role, msg.Content))

		// Include tool calls
		for _, tc := range msg.ToolCalls {
			if tc.Output != nil {
				sb.WriteString(fmt.Sprintf("  [Used tool: %s -> %v]\n", tc.Name, tc.Output))
			}
		}
	}

	return sb.String()
}

// summarize calls the LLM to generate a summary.
func (h *SummarizationHook) summarize(conversationText string) (string, error) {
	prompt := fmt.Sprintf(h.summaryPrompt, conversationText)

	opts := &llm.CompletionOptions{
		MaxTokens:   500, // Limit summary length
		Temperature: 0.3, // Low temperature for focused summaries
		Stream:      false,
	}

	chunks := h.llm.Ainvoke(h.ctx, prompt, opts)

	// Collect response
	var summary strings.Builder
	for chunk := range chunks {
		if chunk.Error != nil {
			return "", chunk.Error
		}
		summary.WriteString(chunk.Text)
	}

	return strings.TrimSpace(summary.String()), nil
}

// Priority returns the hook's execution priority.
//
// Summarization has high priority (100) so it runs before simpler strategies.
func (h *SummarizationHook) Priority() int {
	return 100
}

// estimateTokens provides a rough token estimate for text.
func estimateTokens(text string) int {
	return len(text) / 4
}
