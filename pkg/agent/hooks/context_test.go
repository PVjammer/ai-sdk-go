package hooks_test

import (
	"context"
	"testing"

	"github.com/pvjammer/ai-sdk-go/pkg/agent"
	"github.com/pvjammer/ai-sdk-go/pkg/agent/hooks"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// ── helpers ───────────────────────────────────────────────────────────────────

// msgs builds a []llm.ChatMessage from (role, content) pairs.
func msgs(pairs ...string) []llm.ChatMessage {
	out := make([]llm.ChatMessage, 0, len(pairs)/2)
	for i := 0; i+1 < len(pairs); i += 2 {
		out = append(out, llm.ChatMessage{Role: pairs[i], Content: pairs[i+1]})
	}
	return out
}

// bigContent returns a string of length n.
func bigContent(n int) string {
	b := make([]byte, n)
	for i := range b {
		b[i] = 'x'
	}
	return string(b)
}

func applyPreModel(h agent.AgentHooks, messages []llm.ChatMessage) ([]llm.ChatMessage, error) {
	// Zero value AgentHooks — we only care about the PreModel field.
	if h.PreModel == nil {
		return messages, nil
	}
	return h.PreModel(context.Background(), messages)
}

// ── SlidingWindow tests ───────────────────────────────────────────────────────

func TestSlidingWindow_WithinLimit(t *testing.T) {
	h := hooks.SlidingWindow(10)
	input := msgs(
		"user", "hello",
		"assistant", "hi",
		"user", "how are you?",
	)
	got, err := applyPreModel(h, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got) != 3 {
		t.Errorf("expected 3 messages (within limit), got %d", len(got))
	}
}

func TestSlidingWindow_TrimsToLimit(t *testing.T) {
	// 5 exchanges × 2 messages each = 10 conversation messages; limit = 6.
	input := msgs(
		"user", "q1", "assistant", "a1",
		"user", "q2", "assistant", "a2",
		"user", "q3", "assistant", "a3",
		"user", "q4", "assistant", "a4",
		"user", "q5", "assistant", "a5",
	)
	h := hooks.SlidingWindow(6)
	got, err := applyPreModel(h, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Should be <= 6 messages and start with "user" (aligned to user boundary).
	if len(got) > 6 {
		t.Errorf("expected ≤6 messages after trim, got %d", len(got))
	}
	if got[0].Role != "user" {
		t.Errorf("expected first message to be 'user', got %q", got[0].Role)
	}
}

func TestSlidingWindow_PreservesSystemMessage(t *testing.T) {
	input := msgs(
		"system", "you are helpful",
		"user", "q1", "assistant", "a1",
		"user", "q2", "assistant", "a2",
		"user", "q3", "assistant", "a3",
		"user", "q4", "assistant", "a4",
	)
	h := hooks.SlidingWindow(4) // limit conversation messages to 4
	got, err := applyPreModel(h, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got[0].Role != "system" {
		t.Errorf("expected system message to be preserved at index 0, got %q", got[0].Role)
	}
}

func TestSlidingWindow_NeverStartsMidExchange(t *testing.T) {
	// Build a history where a naive slice [n:] would land on an assistant message.
	// SlidingWindow must align forward to the next user message.
	input := msgs(
		"user", "q1",
		"assistant", "thinking",
		"assistant", "a1", // two assistant messages back-to-back
		"user", "q2",
		"assistant", "a2",
		"user", "q3",
		"assistant", "a3",
	)
	h := hooks.SlidingWindow(5) // a naive last-5 would start at "assistant a1"
	got, err := applyPreModel(h, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got[0].Role != "user" {
		t.Errorf("after trim, expected first message role=user, got %q", got[0].Role)
	}
}

// ── TokenLimit tests ──────────────────────────────────────────────────────────

func TestTokenLimit_WithinLimit(t *testing.T) {
	h := hooks.TokenLimit(10000) // very generous limit
	input := msgs(
		"user", "hi",
		"assistant", "hello",
	)
	got, err := applyPreModel(h, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got) != 2 {
		t.Errorf("expected 2 messages (within limit), got %d", len(got))
	}
}

func TestTokenLimit_TrimsOldestExchange(t *testing.T) {
	// Each message content is 4000 chars ≈ 1000 tokens.
	// Three exchanges = ~6000 tokens; limit = 3000 → should drop at least one exchange.
	big := bigContent(4000)
	input := msgs(
		"user", big,
		"assistant", big,
		"user", big,
		"assistant", big,
		"user", big,
		"assistant", big,
	)
	h := hooks.TokenLimit(3000)
	got, err := applyPreModel(h, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got) >= len(input) {
		t.Errorf("expected fewer messages after trimming, got %d (had %d)", len(got), len(input))
	}
	if got[0].Role != "user" {
		t.Errorf("after trim, expected first message role=user, got %q", got[0].Role)
	}
}

func TestTokenLimit_PreservesSystemMessage(t *testing.T) {
	big := bigContent(4000)
	input := msgs(
		"system", "be helpful",
		"user", big,
		"assistant", big,
		"user", big,
		"assistant", big,
	)
	h := hooks.TokenLimit(1500) // forces a trim
	got, err := applyPreModel(h, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got[0].Role != "system" {
		t.Errorf("system message should always be first, got %q", got[0].Role)
	}
}

func TestTokenLimit_SingleExchangeNotDropped(t *testing.T) {
	// When there is only one exchange, we can't drop anything safely.
	// The hook should not loop forever and should return something valid.
	big := bigContent(40000) // massively over the limit
	input := msgs(
		"user", big,
		"assistant", big,
	)
	h := hooks.TokenLimit(100) // tiny limit, impossible to satisfy with one exchange
	got, err := applyPreModel(h, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Should return without panicking or looping; content may still exceed limit.
	if len(got) == 0 {
		t.Error("expected at least one message returned")
	}
}

func TestTokenLimit_ToolExchangePreservedTogether(t *testing.T) {
	// A tool exchange: user → assistant(tool_call) → tool → assistant(text)
	// When trimming, we should never see an orphaned tool message at the front.
	big := bigContent(2000)
	input := []llm.ChatMessage{
		{Role: "user", Content: big},
		{Role: "assistant", Content: "", ToolCalls: []llm.ToolCall{{Name: "search", ID: "1"}}},
		{Role: "tool", Content: big, ToolCallID: "1", ToolName: "search"},
		{Role: "assistant", Content: big},
		{Role: "user", Content: big},
		{Role: "assistant", Content: big},
	}
	h := hooks.TokenLimit(1200) // should force trimming of the first exchange
	got, err := applyPreModel(h, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// After trimming, the first message must be "user" (not "tool" or "assistant").
	if got[0].Role != "user" {
		t.Errorf("first message must be 'user' after trim, got %q", got[0].Role)
	}
}
