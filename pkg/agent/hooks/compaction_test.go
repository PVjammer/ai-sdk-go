package hooks_test

import (
	"context"
	"strings"
	"testing"

	"github.com/pvjammer/ai-sdk-go/pkg/agent"
	"github.com/pvjammer/ai-sdk-go/pkg/agent/hooks"
	"github.com/pvjammer/ai-sdk-go/pkg/function"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
	"github.com/pvjammer/ai-sdk-go/pkg/skill"
)

func newCompactionTestBase() *agent.AgentBase {
	base := &agent.AgentBase{}
	base.InitBase(function.NilConfig{})
	return base
}

// bigMsg returns a (role, content) message whose content is long enough to push the char/4
// estimate well past a small test token budget.
func bigMsg(role string, n int) llm.ChatMessage {
	return llm.ChatMessage{Role: role, Content: bigContent(n)}
}

func TestCompaction_BelowThreshold_NoChange(t *testing.T) {
	base := newCompactionTestBase()
	history := []llm.ChatMessage{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi there"},
		{Role: "user", Content: "current question"},
	}
	base.SeedSession("s1", history)

	summarizer := llm.NewMockProvider("should not be called")
	h := hooks.Compaction(base, summarizer, hooks.CompactionOptions{MaxTokens: 100000, Threshold: 0.75})

	ctx := skill.WithSessionID(context.Background(), "s1")
	if _, err := h.PreModel(ctx, history); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	got := base.SessionHistory("s1")
	if len(got) != len(history) {
		t.Fatalf("expected history untouched below threshold, got %d messages: %+v", len(got), got)
	}
}

func TestCompaction_AboveThreshold_SummarizesOldTurnsOnly(t *testing.T) {
	base := newCompactionTestBase()
	history := []llm.ChatMessage{
		bigMsg("user", 2000),
		bigMsg("assistant", 2000),
		bigMsg("user", 2000),
		bigMsg("assistant", 2000),
		{Role: "user", Content: "current turn — must survive verbatim"},
	}
	base.SeedSession("s1", history)

	summarizer := llm.NewMockProvider("Goal: test\nProgress: compacted old turns")
	h := hooks.Compaction(base, summarizer, hooks.CompactionOptions{MaxTokens: 500, Threshold: 0.5})

	ctx := skill.WithSessionID(context.Background(), "s1")
	if _, err := h.PreModel(ctx, history); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	got := base.SessionHistory("s1")
	if len(got) == 0 {
		t.Fatal("expected non-empty history after compaction")
	}
	if !strings.Contains(got[0].Content, "Compacted") {
		t.Fatalf("expected first message to be the compacted summary, got: %+v", got[0])
	}
	last := got[len(got)-1]
	if last.Role != "user" || last.Content != "current turn — must survive verbatim" {
		t.Fatalf("expected the current turn's user message to survive verbatim as the last entry, got %+v", last)
	}
}

func TestCompaction_ReCompaction_UpdatesRatherThanStacks(t *testing.T) {
	base := newCompactionTestBase()
	// Pre-seed as if a first compaction already ran: a marker message, then more old turns,
	// then the current turn.
	history := []llm.ChatMessage{
		{Role: "assistant", Content: "[Compacted — session summary follows]\n\nGoal: test\nProgress: did turn 1"},
		bigMsg("user", 2000),
		bigMsg("assistant", 2000),
		bigMsg("user", 2000),
		bigMsg("assistant", 2000),
		{Role: "user", Content: "latest turn"},
	}
	base.SeedSession("s1", history)

	summarizer := llm.NewMockProvider("Goal: test\nProgress: did turns 1 and 2")
	h := hooks.Compaction(base, summarizer, hooks.CompactionOptions{MaxTokens: 500, Threshold: 0.5})

	ctx := skill.WithSessionID(context.Background(), "s1")
	if _, err := h.PreModel(ctx, history); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	got := base.SessionHistory("s1")
	markerCount := 0
	for _, m := range got {
		if strings.Contains(m.Content, "Compacted") {
			markerCount++
		}
	}
	if markerCount != 1 {
		t.Fatalf("expected exactly one compacted-summary marker after re-compaction (update, not append), got %d in %+v", markerCount, got)
	}
	if got[0].Content != "[Compacted — session summary follows]\n\nGoal: test\nProgress: did turns 1 and 2" {
		t.Fatalf("expected the marker message to be the UPDATED summary, got %+v", got[0])
	}
}

func TestCompaction_SummarizerError_NonDestructive(t *testing.T) {
	base := newCompactionTestBase()
	history := []llm.ChatMessage{
		bigMsg("user", 2000),
		bigMsg("assistant", 2000),
		{Role: "user", Content: "current turn"},
	}
	base.SeedSession("s1", history)

	// NewMockProviderWithFailure(0, ...) with zero responses: the first Complete() call finds
	// callCount(1) > len(responses)(0) and errors — exercising the non-destructive path without
	// needing a dedicated "always fails" constructor.
	summarizer := llm.NewMockProvider()
	h := hooks.Compaction(base, summarizer, hooks.CompactionOptions{MaxTokens: 500, Threshold: 0.5})

	ctx := skill.WithSessionID(context.Background(), "s1")
	if _, err := h.PreModel(ctx, history); err != nil {
		t.Fatalf("PreModel itself should not surface a summarizer error: %v", err)
	}

	got := base.SessionHistory("s1")
	if len(got) != len(history) {
		t.Fatalf("expected history untouched after a failed summarization call, got %d messages: %+v", len(got), got)
	}
}

func TestCompaction_TypedResult_TracksFileOpsAndUsage(t *testing.T) {
	base := newCompactionTestBase()
	history := []llm.ChatMessage{
		bigMsg("user", 1000),
		{
			Role:    "assistant",
			Content: bigContent(1000),
			ToolCalls: []llm.ToolCall{
				{ID: "1", Name: "read_file", Args: map[string]interface{}{"path": "a.go"}},
				{ID: "2", Name: "write_file", Args: map[string]interface{}{"path": "b.go"}},
			},
		},
		bigMsg("tool", 1000),
		{Role: "user", Content: "current turn"},
	}
	base.SeedSession("s1", history)

	summarizer := llm.NewMockProvider("Goal: test\nRelevant Files: a.go, b.go")

	var got hooks.CompactionResult
	var gotSession string
	h := hooks.Compaction(base, summarizer, hooks.CompactionOptions{
		MaxTokens: 500,
		Threshold: 0.5,
		OnCompaction: func(sessionID string, result hooks.CompactionResult) {
			gotSession = sessionID
			got = result
		},
	})

	ctx := skill.WithSessionID(context.Background(), "s1")
	if _, err := h.PreModel(ctx, history); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if gotSession != "s1" {
		t.Fatalf("expected OnCompaction to fire for session s1, got %q", gotSession)
	}
	if len(got.Details.ReadFiles) != 1 || got.Details.ReadFiles[0] != "a.go" {
		t.Errorf("expected ReadFiles=[a.go], got %v", got.Details.ReadFiles)
	}
	if len(got.Details.ModifiedFiles) != 1 || got.Details.ModifiedFiles[0] != "b.go" {
		t.Errorf("expected ModifiedFiles=[b.go], got %v", got.Details.ModifiedFiles)
	}
	if got.Usage.TotalTokens == 0 {
		t.Error("expected non-zero Usage from the summarizer call")
	}
	if got.TokensBefore == 0 {
		t.Error("expected non-zero TokensBefore")
	}
}

func TestCompaction_TypedResult_DetailsMergeAcrossRounds(t *testing.T) {
	base := newCompactionTestBase()
	summarizer := llm.NewMockProvider(
		"Goal: test\nRound 1 summary",
		"Goal: test\nRound 2 summary",
	)

	var results []hooks.CompactionResult
	h := hooks.Compaction(base, summarizer, hooks.CompactionOptions{
		MaxTokens: 500,
		Threshold: 0.5,
		OnCompaction: func(_ string, result hooks.CompactionResult) {
			results = append(results, result)
		},
	})
	ctx := skill.WithSessionID(context.Background(), "s1")

	// Round 1: compacts a span that read a.go.
	round1 := []llm.ChatMessage{
		bigMsg("user", 1000),
		{
			Role:      "assistant",
			Content:   bigContent(1000),
			ToolCalls: []llm.ToolCall{{ID: "1", Name: "read_file", Args: map[string]interface{}{"path": "a.go"}}},
		},
		bigMsg("tool", 1000),
		{Role: "user", Content: "turn 1"},
	}
	base.SeedSession("s1", round1)
	if _, err := h.PreModel(ctx, round1); err != nil {
		t.Fatalf("round 1: unexpected error: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected round 1 to compact, got %d OnCompaction calls", len(results))
	}

	// Round 2: append more history (including the round-1 compacted marker, which SeedSession
	// already wrote into base's session), another tool call writing c.go, and a new current turn.
	afterRound1 := base.SessionHistory("s1")
	round2 := append(append([]llm.ChatMessage{}, afterRound1...),
		bigMsg("user", 1000),
		llm.ChatMessage{
			Role:      "assistant",
			Content:   bigContent(1000),
			ToolCalls: []llm.ToolCall{{ID: "2", Name: "write_file", Args: map[string]interface{}{"path": "c.go"}}},
		},
		bigMsg("tool", 1000),
		llm.ChatMessage{Role: "user", Content: "turn 2"},
	)
	base.SeedSession("s1", round2)
	if _, err := h.PreModel(ctx, round2); err != nil {
		t.Fatalf("round 2: unexpected error: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected round 2 to also compact, got %d OnCompaction calls", len(results))
	}

	final := results[1]
	if len(final.Details.ReadFiles) != 1 || final.Details.ReadFiles[0] != "a.go" {
		t.Errorf("expected round 2's Details to still carry forward ReadFiles=[a.go] from round 1, got %v", final.Details.ReadFiles)
	}
	if len(final.Details.ModifiedFiles) != 1 || final.Details.ModifiedFiles[0] != "c.go" {
		t.Errorf("expected round 2's Details to add ModifiedFiles=[c.go], got %v", final.Details.ModifiedFiles)
	}
}
