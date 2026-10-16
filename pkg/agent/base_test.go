package agent

import (
	"testing"

	"github.com/pvjammer/ai-sdk-go/pkg/function"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// TestSeedSession_MidTurnReseedDoesNotDetachHeldPointer guards against a real, deterministic
// bug found via code review and confirmed by tracing executeAgentLoop: executeAgentLoop grabs
// one *agentSession pointer at the start of a turn and keeps appending to it for the whole
// turn. SeedSession used to always allocate a brand-new *agentSession and replace the map
// entry — which silently detached that already-held pointer the moment a PreModel hook
// (Compaction) called SeedSession mid-turn: everything the loop appended afterward (further
// tool calls, the final answer) was still written, just to an object b.sessions[id] no longer
// pointed to, so a later SessionHistory call could never see it. This test simulates exactly
// that sequence without needing a real LLM call: seed, grab the live pointer (standing in for
// what executeAgentLoop holds), append through it (standing in for tool-call rounds before
// compaction fires), reseed (standing in for Compaction firing mid-turn), then append through
// the SAME pointer again (rounds after compaction) — SessionHistory must see everything.
func TestSeedSession_MidTurnReseedDoesNotDetachHeldPointer(t *testing.T) {
	b := &AgentBase{}
	b.InitBase(function.NilConfig{})

	const id = "scratch-1"
	b.SeedSession(id, []llm.ChatMessage{{Role: "user", Content: "seed"}})

	// Stand-in for executeAgentLoop's session := b.getOrCreateSession(sessionID), held for the
	// rest of the turn.
	held := b.sessions[id]

	// Stand-in for tool-call rounds appended before compaction fires.
	held.history = append(held.history, llm.ChatMessage{Role: "assistant", Content: "before compaction"})

	// Stand-in for Compaction's PreModel hook calling SeedSession mid-turn, from the same
	// goroutine, with a shorter (compacted) history.
	b.SeedSession(id, []llm.ChatMessage{{Role: "assistant", Content: "[compacted summary]"}})

	// Stand-in for tool-call rounds appended AFTER compaction fires, via the SAME pointer
	// executeAgentLoop has held the whole time — this is exactly what the bug lost.
	held.history = append(held.history, llm.ChatMessage{Role: "assistant", Content: "after compaction"})

	got := b.SessionHistory(id)
	if len(got) == 0 || got[len(got)-1].Content != "after compaction" {
		t.Fatalf("expected the post-compaction append to be visible via SessionHistory, got %+v", got)
	}

	foundSummary := false
	for _, m := range got {
		if m.Content == "[compacted summary]" {
			foundSummary = true
		}
		if m.Content == "before compaction" {
			t.Fatalf("expected the pre-compaction message to have been replaced by the reseed, but it survived: %+v", got)
		}
	}
	if !foundSummary {
		t.Fatalf("expected the compacted summary to be present, got %+v", got)
	}
}
