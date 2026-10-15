package agent_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/pvjammer/ai-sdk-go/pkg/agent"
	"github.com/pvjammer/ai-sdk-go/pkg/function"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// branchTestAgent is a minimal, full AIAgent implementation over AgentBase — unlike
// hooks_integration_test.go's testAgent (which calls BaseRun directly), Session needs a real
// AIAgent to wrap, so this adds the Metadata/Config/Run methods AgentBase itself doesn't
// provide. RegisterTool/SetHooks/GetHooks/SeedSession/SessionHistory/ClearSession are all
// promoted from the embedded AgentBase, which also satisfies sessionSeeder for free.
type branchTestAgent struct {
	agent.AgentBase
}

func newBranchTestAgent(provider llm.ToolCallingProvider) *branchTestAgent {
	a := &branchTestAgent{}
	a.InitBase(function.NilConfig{})
	a.SetProvider(provider)
	return a
}

func (a *branchTestAgent) Metadata() *agent.AgentMetadata {
	return &agent.AgentMetadata{Name: "branch-test", CLIName: "branch-test"}
}
func (a *branchTestAgent) Config() function.Config { return function.NilConfig{} }
func (a *branchTestAgent) Run(ctx context.Context, sessionID, message string) <-chan agent.Event {
	return a.BaseRun(ctx, sessionID, message, "", 10, false)
}

func TestSession_BranchesWhenToolCalled(t *testing.T) {
	mock := llm.NewMockToolCallingProvider(
		llm.MockToolCallingResponse{
			ToolCalls: []llm.ToolCall{{ID: "1", Name: "noop", Args: map[string]interface{}{}}},
		},
		llm.MockToolCallingResponse{Text: "scratch work done"},
	)
	a := newBranchTestAgent(mock)
	if err := a.RegisterTool(newMockFn("noop", "ok", nil)); err != nil {
		t.Fatalf("RegisterTool: %v", err)
	}

	log := agent.NewSessionLog()
	sess := agent.NewSession("s1", a, agent.SessionConfig{Log: log})

	events := collectEvents(t, sess.Send(context.Background(), "do the thing"))

	var finalResponse string
	for _, e := range events {
		if e.Type == agent.EventResponse {
			if r := e.Data.(agent.ResponseEvent); r.Done {
				finalResponse = r.Content
			}
		}
	}
	if finalResponse != "scratch work done" {
		t.Fatalf("expected final response %q, got %q", "scratch work done", finalResponse)
	}

	path := log.Path(sess.Leaf())
	if len(path) != 2 {
		t.Fatalf("expected trunk path length 2 (ref+answer), got %d: %+v", len(path), path)
	}
	if path[0].Role != "tool" || path[0].ToolName != "turn_scratch" || !strings.HasPrefix(path[0].Content, "ref:") {
		t.Fatalf("expected a turn_scratch branch-reference entry first, got %+v", path[0])
	}
	if path[1].Role != "assistant" || path[1].Content != "scratch work done" {
		t.Fatalf("expected the final answer entry second, got %+v", path[1])
	}

	branchID := strings.TrimPrefix(path[0].Content, "ref:")
	subtree := log.Subtree(branchID)
	if len(subtree) == 0 {
		t.Fatal("expected the branch to be retrievable via Subtree")
	}
	if subtree[0].Role != "user" || subtree[0].Content != "do the thing" {
		t.Fatalf("expected the branch root to be the user message, got %+v", subtree[0])
	}
	foundToolResult := false
	for _, e := range subtree {
		if e.Role == "tool" && e.ToolName == "noop" {
			foundToolResult = true
		}
	}
	if !foundToolResult {
		t.Fatalf("expected the branch to contain the tool-result entry, got %+v", subtree)
	}
}

func TestSession_SplicesWhenNoToolCalled(t *testing.T) {
	mock := llm.NewMockToolCallingProvider(
		llm.MockToolCallingResponse{Text: "direct answer"},
	)
	a := newBranchTestAgent(mock)

	log := agent.NewSessionLog()
	sess := agent.NewSession("s1", a, agent.SessionConfig{Log: log})

	collectEvents(t, sess.Send(context.Background(), "simple question"))

	path := log.Path(sess.Leaf())
	if len(path) != 2 {
		t.Fatalf("expected spliced path length 2 (user+assistant), got %d: %+v", len(path), path)
	}
	if path[0].Role != "user" || path[0].Content != "simple question" {
		t.Fatalf("expected the user entry first, got %+v", path[0])
	}
	if path[1].Role != "assistant" || path[1].Content != "direct answer" {
		t.Fatalf("expected the assistant entry second, got %+v", path[1])
	}
}

func TestSession_Steer(t *testing.T) {
	// A single-round turn: MockToolCallingProvider only tracks the MOST RECENT call's
	// messages, and a steer message is consumed (not persisted back into history) the first
	// time it's drained — so a single-call turn is the only way to observe it land.
	mock := llm.NewMockToolCallingProvider(
		llm.MockToolCallingResponse{Text: "done"},
	)
	a := newBranchTestAgent(mock)

	sess := agent.NewSession("s1", a, agent.SessionConfig{})
	sess.Steer("focus on X")
	collectEvents(t, sess.Send(context.Background(), "do something"))

	found := false
	for _, m := range mock.LastMessages {
		if m.Role == "user" && m.Content == "focus on X" {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected the steered message in the LLM call, got %+v", mock.LastMessages)
	}
}

func TestSessionLog_RoundTripPreservesOrder(t *testing.T) {
	log := agent.NewSessionLog()
	root := log.Append("", agent.Entry{Role: "user", Content: "first"})
	time.Sleep(time.Millisecond) // CreatedAt granularity — make ordering unambiguous
	second := log.Append(root.ID, agent.Entry{Role: "assistant", Content: "second"})
	time.Sleep(time.Millisecond)
	log.Append(root.ID, agent.Entry{Role: "assistant", Content: "second-sibling"})
	time.Sleep(time.Millisecond)
	log.Append(second.ID, agent.Entry{Role: "tool", Content: "third"})

	reloaded := agent.LoadSessionLog(log.Entries())

	before := log.Subtree(root.ID)
	after := reloaded.Subtree(root.ID)
	if len(before) != len(after) {
		t.Fatalf("expected %d entries after reload, got %d", len(before), len(after))
	}
	for i := range before {
		if before[i].ID != after[i].ID || before[i].Content != after[i].Content {
			t.Errorf("entry %d differs after reload: before=%+v after=%+v", i, before[i], after[i])
		}
	}
}

func TestAccumulateEvents(t *testing.T) {
	mock := llm.NewMockToolCallingProvider(
		llm.MockToolCallingResponse{
			ToolCalls: []llm.ToolCall{{ID: "1", Name: "noop", Args: map[string]interface{}{}}},
		},
		llm.MockToolCallingResponse{Text: "all done"},
	)
	a := newBranchTestAgent(mock)
	if err := a.RegisterTool(newMockFn("noop", "the tool result", nil)); err != nil {
		t.Fatalf("RegisterTool: %v", err)
	}

	result := agent.AccumulateEvents(a.BaseRun(context.Background(), "s1", "go", "", 10, false))

	if result.Response != "all done" {
		t.Errorf("expected Response %q, got %q", "all done", result.Response)
	}
	if len(result.ToolCalls) != 1 || result.ToolCalls[0].ToolName != "noop" {
		t.Fatalf("expected one noop ToolCallRecord, got %+v", result.ToolCalls)
	}
	if result.ToolCalls[0].Output != "the tool result" {
		t.Errorf("expected Output %q, got %v", "the tool result", result.ToolCalls[0].Output)
	}
	if result.Error != nil {
		t.Errorf("expected no error, got %v", result.Error)
	}
}
