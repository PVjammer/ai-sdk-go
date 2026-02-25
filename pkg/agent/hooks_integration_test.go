// Integration tests for the AgentHooks system.
//
// These tests exercise the full AgentBase.executeAgentLoop to verify that
// hook call sites fire correctly in the real code path, not just as unit tests
// of isolated functions.
//
// They use MockToolCallingProvider (no live LLM required) and mock AIFunctions.
package agent_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/pvjammer/ai-sdk-go/pkg/agent"
	"github.com/pvjammer/ai-sdk-go/pkg/agent/hooks"
	"github.com/pvjammer/ai-sdk-go/pkg/function"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// ── mock helpers ──────────────────────────────────────────────────────────────

// testAgent wraps AgentBase so we can call SetProvider (not on the AIAgent
// interface) while still testing the real agent loop.
type testAgent struct {
	agent.AgentBase
}

func newTestAgent() *testAgent {
	a := &testAgent{}
	a.InitBase(function.NilConfig{})
	return a
}

// mockFn is a minimal AIFunction that returns a fixed output.
type mockFn struct {
	function.Base
	name   string
	output interface{}
	err    error
}

func newMockFn(name string, output interface{}, err error) *mockFn {
	return &mockFn{
		Base:   function.NewBase(function.NilConfig{}),
		name:   name,
		output: output,
		err:    err,
	}
}

func (f *mockFn) Execute(_ context.Context, _ interface{}) (interface{}, error) {
	return f.output, f.err
}

func (f *mockFn) InputSchema() *function.Schema {
	return &function.Schema{Type: "object", Properties: map[string]*function.Property{}}
}

func (f *mockFn) Metadata() *function.Metadata {
	m := function.NewMetadata(f.name)
	m.Description = f.name + " description"
	return m
}

// collectEvents drains an event channel with a timeout, returning all events.
func collectEvents(t *testing.T, ch <-chan agent.Event) []agent.Event {
	t.Helper()
	var events []agent.Event
	deadline := time.After(2 * time.Second)
	for {
		select {
		case e, ok := <-ch:
			if !ok {
				return events
			}
			events = append(events, e)
		case <-deadline:
			t.Fatal("timed out collecting events")
		}
	}
}

// ── Tests ─────────────────────────────────────────────────────────────────────

// TestPostTool_LargeResultRedirected verifies the critical property:
//
//   - EventToolResult carries the RAW (large) output so the UI sees it
//   - The model's history (what goes to LLM on the next call) contains only
//     the file reference, not the full 10 KB payload
func TestPostTool_LargeResultRedirected(t *testing.T) {
	const toolName = "big_search"
	largeResult := strings.Repeat("result-data ", 900) // ~10 800 bytes, over 8192 threshold

	// LLM: call 1 → request "big_search", call 2 → final text answer.
	mock := llm.NewMockToolCallingProvider(
		llm.MockToolCallingResponse{
			ToolCalls: []llm.ToolCall{{ID: "1", Name: toolName, Args: map[string]interface{}{}}},
		},
		llm.MockToolCallingResponse{
			Text: "Here is your answer.",
		},
	)

	a := newTestAgent()
	a.SetProvider(mock)
	a.SetHooks(hooks.LargeResult(8192, t.TempDir()))

	if err := a.RegisterTool(newMockFn(toolName, largeResult, nil)); err != nil {
		t.Fatalf("RegisterTool: %v", err)
	}

	events := collectEvents(t, a.BaseRun(context.Background(), "s1", "search for me", "", 5, false))

	// ── 1. EventToolResult must carry the full raw output ─────────────────
	var toolResult agent.ToolResultEvent
	for _, e := range events {
		if e.Type == agent.EventToolResult {
			toolResult = e.Data.(agent.ToolResultEvent)
		}
	}
	if toolResult.ToolName != toolName {
		t.Fatalf("did not receive EventToolResult for %q", toolName)
	}
	rawOutput, ok := toolResult.Output.(string)
	if !ok {
		t.Fatalf("expected string output, got %T", toolResult.Output)
	}
	if len(rawOutput) < 8192 {
		t.Errorf("EventToolResult.Output should be the full large result (len=%d)", len(rawOutput))
	}

	// ── 2. The second LLM call must have received the file reference ───────
	// mock.LastMessages is set by ChatWithTools. After PostTool fires, the
	// tool result message in history should be the file-reference string,
	// not the 10 KB payload.
	secondCallMsgs := mock.LastMessages
	var toolMsgContent string
	for _, m := range secondCallMsgs {
		if m.Role == "tool" && m.ToolName == toolName {
			toolMsgContent = m.Content
		}
	}
	if toolMsgContent == "" {
		t.Fatal("no tool message found in second LLM call")
	}
	if toolMsgContent == largeResult {
		t.Error("model received the full large result instead of a file reference")
	}
	if !strings.Contains(toolMsgContent, "saved to:") {
		t.Errorf("expected model to see file reference, got: %q", toolMsgContent[:min(80, len(toolMsgContent))])
	}

	// ── 3. EventResponse should be the final answer ───────────────────────
	var finalResponse string
	for _, e := range events {
		if e.Type == agent.EventResponse {
			finalResponse = e.Data.(agent.ResponseEvent).Content
		}
	}
	if finalResponse != "Here is your answer." {
		t.Errorf("unexpected final response: %q", finalResponse)
	}
}

// TestPreModel_TokenLimitTrims verifies that the TokenLimit PreModel hook
// actually reduces the message slice that reaches the LLM when history is large.
func TestPreModel_TokenLimitTrims(t *testing.T) {
	// Build a long initial history by pre-loading the session.
	a := newTestAgent()

	// We need a mock that always returns a text response.
	mock := llm.NewMockToolCallingProvider(
		llm.MockToolCallingResponse{Text: "answer1"},
		llm.MockToolCallingResponse{Text: "answer2"},
		llm.MockToolCallingResponse{Text: "answer3"},
	)
	a.SetProvider(mock)

	// Token limit of 500 tokens = ~2000 chars. Each message is ~1000 chars = ~250 tokens.
	// So two messages per call is the max.
	a.SetHooks(hooks.TokenLimit(500))

	bigMsg := strings.Repeat("x", 1000)

	// Run three turns to build up history.
	for i, msg := range []string{bigMsg, bigMsg, bigMsg} {
		session := "trim-test"
		ch := a.BaseRun(context.Background(), session, msg, "", 2, false)
		events := collectEvents(t, ch)
		_ = i
		_ = events
	}

	// On the third call, the mock recorded the messages it received.
	// With a 500-token limit and ~250 tokens per message, we should see
	// fewer messages than the full 6 (3 user + 3 assistant) history.
	msgCount := len(mock.LastMessages)
	if msgCount >= 6 {
		t.Errorf("expected PreModel to trim history; got %d messages on last call (want < 6)", msgCount)
	}
}

// TestPreTool_CanDenyExecution verifies that a PreTool hook returning
// proceed=false prevents tool execution and records a denial in history.
func TestPreTool_CanDenyExecution(t *testing.T) {
	const toolName = "dangerous_tool"

	mock := llm.NewMockToolCallingProvider(
		llm.MockToolCallingResponse{
			ToolCalls: []llm.ToolCall{{ID: "1", Name: toolName, Args: map[string]interface{}{}}},
		},
		llm.MockToolCallingResponse{Text: "OK, I won't use that tool."},
	)

	var toolExecuted bool
	a := newTestAgent()
	a.SetProvider(mock)
	a.SetHooks(agent.AgentHooks{
		PreTool: func(_ context.Context, call llm.ToolCall) (llm.ToolCall, bool, error) {
			return call, false, nil // always deny
		},
	})

	mockTool := newMockFn(toolName, "should not reach here", nil)
	// Wrap the mock to detect if Execute is called.
	if err := a.RegisterTool(&detectingFn{mockFn: mockTool, executed: &toolExecuted}); err != nil {
		t.Fatalf("RegisterTool: %v", err)
	}

	events := collectEvents(t, a.BaseRun(context.Background(), "s1", "use the tool", "", 5, false))

	if toolExecuted {
		t.Error("tool should not have executed when PreTool denied it")
	}

	// EventToolResult should carry a denial error.
	var toolResultEvent agent.ToolResultEvent
	for _, e := range events {
		if e.Type == agent.EventToolResult {
			toolResultEvent = e.Data.(agent.ToolResultEvent)
		}
	}
	if toolResultEvent.Error == nil {
		t.Error("expected EventToolResult to carry a denial error")
	}
}

// TestPostResponse_FiresWithFinalAnswer verifies PostResponse receives the
// agent's final text response.
func TestPostResponse_FiresWithFinalAnswer(t *testing.T) {
	const answer = "The answer is 42."

	mock := llm.NewMockToolCallingProvider(
		llm.MockToolCallingResponse{Text: answer},
	)

	var receivedResponse string
	a := newTestAgent()
	a.SetProvider(mock)
	a.SetHooks(agent.AgentHooks{
		PostResponse: func(_ context.Context, response string) error {
			receivedResponse = response
			return nil
		},
	})

	collectEvents(t, a.BaseRun(context.Background(), "s1", "what is the answer?", "", 2, false))

	if receivedResponse != answer {
		t.Errorf("PostResponse: expected %q, got %q", answer, receivedResponse)
	}
}

// TestPostResponse_ErrorIsNonFatal verifies that a PostResponse error does not
// suppress the EventResponse or EventComplete events.
func TestPostResponse_ErrorIsNonFatal(t *testing.T) {
	mock := llm.NewMockToolCallingProvider(
		llm.MockToolCallingResponse{Text: "here is the answer"},
	)

	a := newTestAgent()
	a.SetProvider(mock)
	a.SetHooks(agent.AgentHooks{
		PostResponse: func(_ context.Context, _ string) error {
			return errors.New("post-response hook failed")
		},
	})

	events := collectEvents(t, a.BaseRun(context.Background(), "s1", "hi", "", 2, false))

	var sawResponse, sawComplete, sawError bool
	for _, e := range events {
		switch e.Type {
		case agent.EventResponse:
			sawResponse = true
		case agent.EventComplete:
			sawComplete = true
		case agent.EventError:
			sawError = true
		}
	}

	if !sawResponse {
		t.Error("expected EventResponse even when PostResponse hook fails")
	}
	if !sawComplete {
		t.Error("expected EventComplete even when PostResponse hook fails")
	}
	if !sawError {
		t.Error("expected EventError to carry the PostResponse hook failure")
	}
}

// ── helpers ───────────────────────────────────────────────────────────────────

// detectingFn wraps a mockFn and sets *executed when Execute is called.
type detectingFn struct {
	*mockFn
	executed *bool
}

func (f *detectingFn) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	*f.executed = true
	return f.mockFn.Execute(ctx, input)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
