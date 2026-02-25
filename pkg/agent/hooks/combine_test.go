package hooks_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/pvjammer/ai-sdk-go/pkg/agent"
	"github.com/pvjammer/ai-sdk-go/pkg/agent/hooks"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// appendHook returns an AgentHooks that appends suffix to the message/result.
func appendHook(suffix string) agent.AgentHooks {
	return agent.AgentHooks{
		PreInput: func(_ context.Context, _, msg string) (string, error) {
			return msg + suffix, nil
		},
		PostTool: func(_ context.Context, _, result string, _ error) (string, error) {
			return result + suffix, nil
		},
	}
}

func TestCombine_PreInputPipeline(t *testing.T) {
	// Two hooks appending "+A" and "+B" in order → result should be "msg+A+B".
	h := hooks.Combine(appendHook("+A"), appendHook("+B"))
	got, err := h.PreInput(context.Background(), "s", "msg")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "msg+A+B" {
		t.Errorf("expected %q, got %q", "msg+A+B", got)
	}
}

func TestCombine_PostToolPipeline(t *testing.T) {
	h := hooks.Combine(appendHook("-1"), appendHook("-2"))
	got, err := h.PostTool(context.Background(), "tool", "result", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "result-1-2" {
		t.Errorf("expected %q, got %q", "result-1-2", got)
	}
}

func TestCombine_PreInputShortCircuitsOnError(t *testing.T) {
	sentinel := errors.New("hook error")
	failHook := agent.AgentHooks{
		PreInput: func(_ context.Context, _, msg string) (string, error) {
			return msg, sentinel
		},
	}
	var secondRan bool
	secondHook := agent.AgentHooks{
		PreInput: func(_ context.Context, _, msg string) (string, error) {
			secondRan = true
			return msg + "+second", nil
		},
	}
	h := hooks.Combine(failHook, secondHook)
	_, err := h.PreInput(context.Background(), "s", "msg")
	if !errors.Is(err, sentinel) {
		t.Errorf("expected sentinel error, got %v", err)
	}
	if secondRan {
		t.Error("second hook should not run after first returns error")
	}
}

func TestCombine_PreToolShortCircuitsOnDeny(t *testing.T) {
	denyHook := agent.AgentHooks{
		PreTool: func(_ context.Context, call llm.ToolCall) (llm.ToolCall, bool, error) {
			return call, false, nil // deny
		},
	}
	var secondRan bool
	secondHook := agent.AgentHooks{
		PreTool: func(_ context.Context, call llm.ToolCall) (llm.ToolCall, bool, error) {
			secondRan = true
			return call, true, nil
		},
	}
	h := hooks.Combine(denyHook, secondHook)
	_, proceed, err := h.PreTool(context.Background(), llm.ToolCall{Name: "x"})
	if proceed {
		t.Error("expected proceed=false after denial")
	}
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if secondRan {
		t.Error("second PreTool hook should not run after denial")
	}
}

func TestCombine_PostToolExecErrConsumedByFirstHook(t *testing.T) {
	// The first PostTool hook receives execErr; subsequent hooks receive nil
	// (the error has been "consumed" by the first hook which encoded it in the result).
	var (
		firstSawErr  error
		secondSawErr error
	)
	first := agent.AgentHooks{
		PostTool: func(_ context.Context, _, result string, execErr error) (string, error) {
			firstSawErr = execErr
			return result, nil // first hook handles it and returns normally
		},
	}
	second := agent.AgentHooks{
		PostTool: func(_ context.Context, _, result string, execErr error) (string, error) {
			secondSawErr = execErr
			return result, nil
		},
	}
	h := hooks.Combine(first, second)
	origErr := errors.New("tool failed")
	h.PostTool(context.Background(), "t", "result", origErr)

	if firstSawErr != origErr {
		t.Errorf("first hook: expected execErr=%v, got %v", origErr, firstSawErr)
	}
	if secondSawErr != nil {
		t.Errorf("second hook: expected execErr=nil (consumed), got %v", secondSawErr)
	}
}

func TestCombine_PostResponseAllHooksRun(t *testing.T) {
	// Unlike PreTool, PostResponse hooks all run regardless of errors.
	var ran []string
	mk := func(name string, fail bool) agent.AgentHooks {
		return agent.AgentHooks{
			PostResponse: func(_ context.Context, _ string) error {
				ran = append(ran, name)
				if fail {
					return errors.New("hook " + name + " failed")
				}
				return nil
			},
		}
	}
	h := hooks.Combine(mk("A", true), mk("B", false), mk("C", true))
	err := h.PostResponse(context.Background(), "response")
	// All three should have run.
	if strings.Join(ran, ",") != "A,B,C" {
		t.Errorf("expected all PostResponse hooks to run, got order: %v", ran)
	}
	// Last non-nil error is returned.
	if err == nil {
		t.Error("expected a non-nil error from failing PostResponse hooks")
	}
}

func TestCombine_NilHooksSkipped(t *testing.T) {
	// A hook with only PreInput set should leave PostTool nil.
	// Combining it with an empty hook should not panic.
	h := hooks.Combine(
		agent.AgentHooks{PreInput: func(_ context.Context, _, m string) (string, error) {
			return m + "!", nil
		}},
		agent.AgentHooks{}, // all nil
	)
	got, err := h.PreInput(context.Background(), "s", "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "hello!" {
		t.Errorf("expected %q, got %q", "hello!", got)
	}
	// PostTool should be non-nil (Combine always sets all fields) but safely
	// pass through since both constituent hooks were nil.
	result, err := h.PostTool(context.Background(), "t", "result", nil)
	if err != nil {
		t.Fatalf("unexpected error from nil PostTool: %v", err)
	}
	if result != "result" {
		t.Errorf("expected result unchanged, got %q", result)
	}
}
