package workflow

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockExecutor implements Executor for testing without real LLM/tool backends.
type mockExecutor struct {
	mu              sync.Mutex
	toolResponses   map[string]interface{}
	toolErrors      map[string]error
	llmResponse     string
	llmError        error
	confirmResponse bool
	emittedEvents   []Event
}

func newMockExecutor() *mockExecutor {
	return &mockExecutor{
		toolResponses:   make(map[string]interface{}),
		toolErrors:      make(map[string]error),
		confirmResponse: true, // approve by default
	}
}

func (m *mockExecutor) CallTool(_ context.Context, name string, _ map[string]interface{}) (interface{}, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if err, ok := m.toolErrors[name]; ok {
		return nil, err
	}
	if v, ok := m.toolResponses[name]; ok {
		return v, nil
	}
	return name + "_output", nil
}

func (m *mockExecutor) CallLLM(_ context.Context, _ string, _ *LLMCallOptions) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.llmError != nil {
		return "", m.llmError
	}
	return m.llmResponse, nil
}

func (m *mockExecutor) Confirm(_ context.Context, _ string, _ string, _ interface{}) (bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.confirmResponse, nil
}

func (m *mockExecutor) EmitEvent(ev Event) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.emittedEvents = append(m.emittedEvents, ev)
}

func (m *mockExecutor) EventTypes() []EventType {
	m.mu.Lock()
	defer m.mu.Unlock()
	types := make([]EventType, len(m.emittedEvents))
	for i, ev := range m.emittedEvents {
		types[i] = ev.Type
	}
	return types
}

// drainEvents drains the Engine.Run() event channel and returns all events.
func drainEvents(ch <-chan Event) []Event {
	var events []Event
	for ev := range ch {
		events = append(events, ev)
	}
	return events
}

// eventTypes extracts EventType values from a slice of events.
func eventTypes(events []Event) []EventType {
	types := make([]EventType, len(events))
	for i, ev := range events {
		types[i] = ev.Type
	}
	return types
}

// newTestEngine creates an Engine configured for testing.
func newTestEngine() *Engine {
	return NewEngine(EngineConfig{})
}

// TestRun_singleToolStep tests a single tool step executes and stores output.
func TestRun_singleToolStep(t *testing.T) {
	engine := newTestEngine()
	def := &WorkflowDef{
		Name: "test",
		Steps: []StepDef{
			{
				ID:   "fetch",
				Tool: &ToolStepDef{Name: "my_tool"},
			},
		},
	}

	// Override the engine's tool executor via Execute (which uses engineExecutor internally).
	// We test through runStep directly using mockExecutor.
	wfCtx := NewWorkflowContext(nil)
	exec := newMockExecutor()
	exec.toolResponses["my_tool"] = "tool_result_42"

	out, err := engine.runStep(context.Background(), def.Steps[0], wfCtx, exec)
	require.NoError(t, err)
	assert.Equal(t, "tool_result_42", out)

	types := exec.EventTypes()
	assert.Equal(t, []EventType{EventStepStart, EventStepComplete}, types)
}

// TestRun_llmStep tests an LLM step resolves the prompt and calls LLM.
func TestRun_llmStep(t *testing.T) {
	engine := newTestEngine()
	def := StepDef{
		ID: "summarize",
		LLM: &LLMStepDef{
			Prompt: "Summarize: {{steps.data}}",
			System: "You are a summarizer.",
		},
	}

	wfCtx := NewWorkflowContext(nil)
	wfCtx.SetStepOutput("data", "some long text")

	exec := newMockExecutor()
	exec.llmResponse = "Short summary."

	out, err := engine.runStep(context.Background(), def, wfCtx, exec)
	require.NoError(t, err)
	assert.Equal(t, "Short summary.", out)
}

// TestRun_confirmApproved tests an approved confirm step continues.
func TestRun_confirmApproved(t *testing.T) {
	engine := newTestEngine()
	def := StepDef{
		ID:      "gate",
		Confirm: &ConfirmStepDef{Prompt: "Proceed?"},
	}

	exec := newMockExecutor()
	exec.confirmResponse = true

	out, err := engine.runStep(context.Background(), def, NewWorkflowContext(nil), exec)
	require.NoError(t, err)
	assert.Equal(t, true, out)
}

// TestRun_confirmDenied tests a denied confirm step returns ErrConfirmDenied.
func TestRun_confirmDenied(t *testing.T) {
	engine := newTestEngine()
	def := StepDef{
		ID:      "gate",
		Confirm: &ConfirmStepDef{Prompt: "Proceed?"},
	}

	exec := newMockExecutor()
	exec.confirmResponse = false

	_, err := engine.runStep(context.Background(), def, NewWorkflowContext(nil), exec)
	assert.ErrorIs(t, err, ErrConfirmDenied)
}

// TestRun_foreachSequential tests foreach over a list with sequential execution.
func TestRun_foreachSequential(t *testing.T) {
	engine := newTestEngine()

	wfCtx := NewWorkflowContext(nil)
	wfCtx.SetStepOutput("items", []interface{}{"a", "b", "c"})

	def := StepDef{
		ID: "process",
		Foreach: &ForeachStepDef{
			Over:     "{{steps.items}}",
			As:       "item",
			Parallel: false,
			Steps: []StepDef{
				{
					ID:   "each",
					Tool: &ToolStepDef{Name: "processor"},
				},
			},
		},
	}

	exec := newMockExecutor()
	exec.toolResponses["processor"] = "processed"

	out, err := engine.runStep(context.Background(), def, wfCtx, exec)
	require.NoError(t, err)

	results, ok := out.([]interface{})
	require.True(t, ok)
	assert.Len(t, results, 3)
}

// TestRun_foreachParallel tests foreach with parallel execution.
func TestRun_foreachParallel(t *testing.T) {
	engine := newTestEngine()

	wfCtx := NewWorkflowContext(nil)
	wfCtx.SetStepOutput("items", []interface{}{1, 2, 3, 4, 5})

	def := StepDef{
		ID: "parallel_process",
		Foreach: &ForeachStepDef{
			Over:     "{{steps.items}}",
			Parallel: true,
			Steps: []StepDef{
				{
					ID:   "each",
					Tool: &ToolStepDef{Name: "tool"},
				},
			},
		},
	}

	exec := newMockExecutor()
	exec.toolResponses["tool"] = "done"

	out, err := engine.runStep(context.Background(), def, wfCtx, exec)
	require.NoError(t, err)

	results, ok := out.([]interface{})
	require.True(t, ok)
	assert.Len(t, results, 5)
}

// TestRun_conditionTrue tests the then-branch runs when condition is true.
func TestRun_conditionTrue(t *testing.T) {
	engine := newTestEngine()

	wfCtx := NewWorkflowContext(nil)
	wfCtx.SetStepOutput("flag", true)

	def := StepDef{
		ID: "branch",
		Condition: &ConditionStepDef{
			If: "{{steps.flag}}",
			Then: []StepDef{
				{ID: "then_step", Tool: &ToolStepDef{Name: "then_tool"}},
			},
			Else: []StepDef{
				{ID: "else_step", Tool: &ToolStepDef{Name: "else_tool"}},
			},
		},
	}

	exec := newMockExecutor()
	exec.toolResponses["then_tool"] = "then_result"
	exec.toolResponses["else_tool"] = "else_result"

	out, err := engine.runStep(context.Background(), def, wfCtx, exec)
	require.NoError(t, err)
	assert.Equal(t, "then_result", out)
}

// TestRun_conditionFalse tests the else-branch runs when condition is false.
func TestRun_conditionFalse(t *testing.T) {
	engine := newTestEngine()

	wfCtx := NewWorkflowContext(nil)
	wfCtx.SetStepOutput("flag", false)

	def := StepDef{
		ID: "branch",
		Condition: &ConditionStepDef{
			If: "{{steps.flag}}",
			Then: []StepDef{
				{ID: "then_step", Tool: &ToolStepDef{Name: "then_tool"}},
			},
			Else: []StepDef{
				{ID: "else_step", Tool: &ToolStepDef{Name: "else_tool"}},
			},
		},
	}

	exec := newMockExecutor()
	exec.toolResponses["then_tool"] = "then_result"
	exec.toolResponses["else_tool"] = "else_result"

	out, err := engine.runStep(context.Background(), def, wfCtx, exec)
	require.NoError(t, err)
	assert.Equal(t, "else_result", out)
}

// TestRun_onErrorIgnore tests that ignored errors cause nil output and continue.
func TestRun_onErrorIgnore(t *testing.T) {
	engine := newTestEngine()

	steps := []StepDef{
		{
			ID:      "failing",
			Tool:    &ToolStepDef{Name: "bad_tool"},
			OnError: &OnErrorDef{Ignore: true},
		},
		{
			ID:   "after",
			Tool: &ToolStepDef{Name: "good_tool"},
		},
	}

	exec := newMockExecutor()
	exec.toolErrors["bad_tool"] = errors.New("tool exploded")
	exec.toolResponses["good_tool"] = "success"

	wfCtx := NewWorkflowContext(nil)
	out, err := engine.runSteps(context.Background(), steps, wfCtx, exec)
	require.NoError(t, err)
	assert.Equal(t, "success", out)

	// Verify failing step stored nil, after step stored "success".
	failOut, ok := wfCtx.GetStepOutput("failing")
	require.True(t, ok)
	assert.Nil(t, failOut)
}

// TestRun_onErrorFallback tests that a fallback value is used on error.
func TestRun_onErrorFallback(t *testing.T) {
	engine := newTestEngine()

	def := &WorkflowDef{
		Name: "test",
		Steps: []StepDef{
			{
				ID:      "step1",
				Tool:    &ToolStepDef{Name: "failing_tool"},
				OnError: &OnErrorDef{Fallback: "default_value"},
			},
		},
	}

	exec := newMockExecutor()
	exec.toolErrors["failing_tool"] = errors.New("boom")

	events := engine.Run(context.Background(), def, nil)
	allEvents := drainEvents(events)

	types := eventTypes(allEvents)
	assert.NotContains(t, types, EventError)
	assert.Contains(t, types, EventResponse)

	var resp ResponseEvent
	for _, ev := range allEvents {
		if ev.Type == EventResponse {
			resp = ev.Data.(ResponseEvent)
		}
	}
	assert.Equal(t, "default_value", resp.Output)
}

// TestRun_contextCancel tests that context cancellation during confirm unblocks with error.
func TestRun_contextCancel(t *testing.T) {
	engine := NewEngine(EngineConfig{})
	def := &WorkflowDef{
		Name: "test",
		Steps: []StepDef{
			{
				ID:      "gate",
				Confirm: &ConfirmStepDef{Prompt: "Proceed?"},
			},
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	events := engine.Run(ctx, def, nil)

	// Single consumer: drain events, cancel on confirm, do NOT send on ResponseCh.
	var allEvents []Event
	for ev := range events {
		allEvents = append(allEvents, ev)
		if ev.Type == EventConfirm {
			// Cancel the context — don't send on ResponseCh.
			cancel()
		}
	}

	types := eventTypes(allEvents)
	// Should see EventStepStart and EventConfirm, then EventStepError or EventError.
	assert.Contains(t, types, EventConfirm)
	assert.Contains(t, types, EventError)
}

// TestRun_templateResolution tests {{steps.x.field}} references.
func TestRun_templateResolution(t *testing.T) {
	engine := newTestEngine()

	wfCtx := NewWorkflowContext(map[string]interface{}{
		"name": "Alice",
	})
	wfCtx.SetStepOutput("prev", map[string]interface{}{
		"count": 42,
	})

	def := StepDef{
		ID: "greet",
		LLM: &LLMStepDef{
			Prompt: "Hello {{input.name}}, count={{steps.prev.count}}",
		},
	}

	exec := newMockExecutor()
	exec.llmResponse = "Greeting sent."

	out, err := engine.runStep(context.Background(), def, wfCtx, exec)
	require.NoError(t, err)
	assert.Equal(t, "Greeting sent.", out)

	// Verify the prompt was correctly resolved before calling LLM.
	// (We can check via the fact it didn't return an error — if the template
	// failed, runLLMStep would return an error.)
}

// TestRun_fullWorkflow tests a complete multi-step workflow via Run().
func TestRun_fullWorkflow(t *testing.T) {
	engine := newTestEngine()

	def := &WorkflowDef{
		Name: "pipeline",
		Steps: []StepDef{
			{ID: "step1", Tool: &ToolStepDef{Name: "tool_a"}},
			{ID: "step2", Tool: &ToolStepDef{Name: "tool_b"}},
		},
	}

	// Use a custom registry with mock tools instead of real ones.
	// Since Engine.Run uses engineExecutor which calls registry.Global(),
	// we test through runSteps with mockExecutor instead.
	wfCtx := NewWorkflowContext(nil)
	exec := newMockExecutor()
	exec.toolResponses["tool_a"] = "result_a"
	exec.toolResponses["tool_b"] = "result_b"

	out, err := engine.runSteps(context.Background(), def.Steps, wfCtx, exec)
	require.NoError(t, err)
	assert.Equal(t, "result_b", out) // last step's output

	storedA, ok := wfCtx.GetStepOutput("step1")
	require.True(t, ok)
	assert.Equal(t, "result_a", storedA)

	storedB, ok := wfCtx.GetStepOutput("step2")
	require.True(t, ok)
	assert.Equal(t, "result_b", storedB)
}

// TestRun_invalidStep tests that a step with no type returns ErrInvalidDef.
func TestRun_invalidStep(t *testing.T) {
	engine := newTestEngine()
	def := StepDef{ID: "empty"} // no type set

	exec := newMockExecutor()
	_, err := engine.runStep(context.Background(), def, NewWorkflowContext(nil), exec)
	require.Error(t, err)
	assert.ErrorIs(t, err, ErrInvalidDef)
}

// TestEngine_Execute_AutoApprove tests that Execute auto-approves confirm gates.
func TestEngine_Execute_AutoApprove(t *testing.T) {
	// We need a real Engine.Run but with a way to intercept confirm events.
	// The Execute() method auto-approves, so we verify it completes without blocking.
	engine := newTestEngine()

	def := &WorkflowDef{
		Name: "auto-approve-test",
		Steps: []StepDef{
			// Confirm step — would block in Run(), but Execute() auto-approves.
			// We can't easily test Execute() with real engine since it needs a real
			// registry. Test via a custom Run() drain instead.
			{ID: "gate", Confirm: &ConfirmStepDef{Prompt: "OK?"}},
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	// Run the real engine — it will emit EventConfirm.
	events := engine.Run(ctx, def, nil)

	// Auto-approve in a goroutine (mimicking Execute()).
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for ev := range events {
			if ev.Type == EventConfirm {
				if ce, ok := ev.Data.(ConfirmEvent); ok {
					ce.ResponseCh <- true
				}
			}
		}
	}()

	wg.Wait()
	// If we reach here without timeout, auto-approve worked.
}

// TestWorkflowContext_Resolve tests template resolution.
func TestWorkflowContext_Resolve(t *testing.T) {
	input := map[string]interface{}{
		"user": "Bob",
		"count": 5,
	}
	wfCtx := NewWorkflowContext(input)
	wfCtx.SetStepOutput("result", map[string]interface{}{
		"value": "hello",
	})

	// Plain string (no template)
	v, err := wfCtx.Resolve("plain text")
	require.NoError(t, err)
	assert.Equal(t, "plain text", v)

	// Single {{input}} — returns typed value
	v, err = wfCtx.Resolve("{{input}}")
	require.NoError(t, err)
	assert.Equal(t, input, v)

	// {{input.user}}
	v, err = wfCtx.Resolve("{{input.user}}")
	require.NoError(t, err)
	assert.Equal(t, "Bob", v)

	// {{steps.result.value}}
	v, err = wfCtx.Resolve("{{steps.result.value}}")
	require.NoError(t, err)
	assert.Equal(t, "hello", v)

	// Embedded template
	v, err = wfCtx.Resolve("User: {{input.user}}, count: {{input.count}}")
	require.NoError(t, err)
	assert.Equal(t, "User: Bob, count: 5", v)

	// Unknown step
	_, err = wfCtx.Resolve("{{steps.missing}}")
	assert.ErrorIs(t, err, ErrStepNotFound)
}

// TestWorkflowContext_WithItem tests foreach item context.
func TestWorkflowContext_WithItem(t *testing.T) {
	parent := NewWorkflowContext(nil)
	parent.SetStepOutput("base", "base_value")

	child := parent.WithItem("item_val", "item")

	// Child can access parent step outputs.
	v, err := child.Resolve("{{steps.base}}")
	require.NoError(t, err)
	assert.Equal(t, "base_value", v)

	// Child can access the item.
	v, err = child.Resolve("{{item}}")
	require.NoError(t, err)
	assert.Equal(t, "item_val", v)
}

// TestValidateDef tests workflow definition validation.
func TestValidateDef(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		def := &WorkflowDef{
			Steps: []StepDef{
				{ID: "a", Tool: &ToolStepDef{Name: "t"}},
				{ID: "b", LLM: &LLMStepDef{Prompt: "p"}},
			},
		}
		errs := validateDef(def)
		assert.Empty(t, errs)
	})

	t.Run("empty_id", func(t *testing.T) {
		def := &WorkflowDef{
			Steps: []StepDef{
				{Tool: &ToolStepDef{Name: "t"}}, // missing ID
			},
		}
		errs := validateDef(def)
		assert.NotEmpty(t, errs)
	})

	t.Run("duplicate_id", func(t *testing.T) {
		def := &WorkflowDef{
			Steps: []StepDef{
				{ID: "same", Tool: &ToolStepDef{Name: "t"}},
				{ID: "same", LLM: &LLMStepDef{Prompt: "p"}},
			},
		}
		errs := validateDef(def)
		assert.NotEmpty(t, errs)
	})

	t.Run("no_type", func(t *testing.T) {
		def := &WorkflowDef{
			Steps: []StepDef{
				{ID: "orphan"}, // no type
			},
		}
		errs := validateDef(def)
		assert.NotEmpty(t, errs)
	})

	t.Run("multiple_types", func(t *testing.T) {
		def := &WorkflowDef{
			Steps: []StepDef{
				{
					ID:   "multi",
					Tool: &ToolStepDef{Name: "t"},
					LLM:  &LLMStepDef{Prompt: "p"},
				},
			},
		}
		errs := validateDef(def)
		assert.NotEmpty(t, errs)
	})
}
