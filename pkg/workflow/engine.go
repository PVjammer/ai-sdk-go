package workflow

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/pvjammer/ai-sdk-go/pkg/function"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
	"github.com/pvjammer/ai-sdk-go/pkg/registry"
)

// EngineConfig configures an Engine instance.
type EngineConfig struct {
	// LLMProvider is the LLM backend used by LLM steps.
	// If nil, LLM steps will return an error (no LLM available).
	LLMProvider llm.Provider

	// DefaultModel is the model name used for LLM steps that don't specify one.
	DefaultModel string

	// DefaultMaxTokens is the token limit for LLM steps (default: 2048).
	DefaultMaxTokens int

	// DefaultTemperature is the sampling temperature for LLM steps (default: 0.7).
	DefaultTemperature float64

	// FunctionRegistry is the registry used to resolve tool step names.
	// Defaults to registry.Global() if nil.
	FunctionRegistry *registry.Registry

	// FunctionConfig is passed to tool instances when they are created.
	FunctionConfig function.Config
}

// Engine executes WorkflowDefs.
//
// Engine.Run() spawns a goroutine and returns a buffered event channel.
// Engine.Execute() is a synchronous adapter that auto-approves confirm gates.
type Engine struct {
	cfg EngineConfig
}

// NewEngine creates a new Engine with the given configuration.
func NewEngine(cfg EngineConfig) *Engine {
	if cfg.DefaultMaxTokens == 0 {
		cfg.DefaultMaxTokens = 2048
	}
	if cfg.DefaultTemperature == 0 {
		cfg.DefaultTemperature = 0.7
	}
	if cfg.FunctionConfig == nil {
		cfg.FunctionConfig = function.NilConfig{}
	}
	return &Engine{cfg: cfg}
}

// Run executes a WorkflowDef and returns a buffered event channel (capacity 100).
//
// The channel is closed when the workflow completes or a fatal error occurs.
// Callers MUST drain the channel: confirm gates block the goroutine (not the channel),
// but the goroutine cannot proceed until the confirm response is sent.
//
// Pattern mirrors AgentBase.BaseRun() in pkg/agent/base.go.
func (e *Engine) Run(ctx context.Context, def *WorkflowDef, input interface{}) <-chan Event {
	events := make(chan Event, 100)

	go func() {
		defer close(events)

		wfCtx := NewWorkflowContext(input)
		engineExec := &engineExecutor{engine: e, events: events}

		for _, step := range def.Steps {
			out, err := e.runStep(ctx, step, wfCtx, engineExec)
			if err != nil {
				if step.OnError != nil && handleOnError(ctx, step.OnError, err, &out, step, wfCtx, engineExec) {
					wfCtx.SetStepOutput(step.ID, out)
					continue
				}
				engineExec.EmitEvent(Event{
					Type:      EventError,
					Data:      ErrorEvent{Err: err, Message: fmt.Sprintf("step %q failed: %v", step.ID, err)},
					Timestamp: time.Now(),
				})
				return
			}
			wfCtx.SetStepOutput(step.ID, out)
		}

		finalOut := wfCtx.GetLastOutput(def.Steps)
		engineExec.EmitEvent(Event{
			Type:      EventResponse,
			Data:      ResponseEvent{Output: finalOut},
			Timestamp: time.Now(),
		})
		engineExec.EmitEvent(Event{
			Type:      EventComplete,
			Timestamp: time.Now(),
		})
	}()

	return events
}

// Execute is a synchronous adapter over Run().
//
// Confirm gates are auto-approved (returns true). Use for programmatic or test
// invocation where interactive confirmation is not needed.
func (e *Engine) Execute(ctx context.Context, def *WorkflowDef, input interface{}) (interface{}, error) {
	events := e.Run(ctx, def, input)

	var finalOutput interface{}
	var finalErr error

	for ev := range events {
		switch ev.Type {
		case EventConfirm:
			// Auto-approve confirm gates in synchronous mode.
			if ce, ok := ev.Data.(ConfirmEvent); ok {
				ce.ResponseCh <- true
			}
		case EventResponse:
			if re, ok := ev.Data.(ResponseEvent); ok {
				finalOutput = re.Output
			}
		case EventError:
			if ee, ok := ev.Data.(ErrorEvent); ok {
				finalErr = ee.Err
			}
		}
	}

	return finalOutput, finalErr
}

// runStep dispatches a single step to the appropriate handler.
func (e *Engine) runStep(ctx context.Context, step StepDef, wfCtx *WorkflowContext, exec Executor) (interface{}, error) {
	// Validate step has exactly one type.
	if count := step.typeCount(); count == 0 {
		return nil, fmt.Errorf("%w: step %q has no type (tool/llm/confirm/foreach/condition)", ErrInvalidDef, step.ID)
	} else if count > 1 {
		return nil, fmt.Errorf("%w: step %q has multiple types (only one allowed)", ErrInvalidDef, step.ID)
	}

	exec.EmitEvent(Event{
		Type: EventStepStart,
		Data: StepStartEvent{
			StepID:   step.ID,
			StepType: step.stepType(),
		},
		Timestamp: time.Now(),
	})

	var out interface{}
	var err error

	switch {
	case step.Tool != nil:
		out, err = e.runToolStep(ctx, step.ID, step.Tool, wfCtx, exec)
	case step.LLM != nil:
		out, err = e.runLLMStep(ctx, step.ID, step.LLM, wfCtx, exec)
	case step.Confirm != nil:
		out, err = e.runConfirmStep(ctx, step.ID, step.Confirm, wfCtx, exec)
	case step.Foreach != nil:
		out, err = e.runForeachStep(ctx, step.ID, step.Foreach, wfCtx, exec)
	case step.Condition != nil:
		out, err = e.runConditionStep(ctx, step.ID, step.Condition, wfCtx, exec)
	}

	if err != nil {
		exec.EmitEvent(Event{
			Type:      EventStepError,
			Data:      StepErrorEvent{StepID: step.ID, Attempt: 1, Err: err},
			Timestamp: time.Now(),
		})
		return nil, err
	}

	exec.EmitEvent(Event{
		Type:      EventStepComplete,
		Data:      StepCompleteEvent{StepID: step.ID, Output: out},
		Timestamp: time.Now(),
	})

	return out, nil
}

// runToolStep executes a tool step.
func (e *Engine) runToolStep(ctx context.Context, stepID string, def *ToolStepDef, wfCtx *WorkflowContext, exec Executor) (interface{}, error) {
	resolved, err := wfCtx.ResolveMap(def.With)
	if err != nil {
		return nil, fmt.Errorf("%w: step %q: resolving 'with': %v", ErrStepFailed, stepID, err)
	}
	return exec.CallTool(ctx, def.Name, resolved)
}

// runLLMStep executes an LLM step.
func (e *Engine) runLLMStep(ctx context.Context, stepID string, def *LLMStepDef, wfCtx *WorkflowContext, exec Executor) (interface{}, error) {
	prompt, err := wfCtx.ResolveString(def.Prompt)
	if err != nil {
		return nil, fmt.Errorf("%w: step %q: resolving prompt: %v", ErrStepFailed, stepID, err)
	}

	system, err := wfCtx.ResolveString(def.System)
	if err != nil {
		return nil, fmt.Errorf("%w: step %q: resolving system prompt: %v", ErrStepFailed, stepID, err)
	}

	opts := &LLMCallOptions{
		Model:        def.Model,
		MaxTokens:    def.MaxTokens,
		Temperature:  def.Temperature,
		SystemPrompt: system,
	}

	return exec.CallLLM(ctx, prompt, opts)
}

// runConfirmStep executes a confirm (gate) step.
func (e *Engine) runConfirmStep(ctx context.Context, stepID string, def *ConfirmStepDef, wfCtx *WorkflowContext, exec Executor) (interface{}, error) {
	prompt, err := wfCtx.ResolveString(def.Prompt)
	if err != nil {
		return nil, fmt.Errorf("%w: step %q: resolving confirm prompt: %v", ErrStepFailed, stepID, err)
	}

	approved, err := exec.Confirm(ctx, stepID, prompt, nil)
	if err != nil {
		return nil, err
	}
	if !approved {
		return nil, ErrConfirmDenied
	}
	return true, nil
}

// runForeachStep executes a foreach step over a resolved list.
func (e *Engine) runForeachStep(ctx context.Context, stepID string, def *ForeachStepDef, wfCtx *WorkflowContext, exec Executor) (interface{}, error) {
	raw, err := wfCtx.Resolve(def.Over)
	if err != nil {
		return nil, fmt.Errorf("%w: step %q: resolving 'over': %v", ErrStepFailed, stepID, err)
	}

	items, ok := raw.([]interface{})
	if !ok {
		// Try converting a non-slice value into a single-element list.
		return nil, fmt.Errorf("%w: step %q: 'over' must resolve to []interface{}, got %T", ErrStepFailed, stepID, raw)
	}

	as := def.As
	if as == "" {
		as = "item"
	}

	results := make([]interface{}, len(items))

	if def.Parallel {
		return e.runForeachParallel(ctx, stepID, def, wfCtx, exec, items, as, results)
	}

	// Sequential execution.
	for i, item := range items {
		childCtx := wfCtx.WithItem(item, as)
		iterResult, iterErr := e.runSteps(ctx, def.Steps, childCtx, exec)
		if iterErr != nil {
			return nil, fmt.Errorf("%w: step %q[%d]: %v", ErrStepFailed, stepID, i, iterErr)
		}
		results[i] = iterResult
	}

	return results, nil
}

// runForeachParallel executes foreach iterations concurrently.
func (e *Engine) runForeachParallel(ctx context.Context, stepID string, def *ForeachStepDef, wfCtx *WorkflowContext, exec Executor, items []interface{}, as string, results []interface{}) (interface{}, error) {
	cancelCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var (
		wg      sync.WaitGroup
		mu      sync.Mutex
		firstErr error
	)

	for i, item := range items {
		i, item := i, item // capture loop vars
		wg.Add(1)
		go func() {
			defer wg.Done()

			childCtx := wfCtx.WithItem(item, as)
			iterResult, iterErr := e.runSteps(cancelCtx, def.Steps, childCtx, exec)

			mu.Lock()
			defer mu.Unlock()
			if iterErr != nil {
				if firstErr == nil {
					firstErr = fmt.Errorf("%w: step %q[%d]: %v", ErrStepFailed, stepID, i, iterErr)
					cancel()
				}
				return
			}
			results[i] = iterResult
		}()
	}

	wg.Wait()

	if firstErr != nil {
		return nil, firstErr
	}
	return results, nil
}

// runConditionStep evaluates the condition and runs the appropriate branch.
func (e *Engine) runConditionStep(ctx context.Context, stepID string, def *ConditionStepDef, wfCtx *WorkflowContext, exec Executor) (interface{}, error) {
	condVal, err := wfCtx.Resolve(def.If)
	if err != nil {
		return nil, fmt.Errorf("%w: step %q: resolving condition: %v", ErrStepFailed, stepID, err)
	}

	if coerceBool(condVal) {
		if len(def.Then) == 0 {
			return true, nil
		}
		return e.runSteps(ctx, def.Then, wfCtx, exec)
	}

	if len(def.Else) == 0 {
		return false, nil
	}
	return e.runSteps(ctx, def.Else, wfCtx, exec)
}

// runSteps executes a slice of steps sequentially, returning the last step's output.
func (e *Engine) runSteps(ctx context.Context, steps []StepDef, wfCtx *WorkflowContext, exec Executor) (interface{}, error) {
	var last interface{}
	for _, step := range steps {
		out, err := e.runStep(ctx, step, wfCtx, exec)
		if err != nil {
			if step.OnError != nil && handleOnError(ctx, step.OnError, err, &out, step, wfCtx, exec) {
				wfCtx.SetStepOutput(step.ID, out)
				last = out
				continue
			}
			return nil, err
		}
		wfCtx.SetStepOutput(step.ID, out)
		last = out
	}
	return last, nil
}

// handleOnError applies OnError policy to a step failure.
// Returns true if execution should continue (error was handled), false to propagate.
func handleOnError(ctx context.Context, policy *OnErrorDef, err error, out *interface{}, step StepDef, wfCtx *WorkflowContext, exec Executor) bool {
	if policy == nil {
		return false
	}

	// Retry with exponential backoff.
	if policy.Retry > 0 {
		backoff := 100 * time.Millisecond
		for attempt := 1; attempt <= policy.Retry; attempt++ {
			select {
			case <-ctx.Done():
				return false
			case <-time.After(backoff):
			}
			backoff *= 2

			exec.EmitEvent(Event{
				Type:      EventStepError,
				Data:      StepErrorEvent{StepID: step.ID, Attempt: attempt, Err: err},
				Timestamp: time.Now(),
			})

			// Re-run step (without emitting start/complete — those were emitted by runStep).
			var retryOut interface{}
			var retryErr error
			switch {
			case step.Tool != nil:
				resolved, _ := wfCtx.ResolveMap(step.Tool.With)
				retryOut, retryErr = exec.CallTool(ctx, step.Tool.Name, resolved)
			case step.LLM != nil:
				prompt, _ := wfCtx.ResolveString(step.LLM.Prompt)
				system, _ := wfCtx.ResolveString(step.LLM.System)
				retryOut, retryErr = exec.CallLLM(ctx, prompt, &LLMCallOptions{
					Model:        step.LLM.Model,
					MaxTokens:    step.LLM.MaxTokens,
					Temperature:  step.LLM.Temperature,
					SystemPrompt: system,
				})
			}
			if retryErr == nil {
				*out = retryOut
				return true
			}
			err = retryErr
		}
	}

	if policy.Fallback != nil {
		*out = policy.Fallback
		return true
	}

	if policy.Ignore {
		*out = nil
		return true
	}

	return false
}

// engineExecutor implements the Executor interface using the Engine's real providers.
type engineExecutor struct {
	engine *Engine
	events chan<- Event
}

func (ex *engineExecutor) CallTool(ctx context.Context, name string, input map[string]interface{}) (interface{}, error) {
	reg := ex.engine.cfg.FunctionRegistry
	if reg == nil {
		reg = registry.Global()
	}

	fn := reg.Get(name, ex.engine.cfg.FunctionConfig)
	if fn == nil {
		return nil, fmt.Errorf("%w: tool %q not found in registry", ErrStepFailed, name)
	}

	return fn.Execute(ctx, input)
}

func (ex *engineExecutor) CallLLM(ctx context.Context, prompt string, opts *LLMCallOptions) (string, error) {
	provider := ex.engine.cfg.LLMProvider
	if provider == nil {
		return "", fmt.Errorf("%w: no LLM provider configured", ErrStepFailed)
	}

	model := opts.Model
	if model == "" {
		model = ex.engine.cfg.DefaultModel
	}

	maxTokens := opts.MaxTokens
	if maxTokens == 0 {
		maxTokens = ex.engine.cfg.DefaultMaxTokens
	}

	temperature := opts.Temperature
	if temperature == 0 {
		temperature = ex.engine.cfg.DefaultTemperature
	}

	completionOpts := &llm.CompletionOptions{
		MaxTokens:    maxTokens,
		Temperature:  temperature,
		SystemPrompt: opts.SystemPrompt,
	}

	ch := provider.Ainvoke(ctx, prompt, completionOpts)

	var sb strings.Builder
	for chunk := range ch {
		if chunk.Error != nil {
			return "", fmt.Errorf("%w: LLM error: %v", ErrStepFailed, chunk.Error)
		}
		sb.WriteString(chunk.Text)
	}

	return sb.String(), nil
}

func (ex *engineExecutor) Confirm(ctx context.Context, stepID, prompt string, ctxValue interface{}) (bool, error) {
	// Create a 1-buffered channel so the consumer can send without blocking.
	respCh := make(chan bool, 1)

	ex.EmitEvent(Event{
		Type: EventConfirm,
		Data: ConfirmEvent{
			StepID:     stepID,
			Prompt:     prompt,
			Context:    ctxValue,
			ResponseCh: respCh,
		},
		Timestamp: time.Now(),
	})

	// Block until the consumer responds or context is cancelled.
	select {
	case approved := <-respCh:
		return approved, nil
	case <-ctx.Done():
		return false, ctx.Err()
	}
}

func (ex *engineExecutor) EmitEvent(ev Event) {
	ex.events <- ev
}
