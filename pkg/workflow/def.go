package workflow

// WorkflowDef is the top-level YAML-serializable definition of a workflow.
//
// Example YAML:
//
//	name: jira-sync
//	description: Match today's commits to open Jira tickets and update them
//	steps:
//	  - id: commits
//	    tool: { name: git_log_today, with: { since: today } }
//	  - id: matches
//	    llm:
//	      system: "You match commits to tickets."
//	      prompt: "Commits: {{steps.commits}}\nTickets: {{steps.tickets}}"
type WorkflowDef struct {
	Name        string    `yaml:"name"`
	Description string    `yaml:"description"`
	Version     string    `yaml:"version,omitempty"`
	Steps       []StepDef `yaml:"steps"`
}

// StepDef defines a single workflow step.
//
// Exactly one of Tool, LLM, Confirm, Foreach, or Condition must be non-nil.
// Multiple or zero type fields results in ErrInvalidDef at validation time.
type StepDef struct {
	ID          string            `yaml:"id"`
	Description string            `yaml:"description,omitempty"`
	Tool        *ToolStepDef      `yaml:"tool,omitempty"`
	LLM         *LLMStepDef       `yaml:"llm,omitempty"`
	Confirm     *ConfirmStepDef   `yaml:"confirm,omitempty"`
	Foreach     *ForeachStepDef   `yaml:"foreach,omitempty"`
	Condition   *ConditionStepDef `yaml:"condition,omitempty"`
	OnError     *OnErrorDef       `yaml:"on_error,omitempty"`
}

// stepType returns the type name of the step for use in events and error messages.
func (s *StepDef) stepType() string {
	if s.Tool != nil {
		return "tool"
	}
	if s.LLM != nil {
		return "llm"
	}
	if s.Confirm != nil {
		return "confirm"
	}
	if s.Foreach != nil {
		return "foreach"
	}
	if s.Condition != nil {
		return "condition"
	}
	return "unknown"
}

// typeCount returns the number of type fields set on this step (for validation).
func (s *StepDef) typeCount() int {
	count := 0
	if s.Tool != nil {
		count++
	}
	if s.LLM != nil {
		count++
	}
	if s.Confirm != nil {
		count++
	}
	if s.Foreach != nil {
		count++
	}
	if s.Condition != nil {
		count++
	}
	return count
}

// ToolStepDef configures a step that calls a registered AIFunction tool.
type ToolStepDef struct {
	// Name is the registered name of the tool to call.
	Name string `yaml:"name"`

	// With is the input map passed to the tool. Values may contain template
	// expressions like {{steps.prev_step}} or {{input.field}}.
	With map[string]interface{} `yaml:"with,omitempty"`
}

// LLMStepDef configures a step that calls an LLM.
type LLMStepDef struct {
	// Prompt is the user-facing prompt. Supports template expressions.
	Prompt string `yaml:"prompt"`

	// System is the system prompt. Supports template expressions.
	System string `yaml:"system,omitempty"`

	// Model overrides the engine's default model for this step.
	Model string `yaml:"model,omitempty"`

	// MaxTokens overrides the engine's default max_tokens for this step.
	MaxTokens int `yaml:"max_tokens,omitempty"`

	// Temperature overrides the engine's default temperature for this step.
	Temperature float64 `yaml:"temperature,omitempty"`
}

// ConfirmStepDef configures a step that requires user confirmation before continuing.
//
// The workflow emits an EventConfirm and blocks until the consumer responds.
// If denied, the workflow halts with ErrConfirmDenied.
type ConfirmStepDef struct {
	// Prompt is shown to the user. Supports template expressions.
	Prompt string `yaml:"prompt"`
}

// ForeachStepDef configures a step that iterates over a list.
type ForeachStepDef struct {
	// Over is a template expression that resolves to a []interface{}.
	Over string `yaml:"over"`

	// As is the variable name bound for each item (default: "item").
	// Accessible in sub-steps as {{item}} or {{<As>}}.
	As string `yaml:"as,omitempty"`

	// Steps are executed once per item in the list.
	Steps []StepDef `yaml:"steps"`

	// Parallel executes all iterations concurrently when true.
	// First error cancels remaining iterations.
	Parallel bool `yaml:"parallel,omitempty"`
}

// ConditionStepDef configures a step with conditional branching.
type ConditionStepDef struct {
	// If is a template expression that must resolve to a bool (or truthy string).
	If string `yaml:"if"`

	// Then steps run when If evaluates to true.
	Then []StepDef `yaml:"then,omitempty"`

	// Else steps run when If evaluates to false.
	Else []StepDef `yaml:"else,omitempty"`
}

// OnErrorDef configures error handling for a step.
type OnErrorDef struct {
	// Retry is the number of additional attempts to make after the first failure.
	// Retries use exponential backoff starting at 100ms.
	Retry int `yaml:"retry,omitempty"`

	// Ignore causes the step to store nil output and continue on error.
	Ignore bool `yaml:"ignore,omitempty"`

	// Fallback is a static value stored as the step output on error (instead of failing).
	// Mutually exclusive with Ignore.
	Fallback interface{} `yaml:"fallback,omitempty"`
}
