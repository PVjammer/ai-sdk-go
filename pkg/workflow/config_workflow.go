package workflow

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/pvjammer/ai-sdk-go/pkg/function"
	"gopkg.in/yaml.v3"
)

// ConfigWorkflow is a Workflow loaded from a YAML WorkflowDef.
//
// It delegates execution to the Engine and derives its metadata from the def.
//
// Usage:
//
//	wf, err := workflow.NewConfigWorkflowFromFile("jira-sync.yaml", engine, cfg)
//	if err != nil { ... }
//	out, err := wf.Execute(ctx, input)
type ConfigWorkflow struct {
	WorkflowBase
	def    *WorkflowDef
	engine *Engine
}

// NewConfigWorkflow creates a ConfigWorkflow from an already-parsed WorkflowDef.
func NewConfigWorkflow(def *WorkflowDef, engine *Engine, cfg function.Config) *ConfigWorkflow {
	return &ConfigWorkflow{
		WorkflowBase: NewWorkflowBase(cfg),
		def:          def,
		engine:       engine,
	}
}

// NewConfigWorkflowFromFile reads and parses a YAML workflow definition file.
func NewConfigWorkflowFromFile(path string, engine *Engine, cfg function.Config) (*ConfigWorkflow, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("cannot read workflow file %q: %w", path, err)
	}
	return NewConfigWorkflowFromBytes(data, engine, cfg)
}

// NewConfigWorkflowFromBytes parses a YAML workflow definition from bytes.
//
// Validates that:
//   - Each StepDef has exactly one type field set.
//   - All step IDs are non-empty and unique (including sub-steps).
//
// Returns a multi-error slice describing all validation failures.
func NewConfigWorkflowFromBytes(data []byte, engine *Engine, cfg function.Config) (*ConfigWorkflow, error) {
	var def WorkflowDef
	if err := yaml.Unmarshal(data, &def); err != nil {
		return nil, fmt.Errorf("%w: YAML parse error: %v", ErrInvalidDef, err)
	}

	if errs := validateDef(&def); len(errs) > 0 {
		msgs := make([]string, len(errs))
		for i, e := range errs {
			msgs[i] = e.Error()
		}
		return nil, fmt.Errorf("%w: %s", ErrInvalidDef, strings.Join(msgs, "; "))
	}

	return NewConfigWorkflow(&def, engine, cfg), nil
}

// Execute runs the workflow synchronously, delegating to engine.Execute().
func (w *ConfigWorkflow) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	return w.engine.Execute(ctx, w.def, input)
}

// Metadata returns workflow metadata derived from the WorkflowDef.
func (w *ConfigWorkflow) Metadata() *WorkflowMetadata {
	return &WorkflowMetadata{
		Name:        w.def.Name,
		Description: w.def.Description,
		Version:     w.def.Version,
	}
}

// validateDef validates a WorkflowDef and returns all errors found.
func validateDef(def *WorkflowDef) []error {
	seen := make(map[string]bool)
	return validateSteps(def.Steps, seen, "")
}

// validateSteps recursively validates a slice of step definitions.
func validateSteps(steps []StepDef, seen map[string]bool, prefix string) []error {
	var errs []error

	for i, step := range steps {
		loc := fmt.Sprintf("%sstep[%d]", prefix, i)

		// Validate ID.
		if step.ID == "" {
			errs = append(errs, fmt.Errorf("%s: step ID must not be empty", loc))
		} else if seen[step.ID] {
			errs = append(errs, fmt.Errorf("%s: duplicate step ID %q", loc, step.ID))
		} else {
			seen[step.ID] = true
			loc = fmt.Sprintf("%sstep[%s]", prefix, step.ID)
		}

		// Validate type count.
		count := step.typeCount()
		if count == 0 {
			errs = append(errs, fmt.Errorf("%s: has no type (tool/llm/confirm/foreach/condition)", loc))
		} else if count > 1 {
			errs = append(errs, fmt.Errorf("%s: has multiple types (only one allowed)", loc))
		}

		// Recursively validate sub-steps.
		if step.Foreach != nil {
			subErrs := validateSteps(step.Foreach.Steps, seen, loc+".foreach.")
			errs = append(errs, subErrs...)
		}
		if step.Condition != nil {
			subErrs := validateSteps(step.Condition.Then, seen, loc+".then.")
			errs = append(errs, subErrs...)
			subErrs = validateSteps(step.Condition.Else, seen, loc+".else.")
			errs = append(errs, subErrs...)
		}
	}

	return errs
}
