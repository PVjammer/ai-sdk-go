package workflow

import (
	"fmt"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"sync"
)

// templatePattern matches {{expression}} placeholders.
var templatePattern = regexp.MustCompile(`\{\{([^}]+)\}\}`)

// WorkflowContext stores step outputs and resolves template expressions.
//
// Template expressions support:
//   - {{input}}             — the entire workflow input
//   - {{input.field}}       — a field on the workflow input
//   - {{steps.id}}          — the output of step "id"
//   - {{steps.id.field}}    — a field on the output of step "id"
//   - {{item}}              — the current foreach item (only in foreach sub-contexts)
//
// A single template expression resolving the entire string preserves the
// underlying type (e.g., []interface{} for foreach, bool for conditions).
// Expressions embedded within a larger string are stringified.
type WorkflowContext struct {
	mu          sync.RWMutex
	input       interface{}
	stepOutputs map[string]interface{}
	itemValue   interface{} // set by WithItem for foreach
	itemKey     string      // the "as" name (default: "item")
	parent      *WorkflowContext
}

// NewWorkflowContext creates a root context with the given workflow input.
func NewWorkflowContext(input interface{}) *WorkflowContext {
	return &WorkflowContext{
		input:       input,
		stepOutputs: make(map[string]interface{}),
		itemKey:     "item",
	}
}

// SetStepOutput stores the output of a completed step.
func (wc *WorkflowContext) SetStepOutput(stepID string, output interface{}) {
	wc.mu.Lock()
	defer wc.mu.Unlock()
	wc.stepOutputs[stepID] = output
}

// GetStepOutput retrieves the output of a step by ID.
func (wc *WorkflowContext) GetStepOutput(stepID string) (interface{}, bool) {
	wc.mu.RLock()
	defer wc.mu.RUnlock()
	v, ok := wc.stepOutputs[stepID]
	return v, ok
}

// GetLastOutput returns the output of the last step in the given list.
// Returns nil if the list is empty or the last step has no output.
func (wc *WorkflowContext) GetLastOutput(steps []StepDef) interface{} {
	if len(steps) == 0 {
		return nil
	}
	last := steps[len(steps)-1]
	v, _ := wc.GetStepOutput(last.ID)
	return v
}

// WithItem creates a child context with a foreach-bound item value.
// The item is accessible as {{item}} or {{<as>}} in sub-step templates.
func (wc *WorkflowContext) WithItem(item interface{}, as string) *WorkflowContext {
	key := "item"
	if as != "" {
		key = as
	}
	child := &WorkflowContext{
		input:       wc.input,
		stepOutputs: make(map[string]interface{}),
		itemValue:   item,
		itemKey:     key,
		parent:      wc,
	}
	// Copy parent step outputs into child so sub-steps can reference prior steps.
	wc.mu.RLock()
	for k, v := range wc.stepOutputs {
		child.stepOutputs[k] = v
	}
	wc.mu.RUnlock()
	return child
}

// Resolve resolves a template expression and returns the typed value.
//
// If expr contains exactly one {{...}} and nothing else, the result is the
// typed value (preserving []interface{}, bool, etc.).
// If expr is a plain string with no {{...}}, it is returned as-is.
// If expr contains multiple or embedded {{...}}, results are stringified.
func (wc *WorkflowContext) Resolve(expr string) (interface{}, error) {
	matches := templatePattern.FindAllStringSubmatchIndex(expr, -1)
	if len(matches) == 0 {
		// Plain string — no template expressions.
		return expr, nil
	}

	// Single template covering the entire expression — return typed value.
	if len(matches) == 1 {
		m := matches[0]
		if m[0] == 0 && m[1] == len(expr) {
			inner := strings.TrimSpace(expr[m[2]:m[3]])
			return wc.resolvePath(inner)
		}
	}

	// Multiple or embedded templates — stringify each match.
	result := templatePattern.ReplaceAllStringFunc(expr, func(match string) string {
		inner := strings.TrimSpace(match[2 : len(match)-2])
		val, err := wc.resolvePath(inner)
		if err != nil {
			return match // leave unresolved placeholder on error
		}
		return fmt.Sprintf("%v", val)
	})
	return result, nil
}

// ResolveString resolves a template expression and returns a string.
func (wc *WorkflowContext) ResolveString(expr string) (string, error) {
	val, err := wc.Resolve(expr)
	if err != nil {
		return "", err
	}
	if s, ok := val.(string); ok {
		return s, nil
	}
	return fmt.Sprintf("%v", val), nil
}

// ResolveMap resolves all values in a map, returning a new map with resolved values.
func (wc *WorkflowContext) ResolveMap(m map[string]interface{}) (map[string]interface{}, error) {
	result := make(map[string]interface{}, len(m))
	for k, v := range m {
		switch val := v.(type) {
		case string:
			resolved, err := wc.Resolve(val)
			if err != nil {
				return nil, fmt.Errorf("%w: key %q: %v", ErrTemplateResolution, k, err)
			}
			result[k] = resolved
		default:
			result[k] = v
		}
	}
	return result, nil
}

// resolvePath walks a dot-separated path (e.g., "steps.myStep.field") to a value.
func (wc *WorkflowContext) resolvePath(path string) (interface{}, error) {
	parts := strings.SplitN(path, ".", 2)
	root := parts[0]
	rest := ""
	if len(parts) > 1 {
		rest = parts[1]
	}

	var rootVal interface{}

	switch root {
	case "input":
		rootVal = wc.input
	case "steps":
		if rest == "" {
			return nil, fmt.Errorf("%w: 'steps' requires a step ID (e.g. steps.my_step)", ErrTemplateResolution)
		}
		stepParts := strings.SplitN(rest, ".", 2)
		stepID := stepParts[0]
		stepRest := ""
		if len(stepParts) > 1 {
			stepRest = stepParts[1]
		}
		stepOut, ok := wc.lookupStep(stepID)
		if !ok {
			return nil, fmt.Errorf("%w: step %q not found", ErrStepNotFound, stepID)
		}
		if stepRest == "" {
			return stepOut, nil
		}
		return walkPath(stepOut, stepRest)
	default:
		// Check if this matches the item key (e.g., "item" or the foreach "as" name).
		if root == wc.itemKey || root == "item" {
			itemVal := wc.lookupItem()
			if itemVal != nil || wc.hasItem() {
				if rest == "" {
					return itemVal, nil
				}
				return walkPath(itemVal, rest)
			}
		}
		return nil, fmt.Errorf("%w: unknown root %q in expression %q", ErrTemplateResolution, root, path)
	}

	if rest == "" {
		return rootVal, nil
	}
	return walkPath(rootVal, rest)
}

// lookupStep searches this context and its parent chain for a step output.
func (wc *WorkflowContext) lookupStep(stepID string) (interface{}, bool) {
	wc.mu.RLock()
	v, ok := wc.stepOutputs[stepID]
	wc.mu.RUnlock()
	if ok {
		return v, true
	}
	if wc.parent != nil {
		return wc.parent.lookupStep(stepID)
	}
	return nil, false
}

// lookupItem returns the item value from this context or its parent chain.
func (wc *WorkflowContext) lookupItem() interface{} {
	if wc.hasItem() {
		return wc.itemValue
	}
	if wc.parent != nil {
		return wc.parent.lookupItem()
	}
	return nil
}

// hasItem reports whether this context (not its parents) has an item bound.
func (wc *WorkflowContext) hasItem() bool {
	return wc.itemValue != nil || wc.parent == nil && wc.itemKey != "item"
}

// walkPath navigates a dot-separated field path on a value using maps and reflection.
func walkPath(val interface{}, path string) (interface{}, error) {
	parts := strings.SplitN(path, ".", 2)
	key := parts[0]
	rest := ""
	if len(parts) > 1 {
		rest = parts[1]
	}

	// Try map[string]interface{} first
	if m, ok := val.(map[string]interface{}); ok {
		v, exists := m[key]
		if !exists {
			return nil, fmt.Errorf("%w: key %q not found in map", ErrTemplateResolution, key)
		}
		if rest == "" {
			return v, nil
		}
		return walkPath(v, rest)
	}

	// Try numeric index for slices
	if idx, err := strconv.Atoi(key); err == nil {
		rv := reflect.ValueOf(val)
		if rv.Kind() == reflect.Slice {
			if idx < 0 || idx >= rv.Len() {
				return nil, fmt.Errorf("%w: slice index %d out of range (len=%d)", ErrTemplateResolution, idx, rv.Len())
			}
			elem := rv.Index(idx).Interface()
			if rest == "" {
				return elem, nil
			}
			return walkPath(elem, rest)
		}
	}

	// Try struct field via reflection
	rv := reflect.ValueOf(val)
	if rv.Kind() == reflect.Ptr {
		rv = rv.Elem()
	}
	if rv.Kind() == reflect.Struct {
		// Try exact field name, then case-insensitive
		f := rv.FieldByName(key)
		if !f.IsValid() {
			// Case-insensitive fallback
			rt := rv.Type()
			for i := 0; i < rt.NumField(); i++ {
				if strings.EqualFold(rt.Field(i).Name, key) {
					f = rv.Field(i)
					break
				}
			}
		}
		if f.IsValid() {
			elem := f.Interface()
			if rest == "" {
				return elem, nil
			}
			return walkPath(elem, rest)
		}
	}

	return nil, fmt.Errorf("%w: cannot access field %q on %T", ErrTemplateResolution, key, val)
}

// coerceBool converts a value to bool for condition evaluation.
func coerceBool(val interface{}) bool {
	if val == nil {
		return false
	}
	switch v := val.(type) {
	case bool:
		return v
	case string:
		lower := strings.ToLower(strings.TrimSpace(v))
		return lower == "true" || lower == "yes" || lower == "1"
	case int:
		return v != 0
	case int64:
		return v != 0
	case float64:
		return v != 0
	default:
		// Non-nil, non-zero value
		rv := reflect.ValueOf(val)
		switch rv.Kind() {
		case reflect.Slice, reflect.Map:
			return rv.Len() > 0
		}
		return true
	}
}
