// Package workflow provides the Workflow abstraction for ai-sdk-go.
//
// A Workflow is a named, repeatable pipeline where steps are explicitly ordered.
// The LLM is just one step type among several (tool, LLM, confirm, foreach, condition).
// This is a first-class SDK primitive, parallel to AIFunction and Agent in the hierarchy.
package workflow

import "errors"

// Sentinel errors for the workflow package.
var (
	// ErrStepFailed indicates a workflow step failed and could not be recovered.
	ErrStepFailed = errors.New("workflow step failed")

	// ErrConfirmDenied indicates a confirmation gate was denied by the user.
	ErrConfirmDenied = errors.New("confirmation gate denied")

	// ErrInvalidDef indicates a workflow definition is invalid (e.g., missing step type).
	ErrInvalidDef = errors.New("invalid workflow definition")

	// ErrStepNotFound indicates a referenced step ID is not in the workflow context.
	ErrStepNotFound = errors.New("step not found in workflow context")

	// ErrTemplateResolution indicates a template expression could not be resolved.
	ErrTemplateResolution = errors.New("template resolution error")
)
