package workflow

import (
	"context"

	"github.com/pvjammer/ai-sdk-go/pkg/function"
)

// Workflow is the interface implemented by all workflow types.
//
// A Workflow is a named, repeatable pipeline of explicitly ordered steps.
// It is a first-class SDK primitive, parallel to AIFunction and Agent.
type Workflow interface {
	// Execute runs the workflow synchronously and returns the final output.
	//
	// This is the Go-API invocation path. The Engine.Run() method wraps this
	// with event streaming for real-time feedback.
	Execute(ctx context.Context, input interface{}) (interface{}, error)

	// Metadata returns the workflow's descriptive metadata.
	Metadata() *WorkflowMetadata

	// Config returns the workflow's configuration.
	Config() function.Config
}

// WorkflowMetadata describes a workflow's identity and purpose.
type WorkflowMetadata struct {
	Name        string
	Description string
	Version     string
	Category    string
	Tags        []string
}

// WorkflowBase embeds function.Base to reuse Config() — same pattern as AIFunction.
//
// Concrete workflows embed WorkflowBase to get the Config() method for free.
//
// Example:
//
//	type MyWorkflow struct {
//	    workflow.WorkflowBase
//	    // ...
//	}
//
//	func New(cfg function.Config) *MyWorkflow {
//	    return &MyWorkflow{WorkflowBase: workflow.NewWorkflowBase(cfg)}
//	}
type WorkflowBase struct {
	function.Base
}

// NewWorkflowBase creates a new WorkflowBase with the given configuration.
func NewWorkflowBase(cfg function.Config) WorkflowBase {
	return WorkflowBase{Base: function.NewBase(cfg)}
}
