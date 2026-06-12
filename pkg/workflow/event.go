package workflow

import "time"

// EventType identifies the kind of workflow event being emitted.
type EventType string

const (
	// EventStepStart indicates a workflow step has started executing.
	EventStepStart EventType = "step_start"

	// EventStepComplete indicates a workflow step has finished successfully.
	EventStepComplete EventType = "step_complete"

	// EventStepError indicates a workflow step encountered an error.
	EventStepError EventType = "step_error"

	// EventConfirm indicates the workflow is waiting for user confirmation.
	// The consumer must send true or false on ConfirmEvent.ResponseCh to unblock.
	EventConfirm EventType = "confirm"

	// EventResponse contains the final workflow output.
	EventResponse EventType = "response"

	// EventError indicates a fatal workflow error that stops execution.
	EventError EventType = "error"

	// EventComplete indicates the workflow has finished (after EventResponse).
	EventComplete EventType = "complete"
)

// Event represents something that happened during workflow execution.
//
// Events are streamed via channels to provide real-time feedback.
type Event struct {
	Type      EventType   // What kind of event this is
	Data      interface{} // Type-specific data (cast based on Type)
	Timestamp time.Time   // When the event occurred
}

// StepStartEvent is emitted when a step begins execution.
type StepStartEvent struct {
	StepID   string      // ID of the step
	StepType string      // One of: "tool", "llm", "confirm", "foreach", "condition"
	Input    interface{} // Resolved input passed to the step
}

// StepCompleteEvent is emitted when a step finishes successfully.
type StepCompleteEvent struct {
	StepID string      // ID of the step
	Output interface{} // Output produced by the step
}

// StepErrorEvent is emitted when a step fails.
type StepErrorEvent struct {
	StepID  string // ID of the step
	Attempt int    // Which attempt this was (1-based; >1 means a retry occurred)
	Err     error  // The error that occurred
}

// ConfirmEvent is emitted when a confirm step requires user input.
//
// The engine goroutine blocks until the consumer sends on ResponseCh
// or the context is cancelled. The event channel stays unclogged because
// no new events are written while the engine is waiting.
//
// Usage:
//
//	case ev := <-events:
//	    if ev.Type == workflow.EventConfirm {
//	        ce := ev.Data.(workflow.ConfirmEvent)
//	        // show ce.Prompt to user, then:
//	        ce.ResponseCh <- true // approve
//	    }
type ConfirmEvent struct {
	StepID     string      // ID of the confirm step
	Prompt     string      // Resolved prompt shown to the user
	Context    interface{} // Workflow context at the time of the gate (for display)
	ResponseCh chan<- bool  // 1-buffered; consumer sends true/false to unblock engine
}

// ResponseEvent contains the final output of the workflow.
type ResponseEvent struct {
	Output interface{} // The workflow's final output value
}

// ErrorEvent contains information about a fatal workflow error.
type ErrorEvent struct {
	Err     error  // The error that occurred
	Message string // Human-readable error message
}
