// Package planner provides strategies for agent decision-making.
package planner

import (
	"context"

	"github.com/pvjammer/ai-sdk-go/pkg/agentcontext"
)

// Planner decides what action an agent should take next.
//
// Different implementations can use different strategies:
//   - SimplePlanner: Prompt-based tool calling (works with any model)
//   - ReActPlanner: ReAct pattern (Reasoning + Acting)
//   - NativePlanner: Use model's native function calling (future)
type Planner interface {
	// Plan analyzes the conversation and decides the next action.
	//
	// Returns a Plan describing what to do (respond or use tools).
	Plan(ctx context.Context, messages []agentcontext.Message, tools []ToolDefinition) (*Plan, error)
}

// Plan describes what the agent should do next.
type Plan struct {
	// Action specifies whether to use tools or respond to the user
	Action ActionType

	// ToolCalls contains tools to execute (if Action == ActionUseTool)
	ToolCalls []agentcontext.ToolCall

	// Response contains text to send to user (if Action == ActionRespond)
	Response string

	// Reasoning contains the agent's internal thinking (may be empty)
	Reasoning string
}

// ActionType specifies what kind of action the agent should take.
type ActionType string

const (
	// ActionUseTool means the agent wants to call one or more tools
	ActionUseTool ActionType = "use_tool"

	// ActionRespond means the agent is ready to respond to the user
	ActionRespond ActionType = "respond"
)

// ToolDefinition describes a tool available to the agent.
//
// This is provided to the planner so it can format tool descriptions
// for the LLM in whatever format the planner prefers.
type ToolDefinition struct {
	Name        string                 // Tool name
	Description string                 // What the tool does
	Parameters  map[string]interface{} // JSON schema for parameters
}
