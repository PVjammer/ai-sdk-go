// Package tools provides tool management and execution for agents.
package tools

import (
	"context"
	"fmt"

	"github.com/pvjammer/ai-sdk-go/pkg/planner"
	"github.com/pvjammer/ai-sdk-go/pkg/function"
)

// ToolManager manages tool registration and execution.
//
// It wraps the AIFunction interface and provides
// a bridge between the agent system and the function library.
type ToolManager struct {
	tools map[string]function.AIFunction
}

// NewToolManager creates a new tool manager.
func NewToolManager() *ToolManager {
	return &ToolManager{
		tools: make(map[string]function.AIFunction),
	}
}

// Register adds a tool to the manager.
//
// The tool's name is derived from its metadata.
func (tm *ToolManager) Register(fn function.AIFunction) error {
	name := fn.Metadata().GetName("tool")
	if name == "" {
		return fmt.Errorf("tool has no name")
	}

	if _, exists := tm.tools[name]; exists {
		return fmt.Errorf("tool %s already registered", name)
	}

	tm.tools[name] = fn
	return nil
}

// Execute invokes a tool by name with the given input.
func (tm *ToolManager) Execute(ctx context.Context, name string, input interface{}) (interface{}, error) {
	fn, exists := tm.tools[name]
	if !exists {
		return nil, fmt.Errorf("tool not found: %s", name)
	}

	// Execute the tool
	// Note: AIFunction implementations can be sync or async
	// The Execute method handles both
	return fn.Execute(ctx, input)
}

// GetToolDefinitions returns tool definitions for the planner.
//
// This converts AIFunction metadata to the format expected by planners.
func (tm *ToolManager) GetToolDefinitions() []planner.ToolDefinition {
	defs := make([]planner.ToolDefinition, 0, len(tm.tools))

	for name, fn := range tm.tools {
		meta := fn.Metadata()
		tool := function.AsTool(fn)

		// Extract description and parameters from the tool format
		desc := ""
		params := make(map[string]interface{})

		if description, ok := tool["description"].(string); ok {
			desc = description
		}

		if function, ok := tool["function"].(map[string]interface{}); ok {
			if parameters, ok := function["parameters"].(map[string]interface{}); ok {
				params = parameters
			}
		}

		defs = append(defs, planner.ToolDefinition{
			Name:        name,
			Description: desc,
			Parameters:  params,
		})

		// Fallback: use metadata if tool format didn't provide description
		if desc == "" {
			defs[len(defs)-1].Description = meta.Description
		}
	}

	return defs
}

// GetTool retrieves a registered tool by name.
func (tm *ToolManager) GetTool(name string) (function.AIFunction, bool) {
	fn, exists := tm.tools[name]
	return fn, exists
}

// Count returns the number of registered tools.
func (tm *ToolManager) Count() int {
	return len(tm.tools)
}

// Clear removes all registered tools.
func (tm *ToolManager) Clear() {
	tm.tools = make(map[string]function.AIFunction)
}
