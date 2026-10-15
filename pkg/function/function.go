// Package function provides the core abstractions for AI functions.
//
// This package defines the AIFunction interface and supporting types that enable
// functions to work in three modes:
//  1. Direct Go API - Native function invocation
//  2. LLM Tool Format - OpenAI/Anthropic compatible tool definitions
//  3. CLI - Auto-generated Cobra commands
package function

import (
	"context"
	"sync"
)

// AIFunction defines the interface that all functions must implement.
//
// Functions provide a unified abstraction for operations that can be invoked
// as Go functions, LLM tools, or CLI commands. Each function defines its own
// input schema using struct tags and implements the core execution logic.
//
// Context Handling:
//
// Functions SHOULD respect context cancellation by checking ctx.Done() during
// long-running operations. This allows callers to cancel or timeout operations.
//
// Example with context handling:
//
//	type MyFunction struct {
//	    Base
//	}
//
//	func (f *MyFunction) Execute(ctx context.Context, input interface{}) (interface{}, error) {
//	    in := input.(*MyInput)
//
//	    // Check for cancellation before expensive operations
//	    select {
//	    case <-ctx.Done():
//	        return nil, ctx.Err()
//	    default:
//	    }
//
//	    // Perform work...
//	    result := processData(in.Data)
//
//	    // Check again for long operations
//	    if err := ctx.Err(); err != nil {
//	        return nil, err
//	    }
//
//	    return result, nil
//	}
type AIFunction interface {
	// Execute runs the function with validated input.
	//
	// The input parameter should be a pointer to the function's Input struct.
	// The context parameter should be respected for cancellation and timeouts.
	//
	// Long-running implementations SHOULD periodically check ctx.Done() or ctx.Err()
	// to allow graceful cancellation. Short operations may skip context checks.
	//
	// Returns the result and any error that occurred during execution.
	// If the context is canceled, implementations should return ctx.Err().
	//
	// When used as a tool via agent.AgentBase, Execute may be called concurrently with itself
	// and with other tools' Execute calls: a model's tool-call batch runs in parallel by default
	// (see AgentBase.SetSequentialTools to opt a tool out). Implementations with shared mutable
	// state (an in-memory cache, a non-thread-safe client, a file handle) must guard it
	// themselves; the agent loop does not serialize calls for you unless you opt in.
	Execute(ctx context.Context, input interface{}) (interface{}, error)

	// InputSchema returns the JSON Schema for input validation.
	// This schema is used for validating inputs, generating CLI flags,
	// and creating LLM tool definitions.
	InputSchema() *Schema

	// Metadata returns the function's metadata including names, descriptions,
	// categories, and tags. The metadata system supports format-specific
	// overrides for tool and CLI contexts.
	Metadata() *Metadata

	// Config returns the function's configuration.
	// This allows functions to access user configuration from YAML files
	// and environment variables.
	Config() Config
}

// Base provides common functionality for AI functions.
//
// Functions should embed Base to get access to configuration and common helpers.
// Base implements the Config() method of the AIFunction interface.
//
// Example:
//
//	type MyFunction struct {
//	    Base
//	}
//
//	func New(cfg Config) *MyFunction {
//	    return &MyFunction{
//	        Base: Base{config: cfg},
//	    }
//	}
type Base struct {
	config Config
}

// NewBase creates a new Base with the given configuration.
func NewBase(cfg Config) Base {
	return Base{config: cfg}
}

// Config returns the function's configuration.
func (b *Base) Config() Config {
	return b.config
}

// Result represents the result of an async function invocation.
//
// Used by AsyncInvoke to communicate results through a channel.
type Result struct {
	Output interface{}
	Error  error
}

// Invoke executes the function synchronously.
//
// This is a convenience wrapper that calls Execute with the provided context
// and input. It's equivalent to calling fn.Execute(ctx, input) directly.
//
// Example:
//
//	result, err := function.Invoke(ctx, fn, &Input{Text: "hello"})
func Invoke(ctx context.Context, fn AIFunction, input interface{}) (interface{}, error) {
	return fn.Execute(ctx, input)
}

// AsyncInvoke executes the function asynchronously and returns a result channel.
//
// The function runs in a goroutine and sends its result to the returned channel.
// The channel is buffered with capacity 1 and closed after the result is sent.
//
// Example:
//
//	resultCh := function.AsyncInvoke(ctx, fn, &Input{Text: "hello"})
//	result := <-resultCh
//	if result.Error != nil {
//	    // Handle error
//	}
//	// Use result.Output
func AsyncInvoke(ctx context.Context, fn AIFunction, input interface{}) <-chan Result {
	ch := make(chan Result, 1)
	go func() {
		defer close(ch)
		output, err := fn.Execute(ctx, input)
		ch <- Result{Output: output, Error: err}
	}()
	return ch
}

// InvokeAll executes multiple functions concurrently.
//
// Each function is executed in its own goroutine, and results are collected
// in the order they complete (not the order they were provided).
// The function waits for all invocations to complete before returning.
//
// Example:
//
//	results := function.InvokeAll(ctx, []FunctionCall{
//	    {Function: fn1, Input: input1},
//	    {Function: fn2, Input: input2},
//	})
//	for result := range results {
//	    if result.Error != nil {
//	        // Handle error
//	    }
//	    // Use result.Output
//	}
type FunctionCall struct {
	Function AIFunction
	Input    interface{}
}

// InvokeAll executes multiple function calls concurrently.
func InvokeAll(ctx context.Context, calls []FunctionCall) <-chan Result {
	results := make(chan Result, len(calls))
	var wg sync.WaitGroup

	for _, call := range calls {
		wg.Add(1)
		go func(c FunctionCall) {
			defer wg.Done()
			output, err := c.Function.Execute(ctx, c.Input)
			results <- Result{Output: output, Error: err}
		}(call)
	}

	// Close channel after all goroutines complete
	go func() {
		wg.Wait()
		close(results)
	}()

	return results
}
