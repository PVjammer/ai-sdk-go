package function

import (
	"fmt"
)

// Error wrapping helpers for better error context

// WrapExecutionError wraps an error that occurred during function execution
// with the function name and operation context.
//
// Example:
//
//	if err := validateInput(input); err != nil {
//	    return nil, WrapExecutionError(fn, "input validation", err)
//	}
//	// Error: "function greet: input validation: name is required"
func WrapExecutionError(fn AIFunction, operation string, err error) error {
	if err == nil {
		return nil
	}

	meta := fn.Metadata()
	functionName := meta.GetName("")

	return fmt.Errorf("function %s: %s: %w", functionName, operation, err)
}

// WrapSchemaError wraps an error that occurred during schema generation
// with function context.
//
// Example:
//
//	schema, err := GenerateSchema(inputType)
//	if err != nil {
//	    return WrapSchemaError(fn, err)
//	}
//	// Error: "function greet: schema generation: unsupported type: chan"
func WrapSchemaError(fn AIFunction, err error) error {
	if err == nil {
		return nil
	}

	meta := fn.Metadata()
	functionName := meta.GetName("")

	return fmt.Errorf("function %s: schema generation: %w", functionName, err)
}

// WrapValidationError wraps a validation error with function context.
//
// Example:
//
//	if err := Validate(input); err != nil {
//	    return nil, WrapValidationError(fn, err)
//	}
//	// Error: "function greet: validation failed: name is required"
func WrapValidationError(fn AIFunction, err error) error {
	if err == nil {
		return nil
	}

	meta := fn.Metadata()
	functionName := meta.GetName("")

	return fmt.Errorf("function %s: validation failed: %w", functionName, err)
}

// WrapConfigError wraps a configuration error with function context.
//
// Example:
//
//	apiKey := fn.Config().GetString("api_key")
//	if apiKey == "" {
//	    return nil, WrapConfigError(fn, "api_key", fmt.Errorf("required configuration missing"))
//	}
//	// Error: "function web_search: config 'api_key': required configuration missing"
func WrapConfigError(fn AIFunction, configKey string, err error) error {
	if err == nil {
		return nil
	}

	meta := fn.Metadata()
	functionName := meta.GetName("")

	return fmt.Errorf("function %s: config '%s': %w", functionName, configKey, err)
}

// NewExecutionError creates a new execution error without wrapping.
//
// Use this when you want to create an error message from scratch rather than
// wrapping an existing error.
//
// Example:
//
//	if len(results) == 0 {
//	    return nil, NewExecutionError(fn, "search", "no results found")
//	}
//	// Error: "function web_search: search: no results found"
func NewExecutionError(fn AIFunction, operation string, message string) error {
	meta := fn.Metadata()
	functionName := meta.GetName("")

	return fmt.Errorf("function %s: %s: %s", functionName, operation, message)
}

// NewValidationError creates a new validation error without wrapping.
//
// Example:
//
//	if input.Count < 1 {
//	    return nil, NewValidationError(fn, "count must be at least 1")
//	}
//	// Error: "function process: validation failed: count must be at least 1"
func NewValidationError(fn AIFunction, message string) error {
	meta := fn.Metadata()
	functionName := meta.GetName("")

	return fmt.Errorf("function %s: validation failed: %s", functionName, message)
}

// NewConfigError creates a new configuration error without wrapping.
//
// Example:
//
//	if !fn.Config().IsSet("api_key") {
//	    return nil, NewConfigError(fn, "api_key", "required configuration not set")
//	}
//	// Error: "function web_search: config 'api_key': required configuration not set"
func NewConfigError(fn AIFunction, configKey string, message string) error {
	meta := fn.Metadata()
	functionName := meta.GetName("")

	return fmt.Errorf("function %s: config '%s': %s", functionName, configKey, message)
}

// ErrorContext provides structured error context for complex error scenarios.
//
// Example:
//
//	ctx := &ErrorContext{
//	    Function:  fn,
//	    Operation: "api_request",
//	    Details: map[string]interface{}{
//	        "url":         "https://api.example.com",
//	        "status_code": 500,
//	        "attempt":     3,
//	    },
//	}
//	return nil, ctx.Wrap(originalError)
type ErrorContext struct {
	Function  AIFunction
	Operation string
	Details   map[string]interface{}
}

// Wrap wraps an error with structured context.
//
// The error message includes the function name, operation, and details.
//
// Example output:
//
//	"function web_search: api_request: HTTP 500 from https://api.example.com (attempt 3): connection timeout"
func (ec *ErrorContext) Wrap(err error) error {
	if err == nil {
		return nil
	}

	meta := ec.Function.Metadata()
	functionName := meta.GetName("")

	// Build details string
	detailsStr := ""
	if len(ec.Details) > 0 {
		detailsStr = " ("
		first := true
		for key, value := range ec.Details {
			if !first {
				detailsStr += ", "
			}
			detailsStr += fmt.Sprintf("%s: %v", key, value)
			first = false
		}
		detailsStr += ")"
	}

	return fmt.Errorf("function %s: %s%s: %w", functionName, ec.Operation, detailsStr, err)
}

// Error creates a new error with structured context (without wrapping).
func (ec *ErrorContext) Error(message string) error {
	meta := ec.Function.Metadata()
	functionName := meta.GetName("")

	// Build details string
	detailsStr := ""
	if len(ec.Details) > 0 {
		detailsStr = " ("
		first := true
		for key, value := range ec.Details {
			if !first {
				detailsStr += ", "
			}
			detailsStr += fmt.Sprintf("%s: %v", key, value)
			first = false
		}
		detailsStr += ")"
	}

	return fmt.Errorf("function %s: %s%s: %s", functionName, ec.Operation, detailsStr, message)
}
