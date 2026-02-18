package function

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func setupTestFunction() *MockFunction {
	fn := NewMockFunction()
	fn.metadata = NewMetadata("TestFunction").WithName("test_func")
	return fn
}

// TestWrapExecutionError tests wrapping execution errors
func TestWrapExecutionError(t *testing.T) {
	fn := setupTestFunction()
	originalErr := errors.New("something went wrong")

	wrappedErr := WrapExecutionError(fn, "data processing", originalErr)

	require.Error(t, wrappedErr)
	assert.Contains(t, wrappedErr.Error(), "function test_func")
	assert.Contains(t, wrappedErr.Error(), "data processing")
	assert.Contains(t, wrappedErr.Error(), "something went wrong")

	// Should preserve original error via errors.Is
	assert.True(t, errors.Is(wrappedErr, originalErr))
}

// TestWrapExecutionError_NilError tests handling of nil errors
func TestWrapExecutionError_NilError(t *testing.T) {
	fn := setupTestFunction()

	wrappedErr := WrapExecutionError(fn, "operation", nil)
	assert.Nil(t, wrappedErr)
}

// TestWrapSchemaError tests wrapping schema generation errors
func TestWrapSchemaError(t *testing.T) {
	fn := setupTestFunction()
	originalErr := errors.New("unsupported type: chan")

	wrappedErr := WrapSchemaError(fn, originalErr)

	require.Error(t, wrappedErr)
	assert.Contains(t, wrappedErr.Error(), "function test_func")
	assert.Contains(t, wrappedErr.Error(), "schema generation")
	assert.Contains(t, wrappedErr.Error(), "unsupported type: chan")
	assert.True(t, errors.Is(wrappedErr, originalErr))
}

// TestWrapValidationError tests wrapping validation errors
func TestWrapValidationError(t *testing.T) {
	fn := setupTestFunction()
	originalErr := errors.New("name is required")

	wrappedErr := WrapValidationError(fn, originalErr)

	require.Error(t, wrappedErr)
	assert.Contains(t, wrappedErr.Error(), "function test_func")
	assert.Contains(t, wrappedErr.Error(), "validation failed")
	assert.Contains(t, wrappedErr.Error(), "name is required")
	assert.True(t, errors.Is(wrappedErr, originalErr))
}

// TestWrapConfigError tests wrapping configuration errors
func TestWrapConfigError(t *testing.T) {
	fn := setupTestFunction()
	originalErr := errors.New("required setting not provided")

	wrappedErr := WrapConfigError(fn, "api_key", originalErr)

	require.Error(t, wrappedErr)
	assert.Contains(t, wrappedErr.Error(), "function test_func")
	assert.Contains(t, wrappedErr.Error(), "config 'api_key'")
	assert.Contains(t, wrappedErr.Error(), "required setting not provided")
	assert.True(t, errors.Is(wrappedErr, originalErr))
}

// TestNewExecutionError tests creating new execution errors
func TestNewExecutionError(t *testing.T) {
	fn := setupTestFunction()

	err := NewExecutionError(fn, "search", "no results found")

	require.Error(t, err)
	assert.Contains(t, err.Error(), "function test_func")
	assert.Contains(t, err.Error(), "search")
	assert.Contains(t, err.Error(), "no results found")

	// Full message format check
	expected := "function test_func: search: no results found"
	assert.Equal(t, expected, err.Error())
}

// TestNewValidationError tests creating new validation errors
func TestNewValidationError(t *testing.T) {
	fn := setupTestFunction()

	err := NewValidationError(fn, "count must be positive")

	require.Error(t, err)
	assert.Contains(t, err.Error(), "function test_func")
	assert.Contains(t, err.Error(), "validation failed")
	assert.Contains(t, err.Error(), "count must be positive")

	expected := "function test_func: validation failed: count must be positive"
	assert.Equal(t, expected, err.Error())
}

// TestNewConfigError tests creating new configuration errors
func TestNewConfigError(t *testing.T) {
	fn := setupTestFunction()

	err := NewConfigError(fn, "timeout", "must be a positive duration")

	require.Error(t, err)
	assert.Contains(t, err.Error(), "function test_func")
	assert.Contains(t, err.Error(), "config 'timeout'")
	assert.Contains(t, err.Error(), "must be a positive duration")

	expected := "function test_func: config 'timeout': must be a positive duration"
	assert.Equal(t, expected, err.Error())
}

// TestErrorContext_Wrap tests structured error context wrapping
func TestErrorContext_Wrap(t *testing.T) {
	fn := setupTestFunction()
	originalErr := errors.New("connection timeout")

	ctx := &ErrorContext{
		Function:  fn,
		Operation: "api_request",
		Details: map[string]interface{}{
			"url":    "https://api.example.com",
			"status": 500,
		},
	}

	wrappedErr := ctx.Wrap(originalErr)

	require.Error(t, wrappedErr)
	assert.Contains(t, wrappedErr.Error(), "function test_func")
	assert.Contains(t, wrappedErr.Error(), "api_request")
	assert.Contains(t, wrappedErr.Error(), "url: https://api.example.com")
	assert.Contains(t, wrappedErr.Error(), "status: 500")
	assert.Contains(t, wrappedErr.Error(), "connection timeout")
	assert.True(t, errors.Is(wrappedErr, originalErr))
}

// TestErrorContext_Wrap_NilError tests ErrorContext with nil error
func TestErrorContext_Wrap_NilError(t *testing.T) {
	fn := setupTestFunction()

	ctx := &ErrorContext{
		Function:  fn,
		Operation: "test",
		Details:   map[string]interface{}{"key": "value"},
	}

	wrappedErr := ctx.Wrap(nil)
	assert.Nil(t, wrappedErr)
}

// TestErrorContext_Wrap_NoDetails tests ErrorContext without details
func TestErrorContext_Wrap_NoDetails(t *testing.T) {
	fn := setupTestFunction()
	originalErr := errors.New("something failed")

	ctx := &ErrorContext{
		Function:  fn,
		Operation: "operation",
		Details:   nil,
	}

	wrappedErr := ctx.Wrap(originalErr)

	require.Error(t, wrappedErr)
	assert.Contains(t, wrappedErr.Error(), "function test_func")
	assert.Contains(t, wrappedErr.Error(), "operation")
	assert.Contains(t, wrappedErr.Error(), "something failed")

	// Should not contain parentheses when no details
	assert.NotContains(t, wrappedErr.Error(), "(")
	assert.True(t, errors.Is(wrappedErr, originalErr))
}

// TestErrorContext_Error tests creating new errors with context
func TestErrorContext_Error(t *testing.T) {
	fn := setupTestFunction()

	ctx := &ErrorContext{
		Function:  fn,
		Operation: "retry",
		Details: map[string]interface{}{
			"attempt":    3,
			"max_retries": 3,
		},
	}

	err := ctx.Error("maximum retries exceeded")

	require.Error(t, err)
	assert.Contains(t, err.Error(), "function test_func")
	assert.Contains(t, err.Error(), "retry")
	assert.Contains(t, err.Error(), "attempt: 3")
	assert.Contains(t, err.Error(), "max_retries: 3")
	assert.Contains(t, err.Error(), "maximum retries exceeded")
}

// TestErrorChaining tests that errors can be unwrapped properly
func TestErrorChaining(t *testing.T) {
	fn := setupTestFunction()

	// Create a chain of errors
	baseErr := errors.New("base error")
	wrappedOnce := fmt.Errorf("first wrap: %w", baseErr)
	wrappedTwice := WrapExecutionError(fn, "operation", wrappedOnce)

	// Should be able to find the base error
	assert.True(t, errors.Is(wrappedTwice, baseErr))
	assert.True(t, errors.Is(wrappedTwice, wrappedOnce))

	// Error message should contain all context
	assert.Contains(t, wrappedTwice.Error(), "function test_func")
	assert.Contains(t, wrappedTwice.Error(), "operation")
	assert.Contains(t, wrappedTwice.Error(), "first wrap")
	assert.Contains(t, wrappedTwice.Error(), "base error")
}

// TestMetadataFallback tests that error helpers use metadata fallback
func TestMetadataFallback(t *testing.T) {
	fn := NewMockFunction()
	// Only set derived name (struct name)
	fn.metadata = NewMetadata("MyFunction")

	err := NewExecutionError(fn, "test", "error message")

	require.Error(t, err)
	// Should use derived name when no explicit name is set
	assert.Contains(t, err.Error(), "MyFunction")
}

// TestErrorContext_MultipleDetails tests formatting multiple details
func TestErrorContext_MultipleDetails(t *testing.T) {
	fn := setupTestFunction()

	ctx := &ErrorContext{
		Function:  fn,
		Operation: "complex_op",
		Details: map[string]interface{}{
			"param1": "value1",
			"param2": 42,
			"param3": true,
		},
	}

	err := ctx.Error("operation failed")

	require.Error(t, err)
	// Should contain all details (order may vary due to map iteration)
	assert.Contains(t, err.Error(), "param1: value1")
	assert.Contains(t, err.Error(), "param2: 42")
	assert.Contains(t, err.Error(), "param3: true")

	// Should have proper formatting
	assert.Contains(t, err.Error(), "function test_func: complex_op (")
	assert.True(t, strings.Contains(err.Error(), "): operation failed"))
}
