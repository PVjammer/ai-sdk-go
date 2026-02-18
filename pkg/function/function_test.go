package function

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// MockFunction is a simple test function
type MockFunction struct {
	Base
	executeFunc func(ctx context.Context, input interface{}) (interface{}, error)
	schema      *Schema
	metadata    *Metadata
}

func NewMockFunction() *MockFunction {
	return &MockFunction{
		Base: NewBase(NilConfig{}),
		schema: &Schema{
			Type:       "object",
			Properties: make(map[string]*Property),
			Required:   []string{},
		},
		metadata: NewMetadata("MockFunction").WithName("mock"),
	}
}

func (f *MockFunction) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	if f.executeFunc != nil {
		return f.executeFunc(ctx, input)
	}
	return input, nil
}

func (f *MockFunction) InputSchema() *Schema {
	return f.schema
}

func (f *MockFunction) Metadata() *Metadata {
	return f.metadata
}

// TestInvoke tests synchronous invocation
func TestInvoke(t *testing.T) {
	fn := NewMockFunction()
	fn.executeFunc = func(ctx context.Context, input interface{}) (interface{}, error) {
		return "result", nil
	}

	result, err := Invoke(context.Background(), fn, "input")
	require.NoError(t, err)
	assert.Equal(t, "result", result)
}

// TestInvoke_Error tests error handling
func TestInvoke_Error(t *testing.T) {
	fn := NewMockFunction()
	expectedErr := errors.New("test error")
	fn.executeFunc = func(ctx context.Context, input interface{}) (interface{}, error) {
		return nil, expectedErr
	}

	result, err := Invoke(context.Background(), fn, "input")
	assert.Nil(t, result)
	assert.Equal(t, expectedErr, err)
}

// TestAsyncInvoke tests asynchronous invocation
func TestAsyncInvoke(t *testing.T) {
	fn := NewMockFunction()
	fn.executeFunc = func(ctx context.Context, input interface{}) (interface{}, error) {
		time.Sleep(10 * time.Millisecond) // Simulate work
		return "async result", nil
	}

	resultCh := AsyncInvoke(context.Background(), fn, "input")
	result := <-resultCh

	require.NoError(t, result.Error)
	assert.Equal(t, "async result", result.Output)
}

// TestAsyncInvoke_Error tests async error handling
func TestAsyncInvoke_Error(t *testing.T) {
	fn := NewMockFunction()
	expectedErr := errors.New("async error")
	fn.executeFunc = func(ctx context.Context, input interface{}) (interface{}, error) {
		return nil, expectedErr
	}

	resultCh := AsyncInvoke(context.Background(), fn, "input")
	result := <-resultCh

	assert.Nil(t, result.Output)
	assert.Equal(t, expectedErr, result.Error)
}

// TestInvokeAll tests concurrent execution
func TestInvokeAll(t *testing.T) {
	fn1 := NewMockFunction()
	fn1.executeFunc = func(ctx context.Context, input interface{}) (interface{}, error) {
		time.Sleep(10 * time.Millisecond)
		return "result1", nil
	}

	fn2 := NewMockFunction()
	fn2.executeFunc = func(ctx context.Context, input interface{}) (interface{}, error) {
		time.Sleep(5 * time.Millisecond)
		return "result2", nil
	}

	fn3 := NewMockFunction()
	fn3.executeFunc = func(ctx context.Context, input interface{}) (interface{}, error) {
		time.Sleep(15 * time.Millisecond)
		return "result3", nil
	}

	calls := []FunctionCall{
		{Function: fn1, Input: "input1"},
		{Function: fn2, Input: "input2"},
		{Function: fn3, Input: "input3"},
	}

	results := InvokeAll(context.Background(), calls)

	// Collect all results
	var outputs []string
	for result := range results {
		require.NoError(t, result.Error)
		outputs = append(outputs, result.Output.(string))
	}

	// Should have all 3 results
	assert.Len(t, outputs, 3)
	assert.Contains(t, outputs, "result1")
	assert.Contains(t, outputs, "result2")
	assert.Contains(t, outputs, "result3")
}

// TestInvokeAll_WithErrors tests concurrent execution with errors
func TestInvokeAll_WithErrors(t *testing.T) {
	fn1 := NewMockFunction()
	fn1.executeFunc = func(ctx context.Context, input interface{}) (interface{}, error) {
		return "success", nil
	}

	fn2 := NewMockFunction()
	expectedErr := errors.New("failed")
	fn2.executeFunc = func(ctx context.Context, input interface{}) (interface{}, error) {
		return nil, expectedErr
	}

	calls := []FunctionCall{
		{Function: fn1, Input: "input1"},
		{Function: fn2, Input: "input2"},
	}

	results := InvokeAll(context.Background(), calls)

	// Collect results
	var successCount, errorCount int
	for result := range results {
		if result.Error != nil {
			errorCount++
			assert.Equal(t, expectedErr, result.Error)
		} else {
			successCount++
			assert.Equal(t, "success", result.Output)
		}
	}

	assert.Equal(t, 1, successCount)
	assert.Equal(t, 1, errorCount)
}

// TestInvokeAll_RaceCondition tests for race conditions
// Run with: go test -race
func TestInvokeAll_RaceCondition(t *testing.T) {
	// Create a function that increments a counter
	var counter int64
	fn := NewMockFunction()
	fn.executeFunc = func(ctx context.Context, input interface{}) (interface{}, error) {
		atomic.AddInt64(&counter, 1)
		time.Sleep(1 * time.Millisecond)
		return atomic.LoadInt64(&counter), nil
	}

	// Run many concurrent invocations
	const numCalls = 100
	calls := make([]FunctionCall, numCalls)
	for i := 0; i < numCalls; i++ {
		calls[i] = FunctionCall{Function: fn, Input: i}
	}

	results := InvokeAll(context.Background(), calls)

	// Collect all results
	var resultCount int
	for result := range results {
		require.NoError(t, result.Error)
		resultCount++
	}

	// Should have exactly numCalls results
	assert.Equal(t, numCalls, resultCount)
	assert.Equal(t, int64(numCalls), atomic.LoadInt64(&counter))
}

// TestInvokeAll_EmptySlice tests InvokeAll with no calls
func TestInvokeAll_EmptySlice(t *testing.T) {
	results := InvokeAll(context.Background(), []FunctionCall{})

	// Channel should be closed immediately with no results
	var count int
	for range results {
		count++
	}

	assert.Equal(t, 0, count)
}

// TestInvokeAll_ContextCancellation tests context cancellation behavior
func TestInvokeAll_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	fn := NewMockFunction()
	fn.executeFunc = func(ctx context.Context, input interface{}) (interface{}, error) {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(100 * time.Millisecond):
			return "completed", nil
		}
	}

	calls := []FunctionCall{
		{Function: fn, Input: "input1"},
		{Function: fn, Input: "input2"},
		{Function: fn, Input: "input3"},
	}

	results := InvokeAll(ctx, calls)

	// Cancel context immediately
	cancel()

	// Collect results - should get cancellation errors
	var canceledCount int
	for result := range results {
		if result.Error == context.Canceled {
			canceledCount++
		}
	}

	// Should have at least some canceled results
	assert.Greater(t, canceledCount, 0)
}
