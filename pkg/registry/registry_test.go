package registry

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	"github.com/pvjammer/ai-sdk-go/internal/schema"
	"github.com/pvjammer/ai-sdk-go/pkg/function"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Mock functions for testing

type MockFunction1 struct {
	function.Base
}

type MockInput1 struct {
	Text string `json:"text"`
}

func NewMockFunction1(cfg function.Config) function.AIFunction {
	return &MockFunction1{Base: function.NewBase(cfg)}
}

func (f *MockFunction1) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	return "mock1_result", nil
}

func (f *MockFunction1) InputSchema() *function.Schema {
	return schema.MustGenerateSchema(reflect.TypeOf(MockInput1{}))
}

func (f *MockFunction1) Metadata() *function.Metadata {
	return function.NewMetadata("MockFunction1").
		WithName("mock1").
		WithDescription("Mock function 1").
		WithCategory("test").
		WithTags("mock", "test", "function1")
}

type MockFunction2 struct {
	function.Base
}

func NewMockFunction2(cfg function.Config) function.AIFunction {
	return &MockFunction2{Base: function.NewBase(cfg)}
}

func (f *MockFunction2) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	return "mock2_result", nil
}

func (f *MockFunction2) InputSchema() *function.Schema {
	return schema.MustGenerateSchema(reflect.TypeOf(MockInput1{}))
}

func (f *MockFunction2) Metadata() *function.Metadata {
	return function.NewMetadata("MockFunction2").
		WithName("mock2").
		WithDescription("Mock function 2").
		WithCategory("text").
		WithTags("mock", "text", "function2")
}

// TestNew tests creating a new registry
func TestNew(t *testing.T) {
	reg := New()
	require.NotNil(t, reg)
	assert.Empty(t, reg.List())
	assert.Equal(t, 0, reg.Count())
}

// TestRegister tests registering a function
func TestRegister(t *testing.T) {
	reg := New()

	reg.Register("mock1", NewMockFunction1)

	assert.True(t, reg.Has("mock1"))
	assert.Equal(t, 1, reg.Count())
}

// TestRegister_Replace tests replacing a registration
func TestRegister_Replace(t *testing.T) {
	reg := New()

	reg.Register("mock", NewMockFunction1)
	reg.Register("mock", NewMockFunction2)

	// Should have only one registration
	assert.Equal(t, 1, reg.Count())

	// Should be the second function
	fn := reg.Get("mock", function.NilConfig{})
	result, _ := fn.Execute(context.Background(), nil)
	assert.Equal(t, "mock2_result", result)
}

// TestRegisterWithMetadata tests registering with metadata
func TestRegisterWithMetadata(t *testing.T) {
	reg := New()

	reg.RegisterWithMetadata("test_func", NewMockFunction1,
		"Test function description",
		"test_category",
		[]string{"tag1", "tag2"})

	assert.True(t, reg.Has("test_func"))

	regs := reg.ListWithMetadata()
	require.Len(t, regs, 1)

	assert.Equal(t, "test_func", regs[0].Name)
	assert.Equal(t, "Test function description", regs[0].Description)
	assert.Equal(t, "test_category", regs[0].Category)
	assert.Equal(t, []string{"tag1", "tag2"}, regs[0].Tags)
}

// TestRegisterSafe tests safe registration
func TestRegisterSafe(t *testing.T) {
	reg := New()

	// Valid registration
	err := reg.RegisterSafe("mock", NewMockFunction1)
	assert.NoError(t, err)

	// Empty name
	err = reg.RegisterSafe("", NewMockFunction1)
	assert.Error(t, err)

	// Nil factory
	err = reg.RegisterSafe("test", nil)
	assert.Error(t, err)
}

// TestMustRegister tests MustRegister
func TestMustRegister(t *testing.T) {
	reg := New()

	// Should not panic for valid registration
	assert.NotPanics(t, func() {
		reg.MustRegister("mock", NewMockFunction1)
	})

	// Should panic for invalid registration
	assert.Panics(t, func() {
		reg.MustRegister("", NewMockFunction1)
	})
}

// TestGet tests retrieving a function
func TestGet(t *testing.T) {
	reg := New()
	reg.Register("mock1", NewMockFunction1)

	fn := reg.Get("mock1", function.NilConfig{})
	require.NotNil(t, fn)

	// Verify it's the right function
	result, err := fn.Execute(context.Background(), nil)
	require.NoError(t, err)
	assert.Equal(t, "mock1_result", result)
}

// TestGet_NonExistent tests getting a non-existent function
func TestGet_NonExistent(t *testing.T) {
	reg := New()

	fn := reg.Get("nonexistent", function.NilConfig{})
	assert.Nil(t, fn)
}

// TestGet_CreatesNewInstances tests that Get creates new instances
func TestGet_CreatesNewInstances(t *testing.T) {
	reg := New()
	reg.Register("mock", NewMockFunction1)

	fn1 := reg.Get("mock", function.NilConfig{})
	fn2 := reg.Get("mock", function.NilConfig{})

	// Should be different instances
	assert.NotSame(t, fn1, fn2)
}

// TestHas tests checking function existence
func TestHas(t *testing.T) {
	reg := New()
	reg.Register("mock", NewMockFunction1)

	assert.True(t, reg.Has("mock"))
	assert.False(t, reg.Has("nonexistent"))
}

// TestList tests listing function names
func TestList(t *testing.T) {
	reg := New()

	// Empty registry
	assert.Empty(t, reg.List())

	// Add functions
	reg.Register("function_c", NewMockFunction1)
	reg.Register("function_a", NewMockFunction2)
	reg.Register("function_b", NewMockFunction1)

	names := reg.List()
	assert.Len(t, names, 3)

	// Should be sorted
	assert.Equal(t, []string{"function_a", "function_b", "function_c"}, names)
}

// TestListWithMetadata tests listing with metadata
func TestListWithMetadata(t *testing.T) {
	reg := New()

	reg.Register("mock1", NewMockFunction1)
	reg.Register("mock2", NewMockFunction2)

	regs := reg.ListWithMetadata()
	require.Len(t, regs, 2)

	// Should be sorted by name
	assert.Equal(t, "mock1", regs[0].Name)
	assert.Equal(t, "mock2", regs[1].Name)

	// Should have metadata extracted from functions
	assert.Equal(t, "Mock function 1", regs[0].Description)
	assert.Equal(t, "test", regs[0].Category)
	assert.Contains(t, regs[0].Tags, "mock")

	assert.Equal(t, "Mock function 2", regs[1].Description)
	assert.Equal(t, "text", regs[1].Category)
}

// TestGetByCategory tests filtering by category
func TestGetByCategory(t *testing.T) {
	reg := New()

	reg.Register("mock1", NewMockFunction1) // category: test
	reg.Register("mock2", NewMockFunction2) // category: text

	testFuncs := reg.GetByCategory("test")
	assert.Equal(t, []string{"mock1"}, testFuncs)

	textFuncs := reg.GetByCategory("text")
	assert.Equal(t, []string{"mock2"}, textFuncs)

	nonExistent := reg.GetByCategory("nonexistent")
	assert.Empty(t, nonExistent)
}

// TestGetByTag tests filtering by single tag
func TestGetByTag(t *testing.T) {
	reg := New()

	reg.Register("mock1", NewMockFunction1) // tags: mock, test, function1
	reg.Register("mock2", NewMockFunction2) // tags: mock, text, function2

	mockFuncs := reg.GetByTag("mock")
	assert.Len(t, mockFuncs, 2)
	assert.Contains(t, mockFuncs, "mock1")
	assert.Contains(t, mockFuncs, "mock2")

	testFuncs := reg.GetByTag("test")
	assert.Equal(t, []string{"mock1"}, testFuncs)

	textFuncs := reg.GetByTag("text")
	assert.Equal(t, []string{"mock2"}, textFuncs)
}

// TestGetByTags tests filtering by multiple tags (AND)
func TestGetByTags(t *testing.T) {
	reg := New()

	reg.Register("mock1", NewMockFunction1) // tags: mock, test, function1
	reg.Register("mock2", NewMockFunction2) // tags: mock, text, function2

	// Both have "mock"
	bothFuncs := reg.GetByTags("mock")
	assert.Len(t, bothFuncs, 2)

	// Only mock1 has both "mock" and "test"
	testFuncs := reg.GetByTags("mock", "test")
	assert.Equal(t, []string{"mock1"}, testFuncs)

	// Only mock2 has both "mock" and "text"
	textFuncs := reg.GetByTags("mock", "text")
	assert.Equal(t, []string{"mock2"}, textFuncs)

	// No function has both "test" and "text"
	noFuncs := reg.GetByTags("test", "text")
	assert.Empty(t, noFuncs)

	// No tags returns all
	allFuncs := reg.GetByTags()
	assert.Len(t, allFuncs, 2)
}

// TestGetCategories tests getting all categories
func TestGetCategories(t *testing.T) {
	reg := New()

	// Empty registry
	assert.Empty(t, reg.GetCategories())

	// Add functions
	reg.Register("mock1", NewMockFunction1) // category: test
	reg.Register("mock2", NewMockFunction2) // category: text

	categories := reg.GetCategories()
	assert.Len(t, categories, 2)
	assert.Contains(t, categories, "test")
	assert.Contains(t, categories, "text")

	// Should be sorted
	assert.Equal(t, []string{"test", "text"}, categories)
}

// TestGetTags tests getting all tags
func TestGetTags(t *testing.T) {
	reg := New()

	// Empty registry
	assert.Empty(t, reg.GetTags())

	// Add functions
	reg.Register("mock1", NewMockFunction1) // tags: mock, test, function1
	reg.Register("mock2", NewMockFunction2) // tags: mock, text, function2

	tags := reg.GetTags()

	// Should have all unique tags
	assert.Contains(t, tags, "mock")
	assert.Contains(t, tags, "test")
	assert.Contains(t, tags, "text")
	assert.Contains(t, tags, "function1")
	assert.Contains(t, tags, "function2")

	// Should be sorted
	for i := 1; i < len(tags); i++ {
		assert.True(t, tags[i-1] < tags[i], "tags should be sorted")
	}
}

// TestCount tests counting registrations
func TestCount(t *testing.T) {
	reg := New()

	assert.Equal(t, 0, reg.Count())

	reg.Register("mock1", NewMockFunction1)
	assert.Equal(t, 1, reg.Count())

	reg.Register("mock2", NewMockFunction2)
	assert.Equal(t, 2, reg.Count())

	// Replacing doesn't increase count
	reg.Register("mock1", NewMockFunction1)
	assert.Equal(t, 2, reg.Count())
}

// TestClear tests clearing the registry
func TestClear(t *testing.T) {
	reg := New()

	reg.Register("mock1", NewMockFunction1)
	reg.Register("mock2", NewMockFunction2)
	assert.Equal(t, 2, reg.Count())

	reg.Clear()
	assert.Equal(t, 0, reg.Count())
	assert.Empty(t, reg.List())
}

// TestGlobal tests the global registry
func TestGlobal(t *testing.T) {
	// Get global registry
	reg1 := Global()
	require.NotNil(t, reg1)

	// Should return same instance
	reg2 := Global()
	assert.Same(t, reg1, reg2)
}

// TestSetGlobal tests setting a custom global registry
func TestSetGlobal(t *testing.T) {
	// Create custom registry
	custom := New()
	custom.Register("custom", NewMockFunction1)

	// Set as global
	SetGlobal(custom)

	// Verify global is our custom registry
	global := Global()
	assert.True(t, global.Has("custom"))
}

// TestResetGlobal tests resetting the global registry
func TestResetGlobal(t *testing.T) {
	// Add something to global
	Global().Register("test", NewMockFunction1)
	assert.True(t, Global().Has("test"))

	// Reset
	ResetGlobal()

	// Should be empty
	assert.False(t, Global().Has("test"))
	assert.Equal(t, 0, Global().Count())
}

// TestConcurrentAccess tests thread-safe concurrent access
func TestConcurrentAccess(t *testing.T) {
	reg := New()

	const goroutines = 10
	const operations = 100

	// Concurrent registration
	done := make(chan bool, goroutines)
	for i := 0; i < goroutines; i++ {
		go func(id int) {
			for j := 0; j < operations; j++ {
				name := fmt.Sprintf("func_%d_%d", id, j)
				reg.Register(name, NewMockFunction1)
			}
			done <- true
		}(i)
	}

	// Wait for all goroutines
	for i := 0; i < goroutines; i++ {
		<-done
	}

	// Should have all registrations
	assert.Equal(t, goroutines*operations, reg.Count())

	// Concurrent reads
	for i := 0; i < goroutines; i++ {
		go func() {
			for j := 0; j < operations; j++ {
				_ = reg.List()
				_ = reg.GetCategories()
				_ = reg.GetTags()
			}
			done <- true
		}()
	}

	// Wait for all goroutines
	for i := 0; i < goroutines; i++ {
		<-done
	}
}

// TestHasAllTags tests the hasAllTags helper
func TestHasAllTags(t *testing.T) {
	available := []string{"a", "b", "c", "d"}

	assert.True(t, hasAllTags(available, []string{"a"}))
	assert.True(t, hasAllTags(available, []string{"a", "b"}))
	assert.True(t, hasAllTags(available, []string{"a", "c"}))
	assert.True(t, hasAllTags(available, []string{"a", "b", "c", "d"}))

	assert.False(t, hasAllTags(available, []string{"e"}))
	assert.False(t, hasAllTags(available, []string{"a", "e"}))
	assert.False(t, hasAllTags(available, []string{"e", "f"}))
}
