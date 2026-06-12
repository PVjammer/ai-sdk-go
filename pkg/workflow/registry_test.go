package workflow

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/pvjammer/ai-sdk-go/pkg/function"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Mock workflow implementations for testing.

type mockWorkflow struct {
	WorkflowBase
	name string
}

func newMockWorkflow(name string) FactoryFunc {
	return func(cfg function.Config) Workflow {
		return &mockWorkflow{
			WorkflowBase: NewWorkflowBase(cfg),
			name:         name,
		}
	}
}

func (w *mockWorkflow) Execute(_ context.Context, _ interface{}) (interface{}, error) {
	return w.name + "_result", nil
}

func (w *mockWorkflow) Metadata() *WorkflowMetadata {
	return &WorkflowMetadata{Name: w.name}
}

// TestNew_EmptyRegistry tests creating a new registry.
func TestNew_EmptyRegistry(t *testing.T) {
	reg := New()
	require.NotNil(t, reg)
	assert.Empty(t, reg.List())
	assert.Equal(t, 0, reg.Count())
}

// TestRegister tests basic registration.
func TestRegister(t *testing.T) {
	reg := New()
	reg.Register("wf1", newMockWorkflow("wf1"))
	assert.True(t, reg.Has("wf1"))
	assert.Equal(t, 1, reg.Count())
}

// TestRegister_Replace tests that re-registering replaces the factory.
func TestRegister_Replace(t *testing.T) {
	reg := New()
	reg.Register("wf", newMockWorkflow("first"))
	reg.Register("wf", newMockWorkflow("second"))
	assert.Equal(t, 1, reg.Count())

	wf := reg.Get("wf", function.NilConfig{})
	require.NotNil(t, wf)
	out, err := wf.Execute(context.Background(), nil)
	require.NoError(t, err)
	assert.Equal(t, "second_result", out)
}

// TestGet tests retrieving a workflow.
func TestGet(t *testing.T) {
	reg := New()
	reg.Register("alpha", newMockWorkflow("alpha"))

	wf := reg.Get("alpha", function.NilConfig{})
	require.NotNil(t, wf)

	out, err := wf.Execute(context.Background(), nil)
	require.NoError(t, err)
	assert.Equal(t, "alpha_result", out)
}

// TestGet_NotFound returns nil for unregistered names.
func TestGet_NotFound(t *testing.T) {
	reg := New()
	wf := reg.Get("nonexistent", function.NilConfig{})
	assert.Nil(t, wf)
}

// TestGet_NewInstanceEachTime verifies factory is called each time.
func TestGet_NewInstanceEachTime(t *testing.T) {
	reg := New()
	reg.Register("wf", newMockWorkflow("wf"))

	wf1 := reg.Get("wf", function.NilConfig{})
	wf2 := reg.Get("wf", function.NilConfig{})
	require.NotNil(t, wf1)
	require.NotNil(t, wf2)
	assert.NotSame(t, wf1, wf2)
}

// TestHas tests existence check.
func TestHas(t *testing.T) {
	reg := New()
	assert.False(t, reg.Has("wf"))
	reg.Register("wf", newMockWorkflow("wf"))
	assert.True(t, reg.Has("wf"))
}

// TestList tests sorted name listing.
func TestList(t *testing.T) {
	reg := New()
	reg.Register("charlie", newMockWorkflow("charlie"))
	reg.Register("alpha", newMockWorkflow("alpha"))
	reg.Register("bravo", newMockWorkflow("bravo"))

	names := reg.List()
	assert.Equal(t, []string{"alpha", "bravo", "charlie"}, names)
}

// TestListWithMetadata tests metadata listing.
func TestListWithMetadata(t *testing.T) {
	reg := New()
	reg.RegisterWithMetadata("wf-b", newMockWorkflow("wf-b"), "B workflow", "cat-b", []string{"tag1"})
	reg.RegisterWithMetadata("wf-a", newMockWorkflow("wf-a"), "A workflow", "cat-a", []string{"tag2"})

	regs := reg.ListWithMetadata()
	require.Len(t, regs, 2)
	assert.Equal(t, "wf-a", regs[0].Name)
	assert.Equal(t, "wf-b", regs[1].Name)
	assert.Equal(t, "A workflow", regs[0].Description)
	assert.Equal(t, "cat-a", regs[0].Category)
}

// TestGetByCategory tests category filtering.
func TestGetByCategory(t *testing.T) {
	reg := New()
	reg.RegisterWithMetadata("wf-a", newMockWorkflow("wf-a"), "", "sync", nil)
	reg.RegisterWithMetadata("wf-b", newMockWorkflow("wf-b"), "", "sync", nil)
	reg.RegisterWithMetadata("wf-c", newMockWorkflow("wf-c"), "", "report", nil)

	syncWfs := reg.GetByCategory("sync")
	assert.Equal(t, []string{"wf-a", "wf-b"}, syncWfs)

	reportWfs := reg.GetByCategory("report")
	assert.Equal(t, []string{"wf-c"}, reportWfs)

	assert.Empty(t, reg.GetByCategory("nonexistent"))
}

// TestClear tests removing all registrations.
func TestClear(t *testing.T) {
	reg := New()
	reg.Register("wf1", newMockWorkflow("wf1"))
	reg.Register("wf2", newMockWorkflow("wf2"))
	assert.Equal(t, 2, reg.Count())

	reg.Clear()
	assert.Equal(t, 0, reg.Count())
	assert.Empty(t, reg.List())
}

// TestMustGet_Panics tests that MustGet panics for unregistered names.
func TestMustGet_Panics(t *testing.T) {
	reg := New()
	assert.Panics(t, func() {
		reg.MustGet("nonexistent", function.NilConfig{})
	})
}

// TestGlobal_Singleton tests the global singleton.
func TestGlobal_Singleton(t *testing.T) {
	// Save and restore global state.
	ResetGlobal()
	t.Cleanup(ResetGlobal)

	g1 := Global()
	g2 := Global()
	assert.Same(t, g1, g2)
}

// TestSetGlobal tests replacing the global registry.
func TestSetGlobal(t *testing.T) {
	ResetGlobal()
	t.Cleanup(ResetGlobal)

	custom := New()
	custom.Register("custom-wf", newMockWorkflow("custom-wf"))
	SetGlobal(custom)

	assert.True(t, Global().Has("custom-wf"))
}

// TestRegistry_ThreadSafe tests concurrent access.
func TestRegistry_ThreadSafe(t *testing.T) {
	reg := New()
	var wg sync.WaitGroup

	// 10 goroutines × 100 operations
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			name := fmt.Sprintf("wf-%d", id)
			for j := 0; j < 100; j++ {
				reg.Register(name, newMockWorkflow(name))
				reg.Has(name)
				reg.Get(name, function.NilConfig{})
				reg.List()
				reg.Count()
			}
		}(i)
	}

	wg.Wait()
	assert.Equal(t, 10, reg.Count())
}

