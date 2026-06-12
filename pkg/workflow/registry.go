package workflow

import (
	"fmt"
	"sort"
	"sync"

	"github.com/pvjammer/ai-sdk-go/pkg/function"
)

// FactoryFunc creates a new Workflow instance with the given config.
type FactoryFunc func(cfg function.Config) Workflow

// Registration holds information about a registered workflow.
type Registration struct {
	Name        string
	Description string
	Category    string
	Tags        []string
	Factory     FactoryFunc
}

// Registry manages registered Workflows.
//
// Thread-safe; workflows register via init() and are created on demand.
// Mirrors pkg/agentreg/registry.go with additional metadata support.
type Registry struct {
	mu            sync.RWMutex
	registrations map[string]*Registration
}

// New creates a new empty Registry.
func New() *Registry {
	return &Registry{
		registrations: make(map[string]*Registration),
	}
}

// Register registers a workflow factory with the given name.
//
// If a factory with the same name already exists it is replaced.
//
// Example:
//
//	func init() {
//	    workflow.Global().Register("jira-sync", NewJiraSync)
//	}
func (r *Registry) Register(name string, factory FactoryFunc) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.registrations[name] = &Registration{
		Name:    name,
		Factory: factory,
	}
}

// RegisterWithMetadata registers a workflow factory with full metadata.
func (r *Registry) RegisterWithMetadata(name string, factory FactoryFunc, desc, category string, tags []string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.registrations[name] = &Registration{
		Name:        name,
		Description: desc,
		Category:    category,
		Tags:        tags,
		Factory:     factory,
	}
}

// Get retrieves a workflow by name, creating a new instance with the given config.
//
// Returns nil if no workflow is registered under that name.
func (r *Registry) Get(name string, cfg function.Config) Workflow {
	r.mu.RLock()
	defer r.mu.RUnlock()

	reg, ok := r.registrations[name]
	if !ok {
		return nil
	}

	return reg.Factory(cfg)
}

// Has reports whether a workflow is registered under the given name.
func (r *Registry) Has(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	_, ok := r.registrations[name]
	return ok
}

// List returns all registered workflow names, sorted alphabetically.
func (r *Registry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.registrations))
	for name := range r.registrations {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// ListWithMetadata returns all registrations sorted alphabetically by name.
func (r *Registry) ListWithMetadata() []*Registration {
	r.mu.RLock()
	defer r.mu.RUnlock()

	regs := make([]*Registration, 0, len(r.registrations))
	for _, reg := range r.registrations {
		regs = append(regs, reg)
	}
	sort.Slice(regs, func(i, j int) bool {
		return regs[i].Name < regs[j].Name
	})
	return regs
}

// GetByCategory returns names of workflows in the given category, sorted alphabetically.
func (r *Registry) GetByCategory(category string) []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var names []string
	for name, reg := range r.registrations {
		if reg.Category == category {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	return names
}

// Count returns the number of registered workflows.
func (r *Registry) Count() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.registrations)
}

// Clear removes all registered workflows.
func (r *Registry) Clear() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.registrations = make(map[string]*Registration)
}

// MustGet retrieves a workflow by name or panics if not found.
func (r *Registry) MustGet(name string, cfg function.Config) Workflow {
	wf := r.Get(name, cfg)
	if wf == nil {
		panic(fmt.Sprintf("workflow not registered: %s (available: %v)", name, r.List()))
	}
	return wf
}

// Global registry singleton.
var (
	globalRegistry     *Registry
	globalRegistryOnce sync.Once
	globalRegistryMu   sync.Mutex
)

// Global returns the global workflow registry instance (lazily initialized).
func Global() *Registry {
	globalRegistryOnce.Do(func() {
		globalRegistry = New()
	})
	return globalRegistry
}

// SetGlobal replaces the global registry. Intended for testing.
func SetGlobal(r *Registry) {
	globalRegistryMu.Lock()
	defer globalRegistryMu.Unlock()
	globalRegistry = r
}

// ResetGlobal resets the global registry to a fresh empty instance.
// Intended for testing — resets the sync.Once so Global() re-initializes.
func ResetGlobal() {
	globalRegistryMu.Lock()
	defer globalRegistryMu.Unlock()
	globalRegistry = New()
	globalRegistryOnce = sync.Once{}
	// Pre-populate so the next Global() call returns our fresh instance.
	globalRegistryOnce.Do(func() {})
}
