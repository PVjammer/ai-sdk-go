// Package agentreg provides agent discovery and registration.
//
// It mirrors pkg/registry for AIFunctions but for AIAgents.
// Agents register themselves via init() in their package, and the chat
// command discovers them by name at startup.
package agentreg

import (
	"fmt"
	"sort"
	"sync"

	"github.com/pvjammer/ai-sdk-go/pkg/agent"
	"github.com/pvjammer/ai-sdk-go/pkg/function"
)

// FactoryFunc creates a new AIAgent instance with the given config.
type FactoryFunc func(cfg function.Config) agent.AIAgent

// Registration holds information about a registered agent.
type Registration struct {
	Name    string
	Factory FactoryFunc
}

// Registry manages registered AI agents.
//
// Thread-safe; agents register via init() and are created on demand.
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

// Register registers an agent factory with the given name.
//
// If a factory with the same name already exists it is replaced.
//
// Example:
//
//	func init() {
//	    agentreg.Global().Register("simple", New)
//	}
func (r *Registry) Register(name string, factory FactoryFunc) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.registrations[name] = &Registration{
		Name:    name,
		Factory: factory,
	}
}

// Get retrieves an agent by name, creating a new instance with the given config.
//
// Returns nil if no agent is registered under that name.
func (r *Registry) Get(name string, cfg function.Config) agent.AIAgent {
	r.mu.RLock()
	defer r.mu.RUnlock()

	reg, ok := r.registrations[name]
	if !ok {
		return nil
	}

	return reg.Factory(cfg)
}

// Has reports whether an agent is registered under the given name.
func (r *Registry) Has(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	_, ok := r.registrations[name]
	return ok
}

// List returns all registered agent names, sorted alphabetically.
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

// MustGet retrieves an agent by name or panics if not found.
func (r *Registry) MustGet(name string, cfg function.Config) agent.AIAgent {
	a := r.Get(name, cfg)
	if a == nil {
		panic(fmt.Sprintf("agent not registered: %s (available: %v)", name, r.List()))
	}
	return a
}

// Global registry singleton
var (
	globalRegistry *Registry
	globalOnce     sync.Once
)

// Global returns the global agent registry instance (lazily initialized).
func Global() *Registry {
	globalOnce.Do(func() {
		globalRegistry = New()
	})
	return globalRegistry
}
