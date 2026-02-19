// Package registry provides function discovery and registration.
//
// The registry uses a factory pattern to allow functions to be registered
// without creating instances until needed. This supports plugin-style
// architectures where functions can register themselves via init().
package registry

import (
	"fmt"
	"sort"
	"sync"

	"github.com/pvjammer/ai-sdk-go/pkg/config"
	"github.com/pvjammer/ai-sdk-go/pkg/function"
)

// FactoryFunc is a function that creates a new AIFunction instance.
//
// The factory receives a Config which it can use to initialize the function.
//
// Example:
//
//	func NewTextTransform(cfg function.Config) function.AIFunction {
//	    return texttransform.New(cfg)
//	}
type FactoryFunc func(cfg function.Config) function.AIFunction

// Registration holds information about a registered function.
type Registration struct {
	Name             string
	Factory          FactoryFunc
	Description      string
	Category         string
	Tags             []string
	ConfigParams     []ConfigParam
	RequiresApproval bool // default for the requires_approval config key
}

// Registry manages registered AI functions.
//
// The registry is thread-safe and supports concurrent access.
// Functions are registered by name with a factory function that
// creates instances on demand.
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

// Register registers a function factory with the given name.
//
// The factory will be called each time Get() is called to create
// a new function instance with fresh state.
//
// Example:
//
//	registry.Register("text_transform", texttransform.New)
//
// If a function with the same name is already registered, it will be replaced.
func (r *Registry) Register(name string, factory FactoryFunc) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.registrations[name] = &Registration{
		Name:    name,
		Factory: factory,
	}
}

// RegisterWithMetadata registers a function factory with metadata.
//
// This version allows providing description, category, and tags
// without creating a function instance.
//
// Example:
//
//	registry.RegisterWithMetadata("text_transform", texttransform.New, Registration{
//	    Description: "Transform text",
//	    Category: "text",
//	    Tags: []string{"text", "transform"},
//	})
func (r *Registry) RegisterWithMetadata(name string, factory FactoryFunc, description string, category string, tags []string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.registrations[name] = &Registration{
		Name:        name,
		Factory:     factory,
		Description: description,
		Category:    category,
		Tags:        tags,
	}
}

// RegisterWithParams registers a function factory with declared config parameters.
//
// The config params declare what settings the function supports, enabling
// automatic config file generation and TUI editing.
//
// Example:
//
//	registry.RegisterWithParams("websearch", New, []registry.ConfigParam{
//	    {Key: "backend", Type: "string", Default: "auto", Description: "Search backend"},
//	})
func (r *Registry) RegisterWithParams(name string, factory FactoryFunc, params []ConfigParam) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.registrations[name] = &Registration{
		Name:         name,
		Factory:      factory,
		ConfigParams: params,
	}
}

// SetRequiresApproval marks a registered function as needing user confirmation
// before the agent executes it. This default is picked up by DefaultAppConfig
// when generating config.toml and can be overridden by the user there.
//
// Call this from the same init() that calls Register/RegisterWithParams.
//
// Example:
//
//	func init() {
//	    registry.Global().Register("write_file", NewWriteFile)
//	    registry.Global().SetRequiresApproval("write_file", true)
//	}
func (r *Registry) SetRequiresApproval(name string, required bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if reg, ok := r.registrations[name]; ok {
		reg.RequiresApproval = required
	}
}

// MustRegister registers a function factory and panics on error.
//
// This is useful for init() registration where errors are fatal.
//
// Example:
//
//	func init() {
//	    registry.Global().MustRegister("text_transform", texttransform.New)
//	}
func (r *Registry) MustRegister(name string, factory FactoryFunc) {
	if err := r.RegisterSafe(name, factory); err != nil {
		panic(fmt.Sprintf("failed to register function %s: %v", name, err))
	}
}

// RegisterSafe registers a function with error checking.
//
// Returns an error if the name is empty or factory is nil.
func (r *Registry) RegisterSafe(name string, factory FactoryFunc) error {
	if name == "" {
		return fmt.Errorf("function name cannot be empty")
	}
	if factory == nil {
		return fmt.Errorf("factory function cannot be nil")
	}

	r.Register(name, factory)
	return nil
}

// Get retrieves a function by name, creating a new instance.
//
// Returns nil if the function is not registered.
//
// Example:
//
//	fn := registry.Get("text_transform", config.GlobalConfig())
//	if fn != nil {
//	    result, err := fn.Execute(ctx, input)
//	}
func (r *Registry) Get(name string, cfg function.Config) function.AIFunction {
	r.mu.RLock()
	defer r.mu.RUnlock()

	reg, ok := r.registrations[name]
	if !ok {
		return nil
	}

	return reg.Factory(cfg)
}

// Has checks if a function is registered.
//
// Example:
//
//	if registry.Has("text_transform") {
//	    // Use function
//	}
func (r *Registry) Has(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	_, ok := r.registrations[name]
	return ok
}

// List returns all registered function names.
//
// The list is sorted alphabetically.
//
// Example:
//
//	names := registry.List()
//	for _, name := range names {
//	    fmt.Println(name)
//	}
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

// ListWithMetadata returns all registrations with their metadata.
//
// For functions registered without explicit metadata, this method
// creates a temporary instance to extract metadata from the function itself.
//
// The list is sorted by function name.
func (r *Registry) ListWithMetadata() []*Registration {
	r.mu.RLock()
	defer r.mu.RUnlock()

	regs := make([]*Registration, 0, len(r.registrations))
	for _, reg := range r.registrations {
		// If metadata not provided, get it from function
		if reg.Description == "" || reg.Category == "" {
			fn := reg.Factory(config.New()) // Use empty config for metadata extraction
			meta := fn.Metadata()

			enriched := &Registration{
				Name:        reg.Name,
				Factory:     reg.Factory,
				Description: meta.GetDescription(""),
				Category:    meta.Category,
				Tags:        meta.Tags,
			}
			regs = append(regs, enriched)
		} else {
			regs = append(regs, reg)
		}
	}

	// Sort by name
	sort.Slice(regs, func(i, j int) bool {
		return regs[i].Name < regs[j].Name
	})

	return regs
}

// GetByCategory returns all function names in the given category.
//
// Example:
//
//	textFunctions := registry.GetByCategory("text")
func (r *Registry) GetByCategory(category string) []string {
	regs := r.ListWithMetadata()
	names := make([]string, 0)

	for _, reg := range regs {
		if reg.Category == category {
			names = append(names, reg.Name)
		}
	}

	return names
}

// GetByTag returns all function names with the given tag.
//
// Example:
//
//	transformFunctions := registry.GetByTag("transform")
func (r *Registry) GetByTag(tag string) []string {
	regs := r.ListWithMetadata()
	names := make([]string, 0)

	for _, reg := range regs {
		for _, t := range reg.Tags {
			if t == tag {
				names = append(names, reg.Name)
				break
			}
		}
	}

	return names
}

// GetByTags returns all function names that have ALL of the given tags.
//
// Example:
//
//	textTransformFunctions := registry.GetByTags("text", "transform")
func (r *Registry) GetByTags(tags ...string) []string {
	if len(tags) == 0 {
		return r.List()
	}

	regs := r.ListWithMetadata()
	names := make([]string, 0)

	for _, reg := range regs {
		if hasAllTags(reg.Tags, tags) {
			names = append(names, reg.Name)
		}
	}

	return names
}

// GetCategories returns all unique categories.
//
// Example:
//
//	categories := registry.GetCategories()
//	for _, cat := range categories {
//	    fmt.Printf("Category: %s\n", cat)
//	}
func (r *Registry) GetCategories() []string {
	regs := r.ListWithMetadata()
	categorySet := make(map[string]bool)

	for _, reg := range regs {
		if reg.Category != "" {
			categorySet[reg.Category] = true
		}
	}

	categories := make([]string, 0, len(categorySet))
	for cat := range categorySet {
		categories = append(categories, cat)
	}

	sort.Strings(categories)
	return categories
}

// GetTags returns all unique tags across all functions.
//
// Example:
//
//	tags := registry.GetTags()
//	for _, tag := range tags {
//	    fmt.Printf("Tag: %s\n", tag)
//	}
func (r *Registry) GetTags() []string {
	regs := r.ListWithMetadata()
	tagSet := make(map[string]bool)

	for _, reg := range regs {
		for _, tag := range reg.Tags {
			tagSet[tag] = true
		}
	}

	tags := make([]string, 0, len(tagSet))
	for tag := range tagSet {
		tags = append(tags, tag)
	}

	sort.Strings(tags)
	return tags
}

// GetRegistration returns the registration for a function by name.
//
// Returns nil if the function is not registered.
func (r *Registry) GetRegistration(name string) *Registration {
	r.mu.RLock()
	defer r.mu.RUnlock()

	return r.registrations[name]
}

// Count returns the number of registered functions.
func (r *Registry) Count() int {
	r.mu.RLock()
	defer r.mu.RUnlock()

	return len(r.registrations)
}

// Clear removes all registered functions.
//
// This is primarily useful for testing.
func (r *Registry) Clear() {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.registrations = make(map[string]*Registration)
}

// hasAllTags checks if a slice contains all of the required tags.
func hasAllTags(available []string, required []string) bool {
	for _, req := range required {
		found := false
		for _, avail := range available {
			if avail == req {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// Global registry instance
var globalRegistry *Registry
var globalOnce sync.Once

// Global returns the global registry instance.
//
// The global registry is lazily initialized on first access.
//
// Example:
//
//	func init() {
//	    registry.Global().Register("my_function", NewMyFunction)
//	}
func Global() *Registry {
	globalOnce.Do(func() {
		globalRegistry = New()
	})
	return globalRegistry
}

// SetGlobal sets the global registry instance.
//
// This is useful for testing or when you want to use a custom registry.
func SetGlobal(r *Registry) {
	globalRegistry = r
}

// ResetGlobal resets the global registry.
//
// This is primarily useful for testing.
func ResetGlobal() {
	globalRegistry = New()
}
