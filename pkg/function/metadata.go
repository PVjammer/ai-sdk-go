package function

// Metadata contains declarative metadata for an AI function.
//
// The metadata system supports a three-level fallback chain for names and descriptions:
//  1. Format-specific overrides (ToolName, CLIName, ToolDescription, CLIDescription)
//  2. Shared overrides (Name, Description)
//  3. Derived values (struct type name, default description)
//
// This allows functions to have different names and descriptions in different contexts
// (e.g., "web_search" for LLM tools vs "search" for CLI).
type Metadata struct {
	// Format-specific overrides for LLM tool context
	ToolName        string
	ToolDescription string

	// Format-specific overrides for CLI context
	CLIName         string
	CLIDescription  string
	CLIHidden       bool // Hide from CLI entirely

	// Shared overrides (used when format-specific overrides are not set)
	Name        string
	Description string

	// Organization metadata
	Category string   // Category for filtering (e.g., "text-processing", "web-data")
	Tags     []string // Tags for filtering (e.g., ["search", "web", "realtime"])

	// Examples for documentation and LLM context
	Examples []Example

	// Internal field for derived name (set by NewMetadata)
	derivedName string
}

// Example represents a usage example for the function.
//
// Examples are used for:
//  1. Documentation (shown in CLI help)
//  2. LLM context (included in tool definitions)
//  3. Testing (can be used to generate tests)
type Example struct {
	// Input shows example input values
	Input map[string]interface{} `json:"input"`

	// Output shows the expected output
	Output interface{} `json:"output,omitempty"`

	// Description explains what this example demonstrates
	Description string `json:"description,omitempty"`
}

// NewMetadata creates metadata with the given derived name.
//
// The derived name is typically the struct type name (e.g., "TextTransformFunction")
// and is used as a fallback when no other name is specified.
//
// Example:
//
//	meta := function.NewMetadata("TextTransformFunction").
//	    WithToolName("text_transform").
//	    WithCLIName("transform").
//	    WithCategory("text-processing").
//	    WithTags("text", "string", "utility")
func NewMetadata(derivedName string) *Metadata {
	return &Metadata{
		derivedName: derivedName,
	}
}

// GetName returns the function name for the given context.
//
// Implements the three-level fallback chain:
//  1. Format-specific: ToolName (context="tool") or CLIName (context="cli")
//  2. Shared: Name
//  3. Derived: derivedName
//
// Context should be "tool", "cli", or "" for general/shared name.
func (m *Metadata) GetName(context string) string {
	// Level 1: Format-specific override
	if context == "tool" && m.ToolName != "" {
		return m.ToolName
	}
	if context == "cli" && m.CLIName != "" {
		return m.CLIName
	}

	// Level 2: Shared override
	if m.Name != "" {
		return m.Name
	}

	// Level 3: Derived value
	return m.derivedName
}

// GetDescription returns the function description for the given context.
//
// Implements the three-level fallback chain:
//  1. Format-specific: ToolDescription (context="tool") or CLIDescription (context="cli")
//  2. Shared: Description
//  3. Derived: derivedName + " function"
//
// Context should be "tool", "cli", or "" for general/shared description.
func (m *Metadata) GetDescription(context string) string {
	// Level 1: Format-specific override
	if context == "tool" && m.ToolDescription != "" {
		return m.ToolDescription
	}
	if context == "cli" && m.CLIDescription != "" {
		return m.CLIDescription
	}

	// Level 2: Shared override
	if m.Description != "" {
		return m.Description
	}

	// Level 3: Derived value
	return m.derivedName + " function"
}

// Builder methods for fluent API

// WithName sets the shared name (used by both tool and CLI contexts).
func (m *Metadata) WithName(name string) *Metadata {
	m.Name = name
	return m
}

// WithDescription sets the shared description.
func (m *Metadata) WithDescription(description string) *Metadata {
	m.Description = description
	return m
}

// WithToolName sets the LLM tool-specific name.
func (m *Metadata) WithToolName(name string) *Metadata {
	m.ToolName = name
	return m
}

// WithToolDescription sets the LLM tool-specific description.
func (m *Metadata) WithToolDescription(description string) *Metadata {
	m.ToolDescription = description
	return m
}

// WithCLIName sets the CLI-specific name.
func (m *Metadata) WithCLIName(name string) *Metadata {
	m.CLIName = name
	return m
}

// WithCLIDescription sets the CLI-specific description.
func (m *Metadata) WithCLIDescription(description string) *Metadata {
	m.CLIDescription = description
	return m
}

// HideFromCLI marks the function as hidden from CLI.
func (m *Metadata) HideFromCLI() *Metadata {
	m.CLIHidden = true
	return m
}

// WithCategory sets the function category.
func (m *Metadata) WithCategory(category string) *Metadata {
	m.Category = category
	return m
}

// WithTags sets the function tags.
func (m *Metadata) WithTags(tags ...string) *Metadata {
	m.Tags = tags
	return m
}

// WithExamples sets the function examples.
func (m *Metadata) WithExamples(examples ...Example) *Metadata {
	m.Examples = examples
	return m
}

// AddExample adds a single example to the function.
func (m *Metadata) AddExample(example Example) *Metadata {
	m.Examples = append(m.Examples, example)
	return m
}

// HasTag checks if the metadata contains the given tag.
func (m *Metadata) HasTag(tag string) bool {
	for _, t := range m.Tags {
		if t == tag {
			return true
		}
	}
	return false
}

// HasAnyTag checks if the metadata contains any of the given tags.
func (m *Metadata) HasAnyTag(tags ...string) bool {
	for _, tag := range tags {
		if m.HasTag(tag) {
			return true
		}
	}
	return false
}

// HasAllTags checks if the metadata contains all of the given tags.
func (m *Metadata) HasAllTags(tags ...string) bool {
	for _, tag := range tags {
		if !m.HasTag(tag) {
			return false
		}
	}
	return true
}
