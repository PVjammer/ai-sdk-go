package function

// Config provides access to function configuration.
//
// Configuration is loaded from multiple sources with priority:
//  1. CLI arguments (highest priority)
//  2. Environment variables
//  3. YAML configuration files
//  4. Defaults (lowest priority)
//
// The actual implementation is in pkg/config. This interface allows
// functions to access configuration without depending on the concrete type.
type Config interface {
	// Get retrieves a configuration value using dot notation.
	// Returns nil if the key doesn't exist.
	//
	// Example:
	//   model := config.Get("llm.anthropic.model")
	Get(key string) interface{}

	// GetString retrieves a string configuration value.
	// Returns empty string if the key doesn't exist or value isn't a string.
	GetString(key string) string

	// GetInt retrieves an integer configuration value.
	// Returns 0 if the key doesn't exist or value isn't an integer.
	GetInt(key string) int

	// GetBool retrieves a boolean configuration value.
	// Returns false if the key doesn't exist or value isn't a boolean.
	GetBool(key string) bool

	// GetStringSlice retrieves a string slice configuration value.
	// Returns nil if the key doesn't exist or value isn't a string slice.
	GetStringSlice(key string) []string

	// Set sets a configuration value using dot notation.
	// This allows runtime overrides of configuration.
	Set(key string, value interface{})

	// IsSet checks if a configuration key exists.
	IsSet(key string) bool
}

// NilConfig is a Config implementation that returns zero values for all methods.
//
// This is used as a fallback when no configuration is provided.
type NilConfig struct{}

// Get always returns nil.
func (n NilConfig) Get(key string) interface{} {
	return nil
}

// GetString always returns empty string.
func (n NilConfig) GetString(key string) string {
	return ""
}

// GetInt always returns 0.
func (n NilConfig) GetInt(key string) int {
	return 0
}

// GetBool always returns false.
func (n NilConfig) GetBool(key string) bool {
	return false
}

// GetStringSlice always returns nil.
func (n NilConfig) GetStringSlice(key string) []string {
	return nil
}

// Set does nothing.
func (n NilConfig) Set(key string, value interface{}) {
	// No-op
}

// IsSet always returns false.
func (n NilConfig) IsSet(key string) bool {
	return false
}
