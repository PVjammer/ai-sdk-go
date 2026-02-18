// Package config provides configuration management using Viper.
//
// The configuration system supports:
//   - YAML file loading from multiple locations
//   - Environment variable overrides with AI_FUNC_ prefix
//   - Environment variable substitution in YAML (${VAR})
//   - Dot notation access for nested values
//   - Type-safe getters
package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/viper"
)

// Config represents the configuration for AI functions.
//
// It wraps Viper to provide a clean interface for configuration access
// with support for YAML files, environment variables, and defaults.
type Config struct {
	v *viper.Viper
}

// New creates a new Config instance.
//
// This creates an empty config without loading any files. Use Load()
// or LoadDefaults() to load configuration from files and environment.
func New() *Config {
	return &Config{
		v: viper.New(),
	}
}

// LoadDefaults loads configuration from default locations.
//
// Default locations (in priority order):
//  1. ./.ai-functions.yaml (current directory)
//  2. ~/.ai-functions/config.yaml (user home directory)
//  3. Environment variables with AI_FUNC_ prefix
//
// Example:
//
//	cfg := config.New()
//	if err := cfg.LoadDefaults(); err != nil {
//	    log.Fatal(err)
//	}
func (c *Config) LoadDefaults() error {
	// Set up environment variable support
	c.v.SetEnvPrefix("AI_FUNC")
	c.v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	c.v.AutomaticEnv()

	// Get home directory
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("failed to get home directory: %w", err)
	}

	// Add search paths (in reverse priority order)
	configPaths := []string{
		filepath.Join(homeDir, ".ai-functions"),
		".", // Current directory
	}

	for _, path := range configPaths {
		c.v.AddConfigPath(path)
	}

	// Set config names to search for
	c.v.SetConfigName(".ai-functions")
	c.v.SetConfigType("yaml")

	// Try to read config file (it's okay if it doesn't exist)
	if err := c.v.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return fmt.Errorf("failed to read config: %w", err)
		}
		// Config file not found is okay, we'll use defaults
	}

	// Also try loading config.yaml from home directory
	c.v.SetConfigName("config")
	if err := c.v.MergeInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return fmt.Errorf("failed to merge config: %w", err)
		}
	}

	return nil
}

// LoadFile loads configuration from a specific file.
//
// The file is merged with any existing configuration. Later values
// override earlier ones.
//
// Example:
//
//	cfg := config.New()
//	if err := cfg.LoadFile("/path/to/config.yaml"); err != nil {
//	    log.Fatal(err)
//	}
func (c *Config) LoadFile(path string) error {
	c.v.SetConfigFile(path)

	if err := c.v.MergeInConfig(); err != nil {
		return fmt.Errorf("failed to load config file %s: %w", path, err)
	}

	return nil
}

// Get returns the value for the given key as interface{}.
//
// Returns nil if the key doesn't exist.
//
// Example:
//
//	value := cfg.Get("llm.anthropic.api_key")
func (c *Config) Get(key string) interface{} {
	return c.v.Get(key)
}

// GetString returns the value for the given key as a string.
//
// Returns empty string if the key doesn't exist.
//
// Example:
//
//	apiKey := cfg.GetString("llm.anthropic.api_key")
func (c *Config) GetString(key string) string {
	return c.v.GetString(key)
}

// GetInt returns the value for the given key as an int.
//
// Returns 0 if the key doesn't exist.
//
// Example:
//
//	timeout := cfg.GetInt("http.timeout")
func (c *Config) GetInt(key string) int {
	return c.v.GetInt(key)
}

// GetInt64 returns the value for the given key as an int64.
//
// Returns 0 if the key doesn't exist.
func (c *Config) GetInt64(key string) int64 {
	return c.v.GetInt64(key)
}

// GetFloat64 returns the value for the given key as a float64.
//
// Returns 0.0 if the key doesn't exist.
//
// Example:
//
//	threshold := cfg.GetFloat64("processing.threshold")
func (c *Config) GetFloat64(key string) float64 {
	return c.v.GetFloat64(key)
}

// GetBool returns the value for the given key as a bool.
//
// Returns false if the key doesn't exist.
//
// Example:
//
//	debug := cfg.GetBool("debug")
func (c *Config) GetBool(key string) bool {
	return c.v.GetBool(key)
}

// GetStringSlice returns the value for the given key as a []string.
//
// Returns empty slice if the key doesn't exist.
//
// Example:
//
//	tags := cfg.GetStringSlice("function.tags")
func (c *Config) GetStringSlice(key string) []string {
	return c.v.GetStringSlice(key)
}

// GetStringMap returns the value for the given key as a map[string]interface{}.
//
// Returns empty map if the key doesn't exist.
//
// Example:
//
//	settings := cfg.GetStringMap("function.settings")
func (c *Config) GetStringMap(key string) map[string]interface{} {
	return c.v.GetStringMap(key)
}

// IsSet checks if the given key is set in the configuration.
//
// Example:
//
//	if cfg.IsSet("llm.anthropic.api_key") {
//	    // Use API key
//	}
func (c *Config) IsSet(key string) bool {
	return c.v.IsSet(key)
}

// Set sets a value for the given key.
//
// This is useful for testing or setting runtime values.
//
// Example:
//
//	cfg.Set("debug", true)
func (c *Config) Set(key string, value interface{}) {
	c.v.Set(key, value)
}

// AllSettings returns all settings as a map.
//
// Example:
//
//	settings := cfg.AllSettings()
//	for key, value := range settings {
//	    fmt.Printf("%s: %v\n", key, value)
//	}
func (c *Config) AllSettings() map[string]interface{} {
	return c.v.AllSettings()
}

// AllKeys returns all keys in the configuration.
//
// Example:
//
//	keys := cfg.AllKeys()
//	for _, key := range keys {
//	    fmt.Printf("%s = %v\n", key, cfg.Get(key))
//	}
func (c *Config) AllKeys() []string {
	return c.v.AllKeys()
}

// Sub returns a new Config instance representing a subsection of the config.
//
// Example:
//
//	llmConfig := cfg.Sub("llm.anthropic")
//	apiKey := llmConfig.GetString("api_key")
func (c *Config) Sub(key string) *Config {
	sub := c.v.Sub(key)
	if sub == nil {
		return New() // Return empty config if key doesn't exist
	}
	return &Config{v: sub}
}

// ExpandEnv expands environment variables in a string.
//
// Supports both ${VAR} and $VAR syntax.
//
// Example:
//
//	// config.yaml: api_key: ${API_KEY}
//	apiKey := config.ExpandEnv(cfg.GetString("api_key"))
func ExpandEnv(s string) string {
	return os.ExpandEnv(s)
}

// GetStringWithEnv returns a string value with environment variable expansion.
//
// Example:
//
//	// config.yaml: api_key: ${API_KEY}
//	apiKey := cfg.GetStringWithEnv("api_key")
func (c *Config) GetStringWithEnv(key string) string {
	value := c.GetString(key)
	return ExpandEnv(value)
}

// ConfiguredFile returns the path to the config file being used, if any.
//
// Returns empty string if no config file was loaded.
func (c *Config) ConfiguredFile() string {
	return c.v.ConfigFileUsed()
}

// Global is the default global configuration instance.
var global *Config

// GlobalConfig returns the global configuration instance.
//
// The global config is lazily initialized on first access.
// Use SetGlobalConfig() to set a custom global config.
//
// Example:
//
//	cfg := config.GlobalConfig()
//	apiKey := cfg.GetString("llm.anthropic.api_key")
func GlobalConfig() *Config {
	if global == nil {
		global = New()
		_ = global.LoadDefaults() // Ignore errors for global config
	}
	return global
}

// SetGlobalConfig sets the global configuration instance.
//
// This is useful for testing or when you want to use a custom
// configuration instance as the global default.
//
// Example:
//
//	cfg := config.New()
//	cfg.Set("debug", true)
//	config.SetGlobalConfig(cfg)
func SetGlobalConfig(cfg *Config) {
	global = cfg
}

// ResetGlobalConfig resets the global configuration instance.
//
// This is primarily useful for testing.
func ResetGlobalConfig() {
	global = nil
}

// Note: Config implements the function.Config interface from pkg/function.
// This allows Config instances to be used as function configuration.
