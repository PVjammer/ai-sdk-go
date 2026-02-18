package function

import (
	"fmt"

	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// LLMConfig contains configuration for an LLM provider.
//
// This configuration can be embedded in tool definitions to specify
// which LLM should be used when the tool is invoked by an agent.
type LLMConfig struct {
	// Provider specifies the LLM provider (e.g., "ollama", "anthropic", "openai")
	Provider string `json:"provider"`

	// Endpoint is the API endpoint for the provider
	Endpoint string `json:"endpoint,omitempty"`

	// Model is the model name/identifier (e.g., "llama3.2:latest", "claude-3-5-haiku")
	Model string `json:"model"`

	// Options contains provider-specific options (temperature, max_tokens, etc.)
	Options map[string]interface{} `json:"options,omitempty"`
}

// AsToolWithConfig wraps an AIFunction's tool definition with embedded LLM configuration.
//
// This allows tool definitions to specify which LLM should be used when the tool
// is invoked, which is particularly useful for agent systems that need to configure
// LLMs on a per-tool basis.
//
// The LLM config is added to the tool definition under the "_llm_config" key.
//
// Example:
//
//	llmConfig := LLMConfig{
//	    Provider: "ollama",
//	    Endpoint: "http://localhost:11434",
//	    Model: "llama3.2:latest",
//	}
//	tool := AsToolWithConfig(fn, llmConfig)
//
// The resulting tool definition will include:
//
//	{
//	  "type": "function",
//	  "function": {
//	    "name": "...",
//	    "description": "...",
//	    "parameters": {...},
//	    "_llm_config": {
//	      "provider": "ollama",
//	      "endpoint": "http://localhost:11434",
//	      "model": "llama3.2:latest"
//	    }
//	  }
//	}
func AsToolWithConfig(fn AIFunction, llmConfig LLMConfig) map[string]interface{} {
	// Get base tool definition
	tool := AsTool(fn)

	// Add LLM config to function definition
	if funcDef, ok := tool["function"].(map[string]interface{}); ok {
		funcDef["_llm_config"] = map[string]interface{}{
			"provider": llmConfig.Provider,
			"endpoint": llmConfig.Endpoint,
			"model":    llmConfig.Model,
			"options":  llmConfig.Options,
		}
	}

	return tool
}

// ExtractLLMConfig extracts LLM configuration from a tool definition.
//
// Returns nil if the tool definition doesn't contain LLM configuration.
//
// This is the inverse of AsToolWithConfig - it extracts the "_llm_config"
// from a tool definition and returns it as an LLMConfig struct.
//
// Example:
//
//	config := ExtractLLMConfig(tool)
//	if config != nil {
//	    provider, err := CreateProviderFromConfig(*config)
//	    // ...
//	}
func ExtractLLMConfig(tool map[string]interface{}) *LLMConfig {
	// Navigate to function._llm_config
	funcDef, ok := tool["function"].(map[string]interface{})
	if !ok {
		return nil
	}

	llmConfigRaw, ok := funcDef["_llm_config"]
	if !ok {
		return nil
	}

	llmConfigMap, ok := llmConfigRaw.(map[string]interface{})
	if !ok {
		return nil
	}

	// Extract fields
	config := &LLMConfig{}

	if provider, ok := llmConfigMap["provider"].(string); ok {
		config.Provider = provider
	}

	if endpoint, ok := llmConfigMap["endpoint"].(string); ok {
		config.Endpoint = endpoint
	}

	if model, ok := llmConfigMap["model"].(string); ok {
		config.Model = model
	}

	if options, ok := llmConfigMap["options"].(map[string]interface{}); ok {
		config.Options = options
	}

	return config
}

// CreateProviderFromConfig creates an LLM provider from configuration.
//
// Currently supports:
//   - "ollama": Local Ollama provider
//   - "mock": Mock provider for testing
//
// Returns an error if the provider type is unsupported or configuration is invalid.
//
// Example:
//
//	config := LLMConfig{
//	    Provider: "ollama",
//	    Endpoint: "http://localhost:11434",
//	    Model: "llama3.2:latest",
//	}
//	provider, err := CreateProviderFromConfig(config)
//	if err != nil {
//	    // handle error
//	}
func CreateProviderFromConfig(cfg LLMConfig) (llm.Provider, error) {
	switch cfg.Provider {
	case "ollama", "":
		// Default to Ollama
		endpoint := cfg.Endpoint
		if endpoint == "" {
			endpoint = "http://localhost:11434"
		}

		if cfg.Model == "" {
			return nil, fmt.Errorf("model is required for ollama provider")
		}

		provider, err := llm.NewOllamaProvider(endpoint, cfg.Model)
		if err != nil {
			return nil, fmt.Errorf("failed to create ollama provider: %w", err)
		}

		return provider, nil

	case "mock":
		// Mock provider for testing
		// Use a default response if no options provided
		responses := []string{"Mock response"}
		if cfg.Options != nil {
			if resp, ok := cfg.Options["responses"].([]string); ok {
				responses = resp
			}
		}

		return llm.NewMockProvider(responses...), nil

	default:
		return nil, fmt.Errorf("unsupported provider: %s", cfg.Provider)
	}
}
