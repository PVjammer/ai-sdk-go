package function

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockFunction is a simple function for testing
type mockFunctionForToolConfig struct {
	Base
}

func (f *mockFunctionForToolConfig) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	return map[string]string{"result": "mock"}, nil
}

func (f *mockFunctionForToolConfig) Metadata() *Metadata {
	return NewMetadata("MockFunction").
		WithName("mock").
		WithDescription("A mock function for testing").
		WithToolName("mock_tool").
		WithToolDescription("Mock tool for testing")
}

func (f *mockFunctionForToolConfig) InputSchema() *Schema {
	return &Schema{
		Type: "object",
		Properties: map[string]*Property{
			"input": {
				Type:        "string",
				Description: "Test input",
			},
		},
		Required: []string{"input"},
	}
}

func TestAsToolWithConfig(t *testing.T) {
	t.Run("adds llm config to tool definition", func(t *testing.T) {
		fn := &mockFunctionForToolConfig{}

		llmConfig := LLMConfig{
			Provider: "ollama",
			Endpoint: "http://localhost:11434",
			Model:    "llama3.2:latest",
			Options: map[string]interface{}{
				"temperature": 0.7,
			},
		}

		tool := AsToolWithConfig(fn, llmConfig)

		// Verify tool structure
		assert.Equal(t, "function", tool["type"])

		funcDef, ok := tool["function"].(map[string]interface{})
		require.True(t, ok, "function definition should be a map")

		// Verify base tool fields
		assert.Equal(t, "mock_tool", funcDef["name"])
		assert.Equal(t, "Mock tool for testing", funcDef["description"])

		// Verify LLM config was added
		llmConfigRaw, ok := funcDef["_llm_config"]
		require.True(t, ok, "_llm_config should be present")

		llmConfigMap, ok := llmConfigRaw.(map[string]interface{})
		require.True(t, ok, "_llm_config should be a map")

		assert.Equal(t, "ollama", llmConfigMap["provider"])
		assert.Equal(t, "http://localhost:11434", llmConfigMap["endpoint"])
		assert.Equal(t, "llama3.2:latest", llmConfigMap["model"])

		options, ok := llmConfigMap["options"].(map[string]interface{})
		require.True(t, ok, "options should be a map")
		assert.Equal(t, 0.7, options["temperature"])
	})

	t.Run("works with minimal config", func(t *testing.T) {
		fn := &mockFunctionForToolConfig{}

		llmConfig := LLMConfig{
			Provider: "ollama",
			Model:    "llama3.2:latest",
		}

		tool := AsToolWithConfig(fn, llmConfig)

		funcDef := tool["function"].(map[string]interface{})
		llmConfigMap := funcDef["_llm_config"].(map[string]interface{})

		assert.Equal(t, "ollama", llmConfigMap["provider"])
		assert.Equal(t, "", llmConfigMap["endpoint"])
		assert.Equal(t, "llama3.2:latest", llmConfigMap["model"])
	})
}

func TestExtractLLMConfig(t *testing.T) {
	t.Run("extracts config from tool definition", func(t *testing.T) {
		tool := map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name":        "test_tool",
				"description": "Test tool",
				"parameters": map[string]interface{}{
					"type": "object",
				},
				"_llm_config": map[string]interface{}{
					"provider": "ollama",
					"endpoint": "http://localhost:11434",
					"model":    "llama3.2:latest",
					"options": map[string]interface{}{
						"temperature": 0.5,
					},
				},
			},
		}

		config := ExtractLLMConfig(tool)

		require.NotNil(t, config)
		assert.Equal(t, "ollama", config.Provider)
		assert.Equal(t, "http://localhost:11434", config.Endpoint)
		assert.Equal(t, "llama3.2:latest", config.Model)
		assert.NotNil(t, config.Options)
		assert.Equal(t, 0.5, config.Options["temperature"])
	})

	t.Run("returns nil for tool without config", func(t *testing.T) {
		tool := map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name":        "test_tool",
				"description": "Test tool",
			},
		}

		config := ExtractLLMConfig(tool)
		assert.Nil(t, config)
	})

	t.Run("returns nil for invalid tool structure", func(t *testing.T) {
		tool := map[string]interface{}{
			"type": "function",
			// Missing function definition
		}

		config := ExtractLLMConfig(tool)
		assert.Nil(t, config)
	})

	t.Run("handles partial config", func(t *testing.T) {
		tool := map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name": "test_tool",
				"_llm_config": map[string]interface{}{
					"provider": "ollama",
					"model":    "llama3.2:latest",
					// Missing endpoint and options
				},
			},
		}

		config := ExtractLLMConfig(tool)

		require.NotNil(t, config)
		assert.Equal(t, "ollama", config.Provider)
		assert.Equal(t, "", config.Endpoint)
		assert.Equal(t, "llama3.2:latest", config.Model)
		assert.Nil(t, config.Options)
	})
}

func TestAsToolWithConfig_ExtractLLMConfig_Roundtrip(t *testing.T) {
	t.Run("roundtrip config", func(t *testing.T) {
		fn := &mockFunctionForToolConfig{}

		originalConfig := LLMConfig{
			Provider: "ollama",
			Endpoint: "http://localhost:11434",
			Model:    "llama3.2:latest",
			Options: map[string]interface{}{
				"temperature": 0.3,
				"max_tokens":  100,
			},
		}

		// Create tool with config
		tool := AsToolWithConfig(fn, originalConfig)

		// Extract config back
		extractedConfig := ExtractLLMConfig(tool)

		// Verify they match
		require.NotNil(t, extractedConfig)
		assert.Equal(t, originalConfig.Provider, extractedConfig.Provider)
		assert.Equal(t, originalConfig.Endpoint, extractedConfig.Endpoint)
		assert.Equal(t, originalConfig.Model, extractedConfig.Model)
		assert.Equal(t, originalConfig.Options["temperature"], extractedConfig.Options["temperature"])
		assert.Equal(t, originalConfig.Options["max_tokens"], extractedConfig.Options["max_tokens"])
	})
}

func TestCreateProviderFromConfig(t *testing.T) {
	t.Run("creates ollama provider", func(t *testing.T) {
		config := LLMConfig{
			Provider: "ollama",
			Endpoint: "http://localhost:11434",
			Model:    "llama3.2:latest",
		}

		provider, err := CreateProviderFromConfig(config)

		require.NoError(t, err)
		assert.NotNil(t, provider)
		assert.Equal(t, "ollama:llama3.2:latest", provider.Name())
	})

	t.Run("uses default endpoint for ollama", func(t *testing.T) {
		config := LLMConfig{
			Provider: "ollama",
			Model:    "llama3.2:latest",
			// No endpoint specified
		}

		provider, err := CreateProviderFromConfig(config)

		require.NoError(t, err)
		assert.NotNil(t, provider)
	})

	t.Run("requires model for ollama", func(t *testing.T) {
		config := LLMConfig{
			Provider: "ollama",
			Endpoint: "http://localhost:11434",
			// No model specified
		}

		provider, err := CreateProviderFromConfig(config)

		assert.Error(t, err)
		assert.Nil(t, provider)
		assert.Contains(t, err.Error(), "model is required")
	})

	t.Run("creates mock provider", func(t *testing.T) {
		config := LLMConfig{
			Provider: "mock",
			Model:    "mock-model",
			Options: map[string]interface{}{
				"responses": []string{"Response 1", "Response 2"},
			},
		}

		provider, err := CreateProviderFromConfig(config)

		require.NoError(t, err)
		assert.NotNil(t, provider)
		assert.Equal(t, "mock", provider.Name())
	})

	t.Run("creates mock provider with default response", func(t *testing.T) {
		config := LLMConfig{
			Provider: "mock",
			Model:    "mock-model",
		}

		provider, err := CreateProviderFromConfig(config)

		require.NoError(t, err)
		assert.NotNil(t, provider)
	})

	t.Run("returns error for unsupported provider", func(t *testing.T) {
		config := LLMConfig{
			Provider: "unsupported",
			Model:    "test-model",
		}

		provider, err := CreateProviderFromConfig(config)

		assert.Error(t, err)
		assert.Nil(t, provider)
		assert.Contains(t, err.Error(), "unsupported provider")
	})

	t.Run("defaults to ollama when provider is empty", func(t *testing.T) {
		config := LLMConfig{
			Provider: "",
			Model:    "llama3.2:latest",
		}

		provider, err := CreateProviderFromConfig(config)

		require.NoError(t, err)
		assert.NotNil(t, provider)
	})
}
