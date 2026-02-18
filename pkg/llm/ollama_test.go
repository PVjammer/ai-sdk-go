package llm

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewOllamaProvider(t *testing.T) {
	tests := []struct {
		name      string
		endpoint  string
		model     string
		wantError bool
	}{
		{
			name:      "valid provider",
			endpoint:  "http://localhost:11434",
			model:     "llama3.2:latest",
			wantError: false,
		},
		{
			name:      "empty endpoint uses default",
			endpoint:  "",
			model:     "llama3.2:latest",
			wantError: false,
		},
		{
			name:      "missing model",
			endpoint:  "http://localhost:11434",
			model:     "",
			wantError: true,
		},
		{
			name:      "invalid endpoint",
			endpoint:  "://invalid",
			model:     "llama3.2:latest",
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider, err := NewOllamaProvider(tt.endpoint, tt.model)

			if tt.wantError {
				assert.Error(t, err)
				assert.Nil(t, provider)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, provider)
				if tt.endpoint == "" {
					assert.Equal(t, "http://localhost:11434", provider.endpoint)
				} else {
					assert.Equal(t, tt.endpoint, provider.endpoint)
				}
				assert.Equal(t, tt.model, provider.model)
			}
		})
	}
}

func TestOllamaProvider_Name(t *testing.T) {
	provider, err := NewOllamaProvider("http://localhost:11434", "llama3.2:latest")
	require.NoError(t, err)

	assert.Equal(t, "ollama:llama3.2:latest", provider.Name())
}

func TestOllamaProvider_MaxTokens(t *testing.T) {
	provider, err := NewOllamaProvider("http://localhost:11434", "llama3.2:latest")
	require.NoError(t, err)

	assert.Equal(t, 8192, provider.MaxTokens())
}

func TestOllamaProvider_Complete_Integration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	provider, err := NewOllamaProvider("http://localhost:11434", "llama3.2:latest")
	require.NoError(t, err)

	// Check if Ollama is available
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err = provider.Ping(ctx)
	if err != nil {
		t.Skipf("Ollama not available: %v", err)
	}

	// Test completion
	ctx = context.Background()
	opts := DefaultOptions().
		WithMaxTokens(50).
		WithTemperature(0.1).
		WithSystemPrompt("You are a helpful assistant. Be concise.")

	resp, err := provider.Complete(ctx, "What is 2+2? Answer in one sentence.", opts)
	require.NoError(t, err)

	assert.NotEmpty(t, resp.Text)
	assert.Greater(t, resp.TokensUsed, 0)
	assert.Greater(t, resp.PromptTokens, 0)
	assert.Greater(t, resp.CompletionTokens, 0)
	assert.Contains(t, resp.Model, "llama3.2")
	assert.Equal(t, "ollama:llama3.2:latest", resp.Provider)

	t.Logf("Response: %s", resp.Text)
	t.Logf("Tokens used: %d (prompt: %d, completion: %d)",
		resp.TokensUsed, resp.PromptTokens, resp.CompletionTokens)
}

func TestOllamaProvider_Complete_WithStopWords(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	provider, err := NewOllamaProvider("http://localhost:11434", "llama3.2:latest")
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err = provider.Ping(ctx)
	if err != nil {
		t.Skipf("Ollama not available: %v", err)
	}

	ctx = context.Background()
	opts := DefaultOptions().
		WithMaxTokens(100).
		WithTemperature(0.1).
		WithStopWords(".", "!")

	resp, err := provider.Complete(ctx, "Count from 1 to 10", opts)
	require.NoError(t, err)

	assert.NotEmpty(t, resp.Text)
	t.Logf("Response with stop words: %s", resp.Text)
}

func TestOllamaProvider_Complete_ContextCancellation(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	provider, err := NewOllamaProvider("http://localhost:11434", "llama3.2:latest")
	require.NoError(t, err)

	// Check if Ollama is available first
	pingCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	err = provider.Ping(pingCtx)
	cancel()
	if err != nil {
		t.Skipf("Ollama not available: %v", err)
	}

	// Test with immediate cancellation
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	opts := DefaultOptions()
	_, err = provider.Complete(ctx, "test prompt", opts)

	assert.Error(t, err)
	assert.Equal(t, context.Canceled, err)
}

func TestOllamaProvider_Ping(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	provider, err := NewOllamaProvider("http://localhost:11434", "llama3.2:latest")
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err = provider.Ping(ctx)
	if err != nil {
		t.Skipf("Ollama not available: %v", err)
	}

	// If we get here, Ollama is available
	assert.NoError(t, err)
}

func TestOllamaProvider_ListModels(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	provider, err := NewOllamaProvider("http://localhost:11434", "llama3.2:latest")
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err = provider.Ping(ctx)
	if err != nil {
		t.Skipf("Ollama not available: %v", err)
	}

	models, err := provider.ListModels(context.Background())
	require.NoError(t, err)

	assert.NotEmpty(t, models)
	t.Logf("Available models: %v", models)
}

func TestOllamaProvider_Complete_Unavailable(t *testing.T) {
	// Use an endpoint that definitely won't work
	provider, err := NewOllamaProvider("http://localhost:99999", "llama3.2:latest")
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	opts := DefaultOptions()
	_, err = provider.Complete(ctx, "test", opts)

	assert.Error(t, err)
	// Should get either ErrProviderUnavailable or context timeout
}

func TestIsConnectionError(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		expected bool
	}{
		{
			name:     "nil error",
			err:      nil,
			expected: false,
		},
		{
			name:     "connection refused",
			err:      assert.AnError,
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isConnectionError(tt.err)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestMapFinishReason(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"stop", "stop"},
		{"length", "length"},
		{"", "stop"},
		{"unknown", "unknown"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := mapFinishReason(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}
