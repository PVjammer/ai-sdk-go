package llm

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDefaultOptions(t *testing.T) {
	opts := DefaultOptions()

	assert.Equal(t, 2048, opts.MaxTokens)
	assert.Equal(t, 0.7, opts.Temperature)
	assert.Empty(t, opts.StopWords)
	assert.Equal(t, 0.8, opts.TopP)
	assert.Equal(t, 20, opts.TopK)
	assert.Equal(t, 0.0, opts.MinP)
}

func TestCompletionOptions_Builders(t *testing.T) {
	opts := DefaultOptions().
		WithMaxTokens(1000).
		WithTemperature(0.5).
		WithSystemPrompt("You are a helpful assistant.").
		WithStopWords("stop1", "stop2")

	assert.Equal(t, 1000, opts.MaxTokens)
	assert.Equal(t, 0.5, opts.Temperature)
	assert.Equal(t, "You are a helpful assistant.", opts.SystemPrompt)
	assert.Equal(t, []string{"stop1", "stop2"}, opts.StopWords)
}

func TestMockProvider_Complete(t *testing.T) {
	mock := NewMockProvider(
		"First response",
		"Second response",
		"Third response",
	)

	ctx := context.Background()
	opts := DefaultOptions()

	// First call
	resp1, err := mock.Complete(ctx, "prompt 1", opts)
	require.NoError(t, err)
	assert.Equal(t, "First response", resp1.Text)
	assert.Equal(t, "mock", resp1.Model)
	assert.Equal(t, "mock", resp1.Provider)
	assert.Greater(t, resp1.TokensUsed, 0)

	// Second call
	resp2, err := mock.Complete(ctx, "prompt 2", opts)
	require.NoError(t, err)
	assert.Equal(t, "Second response", resp2.Text)

	// Third call
	resp3, err := mock.Complete(ctx, "prompt 3", opts)
	require.NoError(t, err)
	assert.Equal(t, "Third response", resp3.Text)

	// Fourth call should fail (no more responses)
	_, err = mock.Complete(ctx, "prompt 4", opts)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no more responses")
}

func TestMockProvider_CallTracking(t *testing.T) {
	mock := NewMockProvider("response")

	ctx := context.Background()
	opts := DefaultOptions().
		WithSystemPrompt("system prompt").
		WithTemperature(0.3)

	// Make call
	_, err := mock.Complete(ctx, "test prompt", opts)
	require.NoError(t, err)

	// Verify tracking
	assert.Equal(t, 1, mock.CallCount())
	assert.Equal(t, "test prompt", mock.LastPrompt())
	assert.NotNil(t, mock.LastOptions())
	assert.Equal(t, 0.3, mock.LastOptions().Temperature)
}

func TestMockProvider_VerifyMethods(t *testing.T) {
	mock := NewMockProvider("response")

	ctx := context.Background()
	opts := DefaultOptions().
		WithSystemPrompt("You are helpful").
		WithTemperature(0.5)

	_, err := mock.Complete(ctx, "Hello world", opts)
	require.NoError(t, err)

	// Verify methods
	assert.True(t, mock.VerifyPromptContains("Hello"))
	assert.True(t, mock.VerifyPromptContains("world"))
	assert.False(t, mock.VerifyPromptContains("missing"))

	assert.True(t, mock.VerifySystemPrompt("You are helpful"))
	assert.False(t, mock.VerifySystemPrompt("different"))

	assert.True(t, mock.VerifyTemperature(0.5))
	assert.False(t, mock.VerifyTemperature(0.3))
}

func TestMockProvider_Reset(t *testing.T) {
	mock := NewMockProvider("response1", "response2")

	ctx := context.Background()
	opts := DefaultOptions()

	// Make calls
	_, err := mock.Complete(ctx, "prompt1", opts)
	require.NoError(t, err)
	assert.Equal(t, 1, mock.CallCount())

	// Reset
	mock.Reset()

	// Verify reset
	assert.Equal(t, 0, mock.CallCount())
	assert.Empty(t, mock.LastPrompt())
	assert.Nil(t, mock.LastOptions())

	// Can make calls again
	_, err = mock.Complete(ctx, "prompt2", opts)
	require.NoError(t, err)
	assert.Equal(t, 1, mock.CallCount())
}

func TestMockProvider_WithFailure(t *testing.T) {
	// Fail after 2 successful calls
	mock := NewMockProviderWithFailure(2, "response1", "response2", "response3")

	ctx := context.Background()
	opts := DefaultOptions()

	// First two calls succeed
	_, err := mock.Complete(ctx, "prompt1", opts)
	require.NoError(t, err)

	_, err = mock.Complete(ctx, "prompt2", opts)
	require.NoError(t, err)

	// Third call fails
	_, err = mock.Complete(ctx, "prompt3", opts)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "simulated failure")
}

func TestMockProvider_ContextCancellation(t *testing.T) {
	mock := NewMockProvider("response")

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	opts := DefaultOptions()

	_, err := mock.Complete(ctx, "prompt", opts)
	assert.Error(t, err)
	assert.Equal(t, context.Canceled, err)
}

func TestMockProvider_ContextTimeout(t *testing.T) {
	mock := NewMockProvider("response")

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
	defer cancel()

	time.Sleep(10 * time.Millisecond) // Ensure timeout

	opts := DefaultOptions()

	_, err := mock.Complete(ctx, "prompt", opts)
	assert.Error(t, err)
	assert.Equal(t, context.DeadlineExceeded, err)
}

func TestMockProvider_Name(t *testing.T) {
	mock := NewMockProvider("response")
	assert.Equal(t, "mock", mock.Name())
}

func TestMockProvider_MaxTokens(t *testing.T) {
	mock := NewMockProvider("response")
	assert.Equal(t, 4096, mock.MaxTokens())
}

func TestResponse_TokenEstimation(t *testing.T) {
	mock := NewMockProvider("This is a test response with multiple words.")

	ctx := context.Background()
	opts := DefaultOptions()

	prompt := "This is a test prompt with several words."
	resp, err := mock.Complete(ctx, prompt, opts)
	require.NoError(t, err)

	// Verify token estimation (rough: 1 token ≈ 4 chars)
	expectedPromptTokens := len(prompt) / 4
	expectedCompletionTokens := len(resp.Text) / 4

	assert.Equal(t, expectedPromptTokens, resp.PromptTokens)
	assert.Equal(t, expectedCompletionTokens, resp.CompletionTokens)
	assert.Equal(t, expectedPromptTokens+expectedCompletionTokens, resp.TokensUsed)
}

func TestProvider_Interface(t *testing.T) {
	// Verify MockProvider implements Provider interface
	var _ Provider = (*MockProvider)(nil)

	// Verify OllamaProvider implements Provider interface
	var _ Provider = (*OllamaProvider)(nil)
}
