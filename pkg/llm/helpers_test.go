package llm

import (
	"bytes"
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCollectResponse(t *testing.T) {
	t.Run("collect single chunk", func(t *testing.T) {
		ch := make(chan StreamChunk, 1)
		ch <- StreamChunk{
			Text:             "Complete response",
			Done:             true,
			TokensUsed:       10,
			PromptTokens:     5,
			CompletionTokens: 5,
			Model:            "test-model",
			FinishReason:     "stop",
		}
		close(ch)

		ctx := context.Background()
		resp, err := CollectResponse(ctx, ch)

		require.NoError(t, err)
		assert.Equal(t, "Complete response", resp.Text)
		assert.Equal(t, 10, resp.TokensUsed)
		assert.Equal(t, 5, resp.PromptTokens)
		assert.Equal(t, 5, resp.CompletionTokens)
		assert.Equal(t, "test-model", resp.Model)
		assert.Equal(t, "stop", resp.FinishReason)
	})

	t.Run("collect multiple chunks", func(t *testing.T) {
		ch := make(chan StreamChunk, 3)
		ch <- StreamChunk{Text: "Hello", Done: false}
		ch <- StreamChunk{Text: " world", Done: false}
		ch <- StreamChunk{
			Text:             "!",
			Done:             true,
			TokensUsed:       10,
			PromptTokens:     5,
			CompletionTokens: 5,
			Model:            "test-model",
			FinishReason:     "stop",
		}
		close(ch)

		ctx := context.Background()
		resp, err := CollectResponse(ctx, ch)

		require.NoError(t, err)
		assert.Equal(t, "Hello world!", resp.Text)
		assert.Equal(t, 10, resp.TokensUsed)
	})

	t.Run("context cancellation", func(t *testing.T) {
		ch := make(chan StreamChunk)

		// Create cancelled context
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		// Start collecting (should return immediately)
		_, err := CollectResponse(ctx, ch)

		assert.Error(t, err)
		assert.Equal(t, context.Canceled, err)
	})

	t.Run("chunk with error", func(t *testing.T) {
		ch := make(chan StreamChunk, 1)
		expectedErr := fmt.Errorf("test error")
		ch <- StreamChunk{
			Error: expectedErr,
			Done:  true,
		}
		close(ch)

		ctx := context.Background()
		_, err := CollectResponse(ctx, ch)

		assert.Error(t, err)
		assert.Equal(t, expectedErr, err)
	})

	t.Run("channel closed unexpectedly", func(t *testing.T) {
		ch := make(chan StreamChunk)
		close(ch) // Close without sending any chunks

		ctx := context.Background()
		_, err := CollectResponse(ctx, ch)

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "closed unexpectedly")
	})

	t.Run("context timeout during streaming", func(t *testing.T) {
		ch := make(chan StreamChunk)

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
		defer cancel()

		// Start collecting in goroutine
		go func() {
			time.Sleep(50 * time.Millisecond)
			ch <- StreamChunk{Text: "too late", Done: true}
			close(ch)
		}()

		_, err := CollectResponse(ctx, ch)

		assert.Error(t, err)
		assert.Equal(t, context.DeadlineExceeded, err)
	})
}

func TestStreamToWriter(t *testing.T) {
	t.Run("stream single chunk", func(t *testing.T) {
		ch := make(chan StreamChunk, 1)
		ch <- StreamChunk{
			Text:             "Complete response",
			Done:             true,
			TokensUsed:       10,
			PromptTokens:     5,
			CompletionTokens: 5,
			Model:            "test-model",
			FinishReason:     "stop",
		}
		close(ch)

		var buf bytes.Buffer
		ctx := context.Background()
		resp, err := StreamToWriter(ctx, ch, &buf)

		require.NoError(t, err)
		assert.Equal(t, "Complete response", buf.String())
		assert.Equal(t, "Complete response", resp.Text)
		assert.Equal(t, 10, resp.TokensUsed)
	})

	t.Run("stream multiple chunks", func(t *testing.T) {
		ch := make(chan StreamChunk, 3)
		ch <- StreamChunk{Text: "Hello", Done: false}
		ch <- StreamChunk{Text: " world", Done: false}
		ch <- StreamChunk{
			Text:             "!",
			Done:             true,
			TokensUsed:       10,
			PromptTokens:     5,
			CompletionTokens: 5,
			Model:            "test-model",
			FinishReason:     "stop",
		}
		close(ch)

		var buf bytes.Buffer
		ctx := context.Background()
		resp, err := StreamToWriter(ctx, ch, &buf)

		require.NoError(t, err)
		assert.Equal(t, "Hello world!", buf.String())
		assert.Equal(t, "Hello world!", resp.Text)
		assert.Equal(t, 10, resp.TokensUsed)
	})

	t.Run("context cancellation", func(t *testing.T) {
		ch := make(chan StreamChunk)

		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		var buf bytes.Buffer
		_, err := StreamToWriter(ctx, ch, &buf)

		assert.Error(t, err)
		assert.Equal(t, context.Canceled, err)
	})

	t.Run("chunk with error", func(t *testing.T) {
		ch := make(chan StreamChunk, 1)
		expectedErr := fmt.Errorf("test error")
		ch <- StreamChunk{
			Error: expectedErr,
			Done:  true,
		}
		close(ch)

		var buf bytes.Buffer
		ctx := context.Background()
		_, err := StreamToWriter(ctx, ch, &buf)

		assert.Error(t, err)
		assert.Equal(t, expectedErr, err)
	})

	t.Run("write error", func(t *testing.T) {
		ch := make(chan StreamChunk, 1)
		ch <- StreamChunk{
			Text: "test",
			Done: true,
		}
		close(ch)

		// Use a writer that always fails
		failWriter := &failingWriter{}
		ctx := context.Background()
		_, err := StreamToWriter(ctx, ch, failWriter)

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "failed to write")
	})
}

func TestCollectResponse_WithMockProvider(t *testing.T) {
	t.Run("collect from mock provider non-streaming", func(t *testing.T) {
		mock := NewMockProvider("Test response")

		ctx := context.Background()
		opts := DefaultOptions()
		opts.Stream = false

		chunkCh := mock.Ainvoke(ctx, "test prompt", opts)
		resp, err := CollectResponse(ctx, chunkCh)

		require.NoError(t, err)
		assert.Equal(t, "Test response", resp.Text)
		assert.Equal(t, "mock", resp.Model)
		assert.Equal(t, "stop", resp.FinishReason)
		assert.Greater(t, resp.TokensUsed, 0)
	})

	t.Run("collect from mock provider streaming", func(t *testing.T) {
		mock := NewMockProvider("Hello world from streaming")

		ctx := context.Background()
		opts := DefaultOptions()
		opts.Stream = true

		chunkCh := mock.Ainvoke(ctx, "test prompt", opts)
		resp, err := CollectResponse(ctx, chunkCh)

		require.NoError(t, err)
		assert.Equal(t, "Hello world from streaming", resp.Text)
		assert.Equal(t, "mock", resp.Model)
		assert.Greater(t, resp.TokensUsed, 0)
	})
}

func TestStreamToWriter_WithMockProvider(t *testing.T) {
	t.Run("stream from mock provider", func(t *testing.T) {
		mock := NewMockProvider("Streaming test")

		ctx := context.Background()
		opts := DefaultOptions()
		opts.Stream = true

		var buf bytes.Buffer
		chunkCh := mock.Ainvoke(ctx, "test prompt", opts)
		resp, err := StreamToWriter(ctx, chunkCh, &buf)

		require.NoError(t, err)
		assert.Equal(t, "Streaming test", buf.String())
		assert.Equal(t, "Streaming test", resp.Text)
		assert.Greater(t, resp.TokensUsed, 0)
	})
}

// failingWriter always returns an error on Write
type failingWriter struct{}

func (w *failingWriter) Write(p []byte) (n int, err error) {
	return 0, fmt.Errorf("write failed")
}
