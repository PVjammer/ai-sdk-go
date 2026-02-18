package llm

import (
	"context"
	"fmt"
	"io"
	"strings"
)

// CollectResponse collects all chunks from a streaming channel into a single Response.
//
// This function buffers the complete response text and returns it as a Response object.
// Useful for converting streaming responses to the traditional synchronous format.
//
// The function respects context cancellation and will return immediately if the
// context is cancelled.
//
// Returns an error if:
//   - The context is cancelled
//   - A chunk contains an error
//   - The channel is closed without a final chunk (Done=true)
func CollectResponse(ctx context.Context, chunkCh <-chan StreamChunk) (*Response, error) {
	var textBuilder strings.Builder
	var lastChunk StreamChunk

	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()

		case chunk, ok := <-chunkCh:
			if !ok {
				// Channel closed without final chunk
				if lastChunk.Done {
					// We got a final chunk, return the response
					return &Response{
						Text:             textBuilder.String(),
						TokensUsed:       lastChunk.TokensUsed,
						PromptTokens:     lastChunk.PromptTokens,
						CompletionTokens: lastChunk.CompletionTokens,
						Model:            lastChunk.Model,
						FinishReason:     lastChunk.FinishReason,
						Provider:         "", // Will be set by caller if needed
					}, nil
				}
				return nil, fmt.Errorf("channel closed unexpectedly")
			}

			// Check for errors
			if chunk.Error != nil {
				return nil, chunk.Error
			}

			// Accumulate text
			if chunk.Text != "" {
				textBuilder.WriteString(chunk.Text)
			}

			// Store last chunk for metadata
			lastChunk = chunk

			// If this is the final chunk, return the response
			if chunk.Done {
				return &Response{
					Text:             textBuilder.String(),
					TokensUsed:       chunk.TokensUsed,
					PromptTokens:     chunk.PromptTokens,
					CompletionTokens: chunk.CompletionTokens,
					Model:            chunk.Model,
					FinishReason:     chunk.FinishReason,
					Provider:         "", // Will be set by caller if needed
				}, nil
			}
		}
	}
}

// StreamToWriter streams chunks to an io.Writer and returns the final Response.
//
// This function writes each chunk's text to the writer as it arrives, providing
// real-time output suitable for CLI applications. The final Response is returned
// when streaming completes.
//
// The function respects context cancellation and will return immediately if the
// context is cancelled.
//
// Returns an error if:
//   - The context is cancelled
//   - A chunk contains an error
//   - Writing to the writer fails
//   - The channel is closed without a final chunk (Done=true)
//
// Example:
//
//	chunkCh := provider.ainvoke(ctx, prompt, opts)
//	resp, err := llm.StreamToWriter(ctx, chunkCh, os.Stdout)
func StreamToWriter(ctx context.Context, chunkCh <-chan StreamChunk, w io.Writer) (*Response, error) {
	var textBuilder strings.Builder
	var lastChunk StreamChunk

	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()

		case chunk, ok := <-chunkCh:
			if !ok {
				// Channel closed without final chunk
				if lastChunk.Done {
					// We got a final chunk, return the response
					return &Response{
						Text:             textBuilder.String(),
						TokensUsed:       lastChunk.TokensUsed,
						PromptTokens:     lastChunk.PromptTokens,
						CompletionTokens: lastChunk.CompletionTokens,
						Model:            lastChunk.Model,
						FinishReason:     lastChunk.FinishReason,
						Provider:         "", // Will be set by caller if needed
					}, nil
				}
				return nil, fmt.Errorf("channel closed unexpectedly")
			}

			// Check for errors
			if chunk.Error != nil {
				return nil, chunk.Error
			}

			// Write text to writer
			if chunk.Text != "" {
				_, err := w.Write([]byte(chunk.Text))
				if err != nil {
					return nil, fmt.Errorf("failed to write to output: %w", err)
				}

				// Accumulate text for final response
				textBuilder.WriteString(chunk.Text)
			}

			// Store last chunk for metadata
			lastChunk = chunk

			// If this is the final chunk, return the response
			if chunk.Done {
				return &Response{
					Text:             textBuilder.String(),
					TokensUsed:       chunk.TokensUsed,
					PromptTokens:     chunk.PromptTokens,
					CompletionTokens: chunk.CompletionTokens,
					Model:            chunk.Model,
					FinishReason:     chunk.FinishReason,
					Provider:         "", // Will be set by caller if needed
				}, nil
			}
		}
	}
}
