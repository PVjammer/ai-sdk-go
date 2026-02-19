package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/ollama/ollama/api"
)

// Compile-time interface checks
var _ Provider = (*OllamaProvider)(nil)
var _ ConversationProvider = (*OllamaProvider)(nil)
var _ ToolCallingProvider = (*OllamaProvider)(nil)

// OllamaProvider implements the Provider interface for Ollama.
//
// Ollama is a local LLM runtime that supports various models including
// Llama, Mistral, DeepSeek, and more. It provides an OpenAI-compatible
// API for easy integration.
//
// See: https://ollama.ai
type OllamaProvider struct {
	client   *api.Client
	model    string
	endpoint string
}

// NewOllamaProvider creates a new Ollama provider.
//
// Parameters:
//   - endpoint: Ollama API endpoint (e.g., "http://localhost:11434")
//   - model: Model name (e.g., "llama3.2:latest", "mistral:7b")
//
// Returns an error if the endpoint is invalid or the client cannot be created.
func NewOllamaProvider(endpoint, model string) (*OllamaProvider, error) {
	if endpoint == "" {
		endpoint = "http://localhost:11434"
	}

	if model == "" {
		return nil, fmt.Errorf("model name is required")
	}

	// Parse and validate endpoint
	parsedURL, err := url.Parse(endpoint)
	if err != nil {
		return nil, fmt.Errorf("invalid endpoint URL: %w", err)
	}

	// Create HTTP client
	httpClient := &http.Client{}

	// Create Ollama client
	client := api.NewClient(parsedURL, httpClient)

	return &OllamaProvider{
		client:   client,
		model:    model,
		endpoint: endpoint,
	}, nil
}

// Complete generates a text completion using Ollama.
//
// DEPRECATED: Use ainvoke() for new code.
func (p *OllamaProvider) Complete(ctx context.Context, prompt string, opts *CompletionOptions) (*Response, error) {
	if opts == nil {
		opts = DefaultOptions()
	}

	// Build messages
	messages := []api.Message{}

	// Add system prompt if provided
	if opts.SystemPrompt != "" {
		messages = append(messages, api.Message{
			Role:    "system",
			Content: opts.SystemPrompt,
		})
	}

	// Add user prompt
	messages = append(messages, api.Message{
		Role:    "user",
		Content: prompt,
	})

	// Build request
	req := &api.ChatRequest{
		Model:    p.model,
		Messages: messages,
		Stream:   new(bool), // Disable streaming
		Options: map[string]interface{}{
			"temperature": opts.Temperature,
		},
	}

	// Add max tokens if specified
	if opts.MaxTokens > 0 {
		req.Options["num_predict"] = opts.MaxTokens
	}

	// Add stop words if specified
	if len(opts.StopWords) > 0 {
		req.Options["stop"] = opts.StopWords
	}

	// Add top_p if specified
	if opts.TopP > 0 {
		req.Options["top_p"] = opts.TopP
	}

	// Add top_k if specified
	if opts.TopK > 0 {
		req.Options["top_k"] = opts.TopK
	}

	// Execute request — accumulate content across callbacks to avoid
	// the content-loss bug where Ollama calls respFunc multiple times
	// (content+Done=false, then empty+Done=true) and the last callback
	// overwrites previous content.
	var contentBuilder strings.Builder
	var lastResponse api.ChatResponse
	respFunc := func(resp api.ChatResponse) error {
		if resp.Message.Content != "" {
			contentBuilder.WriteString(resp.Message.Content)
		}
		lastResponse = resp
		return nil
	}

	err := p.client.Chat(ctx, req, respFunc)
	if err != nil {
		// Check if context was cancelled
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		// Wrap provider errors
		if isConnectionError(err) {
			return nil, fmt.Errorf("%w: %v", ErrProviderUnavailable, err)
		}

		return nil, fmt.Errorf("ollama completion failed: %w", err)
	}

	// Validate response
	finalContent := contentBuilder.String()
	if finalContent == "" {
		return nil, ErrInvalidResponse
	}

	// Build response
	return &Response{
		Text:             finalContent,
		TokensUsed:       lastResponse.PromptEvalCount + lastResponse.EvalCount,
		PromptTokens:     lastResponse.PromptEvalCount,
		CompletionTokens: lastResponse.EvalCount,
		Model:            lastResponse.Model,
		FinishReason:     mapFinishReason(lastResponse.DoneReason),
		Provider:         p.Name(),
	}, nil
}

// Ainvoke generates a streaming text completion using Ollama.
//
// Supports both streaming (opts.Stream=true) and non-streaming (opts.Stream=false) modes.
// Returns a channel that receives StreamChunks as the model generates text.
func (p *OllamaProvider) Ainvoke(ctx context.Context, prompt string, opts *CompletionOptions) <-chan StreamChunk {
	ch := make(chan StreamChunk, 10)

	go func() {
		defer close(ch)

		if opts == nil {
			opts = DefaultOptions()
		}

		// Build messages
		messages := []api.Message{}

		// Add system prompt if provided
		if opts.SystemPrompt != "" {
			messages = append(messages, api.Message{
				Role:    "system",
				Content: opts.SystemPrompt,
			})
		}

		// Add user prompt
		messages = append(messages, api.Message{
			Role:    "user",
			Content: prompt,
		})

		// Build request
		stream := opts.Stream
		req := &api.ChatRequest{
			Model:    p.model,
			Messages: messages,
			Stream:   &stream,
			Options: map[string]interface{}{
				"temperature": opts.Temperature,
			},
		}

		// Add max tokens if specified
		if opts.MaxTokens > 0 {
			req.Options["num_predict"] = opts.MaxTokens
		}

		// Add stop words if specified
		if len(opts.StopWords) > 0 {
			req.Options["stop"] = opts.StopWords
		}

		// Add top_p if specified
		if opts.TopP > 0 {
			req.Options["top_p"] = opts.TopP
		}

		// Add top_k if specified
		if opts.TopK > 0 {
			req.Options["top_k"] = opts.TopK
		}

		// Execute request with streaming
		var lastResponse api.ChatResponse
		var contentBuilder strings.Builder
		respFunc := func(resp api.ChatResponse) error {
			// Check context cancellation
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}

			// Store last response for final metadata
			lastResponse = resp

			if stream {
				// Streaming mode: send each chunk as it arrives
				chunk := StreamChunk{
					Text: resp.Message.Content,
					Done: resp.Done,
				}

				// Add metadata on final chunk
				if resp.Done {
					chunk.TokensUsed = resp.PromptEvalCount + resp.EvalCount
					chunk.PromptTokens = resp.PromptEvalCount
					chunk.CompletionTokens = resp.EvalCount
					chunk.Model = resp.Model
					chunk.FinishReason = mapFinishReason(resp.DoneReason)
				}

				ch <- chunk
			} else {
				// Non-streaming mode: accumulate content across callbacks.
				// Ollama may call this multiple times (content+Done=false,
				// then empty+Done=true), so we must accumulate.
				if resp.Message.Content != "" {
					contentBuilder.WriteString(resp.Message.Content)
				}

				if resp.Done {
					ch <- StreamChunk{
						Text:             contentBuilder.String(),
						Done:             true,
						TokensUsed:       resp.PromptEvalCount + resp.EvalCount,
						PromptTokens:     resp.PromptEvalCount,
						CompletionTokens: resp.EvalCount,
						Model:            resp.Model,
						FinishReason:     mapFinishReason(resp.DoneReason),
					}
				}
			}

			return nil
		}

		err := p.client.Chat(ctx, req, respFunc)
		if err != nil {
			// Check if context was cancelled
			if ctx.Err() != nil {
				ch <- StreamChunk{
					Error: ctx.Err(),
					Done:  true,
				}
				return
			}

			// Wrap provider errors
			if isConnectionError(err) {
				ch <- StreamChunk{
					Error: fmt.Errorf("%w: %v", ErrProviderUnavailable, err),
					Done:  true,
				}
				return
			}

			ch <- StreamChunk{
				Error: fmt.Errorf("ollama completion failed: %w", err),
				Done:  true,
			}
			return
		}

		// Validate response
		if !stream && contentBuilder.Len() == 0 && lastResponse.Message.Content == "" {
			ch <- StreamChunk{
				Error: ErrInvalidResponse,
				Done:  true,
			}
		}
	}()

	return ch
}

// ChatMessages implements ConversationProvider for multi-turn conversations.
//
// This passes messages with proper role attribution directly to Ollama's Chat API,
// instead of packing everything into a single "user" message. This is critical
// for tool-calling flows where the LLM needs to see tool results as separate messages.
func (p *OllamaProvider) ChatMessages(ctx context.Context, messages []ChatMessage, opts *CompletionOptions) <-chan StreamChunk {
	ch := make(chan StreamChunk, 10)

	go func() {
		defer close(ch)

		if opts == nil {
			opts = DefaultOptions()
		}

		// Convert ChatMessage to api.Message
		apiMessages := make([]api.Message, len(messages))
		for i, msg := range messages {
			apiMsg := api.Message{
				Role:       msg.Role,
				Content:    msg.Content,
				ToolCallID: msg.ToolCallID,
				ToolName:   msg.ToolName,
			}
			// Convert tool calls if present (for assistant messages requesting tools)
			if len(msg.ToolCalls) > 0 {
				for _, tc := range msg.ToolCalls {
					args := api.NewToolCallFunctionArguments()
					for k, v := range tc.Args {
						args.Set(k, v)
					}
					apiMsg.ToolCalls = append(apiMsg.ToolCalls, api.ToolCall{
						ID: tc.ID,
						Function: api.ToolCallFunction{
							Name:      tc.Name,
							Arguments: args,
						},
					})
				}
			}
			apiMessages[i] = apiMsg
		}

		// Build request
		stream := opts.Stream
		req := &api.ChatRequest{
			Model:    p.model,
			Messages: apiMessages,
			Stream:   &stream,
			Options: map[string]interface{}{
				"temperature": opts.Temperature,
			},
		}

		// Add max tokens if specified
		if opts.MaxTokens > 0 {
			req.Options["num_predict"] = opts.MaxTokens
		}

		// Add stop words if specified
		if len(opts.StopWords) > 0 {
			req.Options["stop"] = opts.StopWords
		}

		// Add top_p if specified
		if opts.TopP > 0 {
			req.Options["top_p"] = opts.TopP
		}

		// Add top_k if specified
		if opts.TopK > 0 {
			req.Options["top_k"] = opts.TopK
		}

		// Execute request with streaming
		var contentBuilder strings.Builder
		respFunc := func(resp api.ChatResponse) error {
			// Check context cancellation
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}

			if stream {
				// Streaming mode: send each chunk as it arrives
				chunk := StreamChunk{
					Text: resp.Message.Content,
					Done: resp.Done,
				}

				// Add metadata on final chunk
				if resp.Done {
					chunk.TokensUsed = resp.PromptEvalCount + resp.EvalCount
					chunk.PromptTokens = resp.PromptEvalCount
					chunk.CompletionTokens = resp.EvalCount
					chunk.Model = resp.Model
					chunk.FinishReason = mapFinishReason(resp.DoneReason)
				}

				ch <- chunk
			} else {
				// Non-streaming mode: accumulate content across callbacks
				if resp.Message.Content != "" {
					contentBuilder.WriteString(resp.Message.Content)
				}

				if resp.Done {
					ch <- StreamChunk{
						Text:             contentBuilder.String(),
						Done:             true,
						TokensUsed:       resp.PromptEvalCount + resp.EvalCount,
						PromptTokens:     resp.PromptEvalCount,
						CompletionTokens: resp.EvalCount,
						Model:            resp.Model,
						FinishReason:     mapFinishReason(resp.DoneReason),
					}
				}
			}

			return nil
		}

		err := p.client.Chat(ctx, req, respFunc)
		if err != nil {
			if ctx.Err() != nil {
				ch <- StreamChunk{
					Error: ctx.Err(),
					Done:  true,
				}
				return
			}

			if isConnectionError(err) {
				ch <- StreamChunk{
					Error: fmt.Errorf("%w: %v", ErrProviderUnavailable, err),
					Done:  true,
				}
				return
			}

			ch <- StreamChunk{
				Error: fmt.Errorf("ollama chat failed: %w", err),
				Done:  true,
			}
			return
		}

		// Validate response
		if !stream && contentBuilder.Len() == 0 {
			ch <- StreamChunk{
				Error: ErrInvalidResponse,
				Done:  true,
			}
		}
	}()

	return ch
}

// ChatWithTools implements ToolCallingProvider.
//
// It sends the conversation history with structured tool definitions to Ollama.
// When the model requests tool calls, the final StreamChunk has ToolCalls set.
// When the model produces a text response, the final StreamChunk has Text set.
//
// This method always uses non-streaming mode to ensure complete tool call data.
func (p *OllamaProvider) ChatWithTools(ctx context.Context, messages []ChatMessage, tools []ToolDef, opts *CompletionOptions) <-chan StreamChunk {
	ch := make(chan StreamChunk, 10)

	go func() {
		defer close(ch)

		if opts == nil {
			opts = DefaultOptions()
		}

		// Convert ChatMessage slice to api.Message slice (same as ChatMessages)
		apiMessages := make([]api.Message, 0, len(messages))
		for _, msg := range messages {
			apiMsg := api.Message{
				Role:       msg.Role,
				Content:    msg.Content,
				ToolCallID: msg.ToolCallID,
				ToolName:   msg.ToolName,
			}
			if len(msg.ToolCalls) > 0 {
				for _, tc := range msg.ToolCalls {
					args := api.NewToolCallFunctionArguments()
					for k, v := range tc.Args {
						args.Set(k, v)
					}
					apiMsg.ToolCalls = append(apiMsg.ToolCalls, api.ToolCall{
						ID: tc.ID,
						Function: api.ToolCallFunction{
							Name:      tc.Name,
							Arguments: args,
						},
					})
				}
			}
			apiMessages = append(apiMessages, apiMsg)
		}

		// Convert ToolDef slice to api.Tools
		apiTools := make(api.Tools, 0, len(tools))
		for _, td := range tools {
			params, err := toolDefToOllamaParams(td.Parameters)
			if err != nil {
				ch <- StreamChunk{
					Error: fmt.Errorf("invalid tool parameters for %s: %w", td.Name, err),
					Done:  true,
				}
				return
			}
			apiTools = append(apiTools, api.Tool{
				Type: "function",
				Function: api.ToolFunction{
					Name:        td.Name,
					Description: td.Description,
					Parameters:  params,
				},
			})
		}

		// Always use non-streaming: tool calls must be complete before execution
		streamFalse := false
		req := &api.ChatRequest{
			Model:    p.model,
			Messages: apiMessages,
			Tools:    apiTools,
			Stream:   &streamFalse,
			Options: map[string]interface{}{
				"temperature": opts.Temperature,
			},
		}

		if opts.MaxTokens > 0 {
			req.Options["num_predict"] = opts.MaxTokens
		}
		if len(opts.StopWords) > 0 {
			req.Options["stop"] = opts.StopWords
		}

		// Accumulate content and tool calls across callbacks
		// (Ollama may invoke the callback multiple times even in non-streaming mode)
		var contentBuilder strings.Builder
		var toolCalls []ToolCall
		var lastResponse api.ChatResponse

		respFunc := func(resp api.ChatResponse) error {
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}

			lastResponse = resp

			if resp.Message.Content != "" {
				contentBuilder.WriteString(resp.Message.Content)
			}

			// Collect tool calls from each callback
			if len(resp.Message.ToolCalls) > 0 {
				for _, tc := range resp.Message.ToolCalls {
					toolCalls = append(toolCalls, ToolCall{
						ID:   tc.ID,
						Name: tc.Function.Name,
						Args: tc.Function.Arguments.ToMap(),
					})
				}
			}

			return nil
		}

		err := p.client.Chat(ctx, req, respFunc)
		if err != nil {
			if ctx.Err() != nil {
				ch <- StreamChunk{Error: ctx.Err(), Done: true}
				return
			}
			if isConnectionError(err) {
				ch <- StreamChunk{
					Error: fmt.Errorf("%w: %v", ErrProviderUnavailable, err),
					Done:  true,
				}
				return
			}
			ch <- StreamChunk{
				Error: fmt.Errorf("ollama tool chat failed: %w", err),
				Done:  true,
			}
			return
		}

		ch <- StreamChunk{
			Text:             contentBuilder.String(),
			ToolCalls:        toolCalls,
			Done:             true,
			TokensUsed:       lastResponse.PromptEvalCount + lastResponse.EvalCount,
			PromptTokens:     lastResponse.PromptEvalCount,
			CompletionTokens: lastResponse.EvalCount,
			Model:            lastResponse.Model,
			FinishReason:     mapFinishReason(lastResponse.DoneReason),
		}
	}()

	return ch
}

// toolDefToOllamaParams converts a ToolDef's Parameters (JSON Schema) to
// Ollama's ToolFunctionParameters by marshal/unmarshal round-trip.
func toolDefToOllamaParams(params interface{}) (api.ToolFunctionParameters, error) {
	if params == nil {
		return api.ToolFunctionParameters{Type: "object"}, nil
	}

	data, err := json.Marshal(params)
	if err != nil {
		return api.ToolFunctionParameters{}, fmt.Errorf("marshal parameters: %w", err)
	}

	var tfp api.ToolFunctionParameters
	if err := json.Unmarshal(data, &tfp); err != nil {
		return api.ToolFunctionParameters{}, fmt.Errorf("unmarshal parameters: %w", err)
	}

	return tfp, nil
}

// Name returns the provider identifier.
func (p *OllamaProvider) Name() string {
	return fmt.Sprintf("ollama:%s", p.model)
}

// MaxTokens returns the maximum context size.
//
// Note: This is model-dependent. Different models have different context windows.
// Common values:
//   - llama3.2: 8K tokens
//   - mistral: 32K tokens
//   - qwen2.5: 128K tokens
func (p *OllamaProvider) MaxTokens() int {
	// Default to 8K (conservative)
	// TODO: Query model info from Ollama to get actual context size
	return 8192
}

// Ping checks if the Ollama server is reachable.
func (p *OllamaProvider) Ping(ctx context.Context) error {
	// Try to list models as a health check
	_, err := p.client.List(ctx)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrProviderUnavailable, err)
	}
	return nil
}

// ListModels returns available models from the Ollama server.
func (p *OllamaProvider) ListModels(ctx context.Context) ([]string, error) {
	resp, err := p.client.List(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to list models: %w", err)
	}

	models := make([]string, len(resp.Models))
	for i, model := range resp.Models {
		models[i] = model.Name
	}

	return models, nil
}

// isConnectionError checks if an error is a connection-related error.
func isConnectionError(err error) bool {
	if err == nil {
		return false
	}

	errStr := err.Error()
	connectionErrors := []string{
		"connection refused",
		"no such host",
		"timeout",
		"network is unreachable",
		"connection reset",
	}

	for _, substr := range connectionErrors {
		if contains(errStr, substr) {
			return true
		}
	}

	return false
}

// contains checks if a string contains a substring (case-insensitive).
func contains(s, substr string) bool {
	return len(s) >= len(substr) &&
		(s == substr || len(s) > len(substr) &&
		anySubstring(s, substr))
}

func anySubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// mapFinishReason maps Ollama's done reason to standard finish reasons.
func mapFinishReason(doneReason string) string {
	switch doneReason {
	case "stop":
		return "stop"
	case "length":
		return "length"
	case "":
		return "stop" // Default
	default:
		return doneReason
	}
}
