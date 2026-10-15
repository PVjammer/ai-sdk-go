package llm

// OpenAIProvider speaks the OpenAI Chat Completions API (POST {baseURL}/chat/completions),
// which covers real OpenAI as well as any OpenAI-compatible server — llama.cpp's
// `llama-server`, vLLM, LM Studio, litellm proxies, etc.
//
// Ported from baish (~/Workspace/ai-projects/POC/ai-shell/llmprovider/openai.go), which
// implemented this against these same interfaces as an out-of-tree addition. This is that
// addition, backported, per docs/design/compaction-v2.md's broader "pull baish's good ideas
// into ai-sdk-go" mandate — not a rewrite.

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Compile-time interface checks.
var _ Provider = (*OpenAIProvider)(nil)
var _ ConversationProvider = (*OpenAIProvider)(nil)
var _ ToolCallingProvider = (*OpenAIProvider)(nil)

// OpenAIProvider implements Provider against the OpenAI Chat Completions API.
// baseURL is used verbatim — pass "https://api.openai.com/v1" for real OpenAI, or e.g.
// "http://192.168.1.88:4000/v1" for a local litellm/vLLM/llama.cpp endpoint.
//
// Unlike OllamaProvider's Hermes mode (which injects tool schemas into the system prompt and
// regex-parses <tool_call> blocks to work around Ollama template bugs), OpenAIProvider always
// uses the API's native tools/tool_calls fields. No text parsing is needed.
type OpenAIProvider struct {
	baseURL    string
	model      string
	apiKey     string
	httpClient *http.Client
	maxTokens  int // context window size reported by MaxTokens()
}

// OpenAIOption configures an OpenAIProvider.
type OpenAIOption func(*OpenAIProvider)

// WithAPIKey sets the bearer token sent as "Authorization: Bearer <key>".
// Most local servers don't require one; leave unset in that case.
func WithAPIKey(key string) OpenAIOption { return func(p *OpenAIProvider) { p.apiKey = key } }

// WithContextLength overrides the value returned by MaxTokens(). The API itself has no way to
// report this, so callers who know their model's context window should set it explicitly.
func WithContextLength(tokens int) OpenAIOption {
	return func(p *OpenAIProvider) { p.maxTokens = tokens }
}

// WithHTTPClient overrides the default http.Client (e.g. for custom timeouts or TLS config).
func WithHTTPClient(c *http.Client) OpenAIOption { return func(p *OpenAIProvider) { p.httpClient = c } }

// NewOpenAIProvider creates a provider targeting an OpenAI-compatible /chat/completions
// endpoint at baseURL. baseURL should include any API version path segment the server expects
// (e.g. ".../v1"); "" defaults to "https://api.openai.com/v1".
func NewOpenAIProvider(baseURL, model string, opts ...OpenAIOption) (*OpenAIProvider, error) {
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}
	if model == "" {
		return nil, fmt.Errorf("model name is required")
	}

	p := &OpenAIProvider{
		baseURL:    strings.TrimRight(baseURL, "/"),
		model:      model,
		httpClient: &http.Client{Timeout: 5 * time.Minute},
		maxTokens:  8192,
	}
	for _, opt := range opts {
		opt(p)
	}
	return p, nil
}

// Name returns the provider identifier in "provider:model" format.
func (p *OpenAIProvider) Name() string { return fmt.Sprintf("openai:%s", p.model) }

// MaxTokens returns the context window size (default 8192, override via WithContextLength).
func (p *OpenAIProvider) MaxTokens() int { return p.maxTokens }

// Complete generates a text completion.
//
// DEPRECATED: use Ainvoke for new code.
func (p *OpenAIProvider) Complete(ctx context.Context, prompt string, opts *CompletionOptions) (*Response, error) {
	ch := p.Ainvoke(ctx, prompt, opts)
	var final StreamChunk
	var text strings.Builder
	for chunk := range ch {
		if chunk.Error != nil {
			return nil, chunk.Error
		}
		text.WriteString(chunk.Text)
		final = chunk
	}
	return &Response{
		Text:             text.String(),
		TokensUsed:       final.TokensUsed,
		PromptTokens:     final.PromptTokens,
		CompletionTokens: final.CompletionTokens,
		Model:            final.Model,
		FinishReason:     final.FinishReason,
		Provider:         p.Name(),
	}, nil
}

// Ainvoke generates a streaming completion for a single prompt.
func (p *OpenAIProvider) Ainvoke(ctx context.Context, prompt string, opts *CompletionOptions) <-chan StreamChunk {
	if opts == nil {
		opts = DefaultOptions()
	}
	messages := []ChatMessage{}
	if opts.SystemPrompt != "" {
		messages = append(messages, ChatMessage{Role: "system", Content: opts.SystemPrompt})
	}
	messages = append(messages, ChatMessage{Role: "user", Content: prompt})
	return p.ChatMessages(ctx, messages, opts)
}

// ChatMessages implements ConversationProvider.
func (p *OpenAIProvider) ChatMessages(ctx context.Context, messages []ChatMessage, opts *CompletionOptions) <-chan StreamChunk {
	ch := make(chan StreamChunk, 10)
	go func() {
		defer close(ch)
		if opts == nil {
			opts = DefaultOptions()
		}

		req := p.buildRequest(messages, nil, opts, opts.Stream)

		if opts.Stream {
			p.streamChat(ctx, req, ch)
			return
		}
		p.syncChat(ctx, req, ch, false)
	}()
	return ch
}

// ChatWithTools implements ToolCallingProvider using the API's native tools field. Always runs
// non-streaming so a complete tool_calls array is available before returning.
func (p *OpenAIProvider) ChatWithTools(ctx context.Context, messages []ChatMessage, tools []ToolDef, opts *CompletionOptions) <-chan StreamChunk {
	ch := make(chan StreamChunk, 10)
	go func() {
		defer close(ch)
		if opts == nil {
			opts = DefaultOptions()
		}

		req := p.buildRequest(messages, tools, opts, false)
		p.syncChat(ctx, req, ch, true)
	}()
	return ch
}

// buildRequest converts ChatMessage/ToolDef/CompletionOptions into the OpenAI wire format.
func (p *OpenAIProvider) buildRequest(messages []ChatMessage, tools []ToolDef, opts *CompletionOptions, stream bool) oaiChatRequest {
	oaiMessages := make([]oaiMessage, len(messages))
	for i, m := range messages {
		om := oaiMessage{Role: m.Role, ToolCallID: m.ToolCallID}
		// Assistant messages that carry tool calls and no text conventionally omit content
		// (null) rather than sending "". Every other message (including empty-string tool
		// results) sends content explicitly.
		if !(m.Role == "assistant" && len(m.ToolCalls) > 0 && m.Content == "") {
			c := m.Content
			om.Content = &c
		}
		for _, tc := range m.ToolCalls {
			args, _ := json.Marshal(tc.Args)
			om.ToolCalls = append(om.ToolCalls, oaiToolCall{
				ID:   tc.ID,
				Type: "function",
				Function: oaiToolCallFunction{
					Name:      tc.Name,
					Arguments: string(args),
				},
			})
		}
		oaiMessages[i] = om
	}

	req := oaiChatRequest{
		Model:    p.model,
		Messages: oaiMessages,
		Stream:   stream,
	}
	if opts.Temperature > 0 {
		req.Temperature = &opts.Temperature
	}
	if opts.MaxTokens > 0 {
		req.MaxTokens = &opts.MaxTokens
	}
	if opts.TopP > 0 {
		req.TopP = &opts.TopP
	}
	if len(opts.StopWords) > 0 {
		req.Stop = opts.StopWords
	}

	for _, td := range tools {
		req.Tools = append(req.Tools, oaiTool{
			Type: "function",
			Function: oaiToolFunction{
				Name:        td.Name,
				Description: td.Description,
				Parameters:  td.Parameters,
			},
		})
	}

	return req
}

// syncChat performs a non-streaming POST and emits a single final StreamChunk.
func (p *OpenAIProvider) syncChat(ctx context.Context, req oaiChatRequest, ch chan<- StreamChunk, wantTools bool) {
	body, err := json.Marshal(req)
	if err != nil {
		ch <- StreamChunk{Error: fmt.Errorf("marshal request: %w", err), Done: true}
		return
	}

	resp, err := p.post(ctx, body)
	if err != nil {
		ch <- StreamChunk{Error: p.wrapErr(ctx, err), Done: true}
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		ch <- StreamChunk{Error: p.httpError(resp), Done: true}
		return
	}

	var parsed oaiChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		ch <- StreamChunk{Error: fmt.Errorf("decode response: %w", err), Done: true}
		return
	}
	if len(parsed.Choices) == 0 {
		ch <- StreamChunk{Error: ErrInvalidResponse, Done: true}
		return
	}

	choice := parsed.Choices[0]
	chunk := StreamChunk{
		Text:             choice.Message.finalText(),
		Done:             true,
		TokensUsed:       parsed.Usage.TotalTokens,
		PromptTokens:     parsed.Usage.PromptTokens,
		CompletionTokens: parsed.Usage.CompletionTokens,
		Model:            parsed.Model,
		FinishReason:     mapOpenAIFinishReason(choice.FinishReason),
	}
	if wantTools {
		for _, tc := range choice.Message.ToolCalls {
			chunk.ToolCalls = append(chunk.ToolCalls, toolCallFromWire(tc))
		}
	}
	ch <- chunk
}

// streamChat performs a streaming POST and emits StreamChunks as SSE deltas arrive, per the
// OpenAI streaming format ("data: {...}\n\n", "data: [DONE]").
func (p *OpenAIProvider) streamChat(ctx context.Context, req oaiChatRequest, ch chan<- StreamChunk) {
	body, err := json.Marshal(req)
	if err != nil {
		ch <- StreamChunk{Error: fmt.Errorf("marshal request: %w", err), Done: true}
		return
	}

	resp, err := p.post(ctx, body)
	if err != nil {
		ch <- StreamChunk{Error: p.wrapErr(ctx, err), Done: true}
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		ch <- StreamChunk{Error: p.httpError(resp), Done: true}
		return
	}

	var model string
	var finishReason string
	filter := &thinkFilter{}
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		select {
		case <-ctx.Done():
			ch <- StreamChunk{Error: ctx.Err(), Done: true}
			return
		default:
		}

		line := strings.TrimSpace(scanner.Text())
		if line == "" || !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "[DONE]" {
			break
		}

		var evt oaiStreamEvent
		if err := json.Unmarshal([]byte(payload), &evt); err != nil {
			continue
		}
		if evt.Model != "" {
			model = evt.Model
		}
		if len(evt.Choices) == 0 {
			continue
		}
		delta := evt.Choices[0]
		if delta.FinishReason != "" {
			finishReason = delta.FinishReason
		}
		if delta.Delta.Content != "" {
			if out := filter.feed(delta.Delta.Content); out != "" {
				ch <- StreamChunk{Text: out}
			}
		}
	}
	if err := scanner.Err(); err != nil {
		ch <- StreamChunk{Error: fmt.Errorf("read stream: %w", err), Done: true}
		return
	}
	if out := filter.flush(); out != "" {
		ch <- StreamChunk{Text: out}
	}

	ch <- StreamChunk{
		Done:         true,
		Model:        model,
		FinishReason: mapOpenAIFinishReason(finishReason),
	}
}

func (p *OpenAIProvider) post(ctx context.Context, body []byte) (*http.Response, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	}
	return p.httpClient.Do(httpReq)
}

func (p *OpenAIProvider) wrapErr(ctx context.Context, err error) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if isConnErr(err) {
		return fmt.Errorf("%w: %v", ErrProviderUnavailable, err)
	}
	return fmt.Errorf("openai: request failed: %w", err)
}

func (p *OpenAIProvider) httpError(resp *http.Response) error {
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	switch resp.StatusCode {
	case http.StatusUnauthorized, http.StatusForbidden:
		return fmt.Errorf("%w: HTTP %d: %s", ErrAuthenticationFailed, resp.StatusCode, b)
	case http.StatusTooManyRequests:
		return fmt.Errorf("%w: HTTP %d: %s", ErrRateLimitExceeded, resp.StatusCode, b)
	default:
		return fmt.Errorf("openai: HTTP %d: %s", resp.StatusCode, b)
	}
}

// Ping checks whether the server is reachable via GET {baseURL}/models.
func (p *OpenAIProvider) Ping(ctx context.Context) error {
	_, err := p.ListModels(ctx)
	return err
}

// ListModels returns model IDs from GET {baseURL}/models.
func (p *OpenAIProvider) ListModels(ctx context.Context) ([]string, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, p.baseURL+"/models", nil)
	if err != nil {
		return nil, err
	}
	if p.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	}
	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, p.wrapErr(ctx, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, p.httpError(resp)
	}

	var parsed struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, fmt.Errorf("decode models: %w", err)
	}
	models := make([]string, len(parsed.Data))
	for i, m := range parsed.Data {
		models[i] = m.ID
	}
	return models, nil
}

// ── Wire types ───────────────────────────────────────────────────────────

type oaiChatRequest struct {
	Model       string       `json:"model"`
	Messages    []oaiMessage `json:"messages"`
	Stream      bool         `json:"stream"`
	Temperature *float64     `json:"temperature,omitempty"`
	MaxTokens   *int         `json:"max_tokens,omitempty"`
	TopP        *float64     `json:"top_p,omitempty"`
	Stop        []string     `json:"stop,omitempty"`
	Tools       []oaiTool    `json:"tools,omitempty"`
}

type oaiMessage struct {
	Role       string        `json:"role"`
	Content    *string       `json:"content"`
	ToolCallID string        `json:"tool_call_id,omitempty"`
	ToolCalls  []oaiToolCall `json:"tool_calls,omitempty"`

	// ReasoningContent is populated by servers (llama.cpp, DeepSeek-style APIs) that split a
	// reasoning model's <think> block out of Content. Only read on responses; never set on
	// outgoing request messages.
	ReasoningContent *string `json:"reasoning_content,omitempty"`
}

// contentString returns Content, treating a nil pointer (tool-call-only assistant messages
// omit content in some servers) as "".
func (m oaiMessage) contentString() string {
	if m.Content == nil {
		return ""
	}
	return *m.Content
}

// stripThinking removes a leading <think>...</think> block from s, using the same thinkFilter
// logic streaming responses use — so both paths agree on the same edge case: a block that
// opens but never closes (truncated by a token limit) has no recoverable answer and yields "",
// rather than leaking the dangling "<think>" tag as if it were real content.
func stripThinking(s string) string {
	f := &thinkFilter{}
	return strings.TrimSpace(f.feed(s) + f.flush())
}

// finalText returns the message's answer text with any inline <think> block removed. If that
// leaves nothing — e.g. the model spent its whole token budget thinking and never reached an
// answer, or a server splits reasoning into ReasoningContent but the answer landed there too —
// it falls back to a cleaned ReasoningContent rather than reporting an empty response.
func (m oaiMessage) finalText() string {
	if clean := stripThinking(m.contentString()); clean != "" {
		return clean
	}
	if m.ReasoningContent != nil {
		return stripThinking(*m.ReasoningContent)
	}
	return ""
}

const (
	thinkOpenTag  = "<think>"
	thinkCloseTag = "</think>"
)

// thinkFilter suppresses a single leading <think>...</think> block from a stream of text
// deltas, forwarding everything else unchanged. It only buffers while a leading block is still
// possible (the same behavior stripThinking gives non-streaming responses); once the stream
// diverges from "<think>" or the block closes, deltas pass straight through.
type thinkFilter struct {
	buf   strings.Builder
	state int // 0=detecting, 1=inside a think block, 2=passthrough
}

// feed processes one delta and returns the text (possibly empty) that should be forwarded to
// the caller.
func (f *thinkFilter) feed(s string) string {
	if f.state == 2 {
		return s
	}

	f.buf.WriteString(s)
	buffered := f.buf.String()

	if f.state == 0 {
		switch {
		case strings.HasPrefix(buffered, thinkOpenTag):
			f.state = 1
		case len(buffered) < len(thinkOpenTag) && strings.HasPrefix(thinkOpenTag, buffered):
			return "" // still an unambiguous prefix of "<think>" — keep buffering
		default:
			f.state = 2
			f.buf.Reset()
			return buffered
		}
	}

	// state == 1: inside the block, looking for the close tag.
	if idx := strings.Index(buffered, thinkCloseTag); idx >= 0 {
		rest := buffered[idx+len(thinkCloseTag):]
		f.state = 2
		f.buf.Reset()
		return strings.TrimLeft(rest, "\n")
	}
	return ""
}

// flush returns any buffered text once the stream ends. If a "<think>" prefix was never
// confirmed, the buffered text is real content and is returned as-is. If a think block was
// opened but never closed (truncated by a token limit), it's discarded — there's no answer to
// recover.
func (f *thinkFilter) flush() string {
	if f.state != 0 {
		return ""
	}
	return f.buf.String()
}

type oaiToolCall struct {
	ID       string              `json:"id"`
	Type     string              `json:"type"`
	Function oaiToolCallFunction `json:"function"`
}

type oaiToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON-encoded string per the OpenAI spec
}

type oaiTool struct {
	Type     string          `json:"type"`
	Function oaiToolFunction `json:"function"`
}

type oaiToolFunction struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Parameters  interface{} `json:"parameters,omitempty"`
}

type oaiChatResponse struct {
	Model   string `json:"model"`
	Choices []struct {
		Message      oaiMessage `json:"message"`
		FinishReason string     `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

type oaiStreamEvent struct {
	Model   string `json:"model"`
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
}

func toolCallFromWire(tc oaiToolCall) ToolCall {
	var args map[string]interface{}
	_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
	return ToolCall{ID: tc.ID, Name: tc.Function.Name, Args: args}
}

func mapOpenAIFinishReason(reason string) string {
	switch reason {
	case "":
		return "stop"
	case "tool_calls":
		return "tool_calls"
	default:
		return reason
	}
}

func isConnErr(err error) bool {
	if err == nil {
		return false
	}
	s := err.Error()
	for _, substr := range []string{"connection refused", "no such host", "timeout", "network is unreachable", "connection reset"} {
		if strings.Contains(s, substr) {
			return true
		}
	}
	return false
}
