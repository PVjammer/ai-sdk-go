// Package summarize provides text summarization using LLMs.
//
// This package implements abstractive summarization using local or API-based
// language models. It supports multiple output styles, focus areas, and languages.
package summarize

import (
	"context"
	"fmt"
	"reflect"
	"strings"

	"github.com/pvjammer/ai-sdk-go/internal/schema"
	"github.com/pvjammer/ai-sdk-go/internal/validation"
	"github.com/pvjammer/ai-sdk-go/pkg/function"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// SummarizeFunction performs LLM-based text summarization.
type SummarizeFunction struct {
	function.Base
	provider llm.Provider
}

// Input defines the input schema for summarization.
type Input struct {
	Text      string `json:"text" validate:"required,min=10" description:"Text to summarize"`
	MaxLength int    `json:"max_length" validate:"min=10,max=1000" description:"Maximum summary length in words" default:"100"`
	Style     string `json:"style" validate:"oneof=paragraph bullets points" enum:"paragraph,bullets,points" description:"Output style" default:"paragraph"`
	Focus     string `json:"focus" validate:"oneof=general technical business academic" enum:"general,technical,business,academic" description:"Summary focus area" default:"general"`
	Language  string `json:"language" description:"Output language (e.g., english, spanish, french)" default:"english"`
}

// Output defines the output structure.
type Output struct {
	Summary      string  `json:"summary"`
	OriginalLen  int     `json:"original_length"`
	SummaryLen   int     `json:"summary_length"`
	Compression  float64 `json:"compression_ratio"`
	TokensUsed   int     `json:"tokens_used"`
	Model        string  `json:"model"`
}

// New creates a new SummarizeFunction. The LLM provider is resolved lazily
// on first Execute() call so that config bridging (BridgeToViper) has completed.
func New(cfg function.Config) function.AIFunction {
	return &SummarizeFunction{
		Base: function.NewBase(cfg),
		// provider is nil — resolved in Execute via resolveProvider()
	}
}

// NewDefault creates a new SummarizeFunction with default Ollama provider.
func NewDefault() *SummarizeFunction {
	// Use default Ollama provider
	provider, _ := llm.NewOllamaProvider("http://localhost:11434", "llama3.2:latest")

	return &SummarizeFunction{
		Base:     function.NewBase(function.NilConfig{}),
		provider: provider,
	}
}

// NewWithProvider creates a new SummarizeFunction with a specific provider.
// Useful for testing with mock providers.
func NewWithProvider(provider llm.Provider) *SummarizeFunction {
	return &SummarizeFunction{
		Base:     function.NewBase(function.NilConfig{}),
		provider: provider,
	}
}

// resolveProvider returns the existing provider or creates one from config.
// This is called at execution time so that BridgeToViper has already populated
// per-function and global LLM values.
func (f *SummarizeFunction) resolveProvider() llm.Provider {
	if f.provider != nil {
		return f.provider
	}
	f.provider = createProvider(f.Config())
	return f.provider
}

// createProvider creates an LLM provider based on configuration.
// Priority: per-function keys → global [llm] keys → hardcoded defaults.
func createProvider(cfg function.Config) llm.Provider {
	// Per-function LLM override: summarize.llm_provider / llm_endpoint / llm_model
	providerType := cfg.GetString("summarize.llm_provider")
	if providerType == "" {
		providerType = cfg.GetString("llm.provider")
	}

	switch providerType {
	case "ollama", "":
		endpoint := cfg.GetString("summarize.llm_endpoint")
		if endpoint == "" {
			endpoint = cfg.GetString("llm.ollama.endpoint")
		}
		if endpoint == "" {
			endpoint = "http://localhost:11434"
		}

		model := cfg.GetString("summarize.llm_model")
		if model == "" {
			model = cfg.GetString("llm.ollama.model")
		}
		if model == "" {
			model = "llama3.2:latest"
		}

		provider, err := llm.NewOllamaProvider(endpoint, model)
		if err != nil {
			provider, _ = llm.NewOllamaProvider("http://localhost:11434", "llama3.2:latest")
		}
		return provider

	case "mock":
		return llm.NewMockProvider("This is a mock summary.")

	default:
		provider, _ := llm.NewOllamaProvider("http://localhost:11434", "llama3.2:latest")
		return provider
	}
}

// Execute implements the AIFunction interface.
func (f *SummarizeFunction) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	// Type assert and validate
	var in *Input

	switch v := input.(type) {
	case *Input:
		in = v
	case map[string]interface{}:
		in = f.parseMapInput(v)
	default:
		return nil, function.NewExecutionError(f, "input validation", "invalid input type")
	}

	if err := validation.Validate(in); err != nil {
		return nil, function.WrapValidationError(f, err)
	}

	// Check context
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Build prompts
	systemPrompt := f.buildSystemPrompt(in)
	userPrompt := f.buildUserPrompt(in)

	// Call LLM using Ainvoke
	opts := llm.DefaultOptions().
		WithMaxTokens(in.MaxLength * 5). // Rough estimate: words → tokens
		WithTemperature(0.3).              // Lower for factual summarization
		WithSystemPrompt(systemPrompt)
	opts.Stream = false // Non-streaming for synchronous Execute()

	provider := f.resolveProvider()

	chunkCh := provider.Ainvoke(ctx, userPrompt, opts)
	resp, err := llm.CollectResponse(ctx, chunkCh)
	if err != nil {
		return nil, function.WrapExecutionError(f, "llm_completion", err)
	}

	// Set provider name (CollectResponse doesn't set it)
	resp.Provider = provider.Name()

	// Validate response
	if resp.Text == "" {
		return nil, function.NewExecutionError(f, "empty_response", "LLM returned empty summary")
	}

	// Calculate metrics
	originalWords := len(strings.Fields(in.Text))
	summaryWords := len(strings.Fields(resp.Text))
	compression := 0.0
	if originalWords > 0 {
		compression = float64(summaryWords) / float64(originalWords)
	}

	output := &Output{
		Summary:      strings.TrimSpace(resp.Text),
		OriginalLen:  originalWords,
		SummaryLen:   summaryWords,
		Compression:  compression,
		TokensUsed:   resp.TokensUsed,
		Model:        resp.Model,
	}

	return output, nil
}

// parseMapInput parses map[string]interface{} input from CLI.
// Priority: hardcoded defaults < config (Viper) < explicitly-set per-call args.
// The "_changed_flags" key (set by buildInputFromFlags) distinguishes user-set
// flags from Cobra defaults so config values are not masked by schema defaults.
func (f *SummarizeFunction) parseMapInput(m map[string]interface{}) *Input {
	// Determine which keys were explicitly set by the caller
	changed, _ := m["_changed_flags"].(map[string]bool)

	// Start with hardcoded defaults
	in := &Input{
		MaxLength: 100,
		Style:     "paragraph",
		Focus:     "general",
		Language:  "english",
	}

	// Layer config-driven defaults (overrides hardcoded)
	cfg := f.Config()
	if cfg != nil {
		if v := cfg.GetInt("summarize.max_length"); v > 0 {
			in.MaxLength = v
		}
		if v := cfg.GetString("summarize.style"); v != "" {
			in.Style = v
		}
		if v := cfg.GetString("summarize.focus"); v != "" {
			in.Focus = v
		}
		if v := cfg.GetString("summarize.language"); v != "" {
			in.Language = v
		}
	}

	// Layer per-call values — only override if the key was explicitly set
	// (or if there's no _changed_flags metadata, e.g. from tool calls)
	isSet := func(key string) bool {
		return changed == nil || changed[key]
	}

	if text, ok := m["text"].(string); ok && isSet("text") {
		in.Text = text
	}
	if maxLen, ok := m["max_length"].(int); ok && isSet("max_length") {
		in.MaxLength = maxLen
	}
	if style, ok := m["style"].(string); ok && isSet("style") {
		in.Style = style
	}
	if focus, ok := m["focus"].(string); ok && isSet("focus") {
		in.Focus = focus
	}
	if lang, ok := m["language"].(string); ok && isSet("language") {
		in.Language = lang
	}

	return in
}

// buildSystemPrompt creates the system prompt based on input parameters.
func (f *SummarizeFunction) buildSystemPrompt(in *Input) string {
	return fmt.Sprintf(`You are an expert summarization assistant. Your task is to create concise, accurate summaries.

Focus: %s
Style: %s
Language: %s
Max length: approximately %d words

Guidelines:
- Preserve key information and main ideas
- Use clear, concise language
- Maintain factual accuracy
- Format according to requested style
- Output in %s only
- Do not add information not present in the source
- Be direct and avoid meta-commentary`, in.Focus, in.Style, in.Language, in.MaxLength, in.Language)
}

// buildUserPrompt creates the user prompt with style-specific instructions.
func (f *SummarizeFunction) buildUserPrompt(in *Input) string {
	styleInstructions := map[string]string{
		"paragraph": "Write a cohesive paragraph summary that flows naturally.",
		"bullets":   "Create a bulleted list of key points using '•' as the bullet character.",
		"points":    "List the main points concisely, one per line, without bullet characters.",
	}

	instruction := styleInstructions[in.Style]
	if instruction == "" {
		instruction = styleInstructions["paragraph"]
	}

	// Add focus-specific guidance
	focusGuidance := f.getFocusGuidance(in.Focus)

	return fmt.Sprintf(`%s

%s

Text to summarize:
"""
%s
"""

Summary:`, instruction, focusGuidance, in.Text)
}

// getFocusGuidance returns focus-specific guidance for the model.
func (f *SummarizeFunction) getFocusGuidance(focus string) string {
	switch focus {
	case "technical":
		return `Emphasize:
- Technical details and terminology
- Methods and approaches
- Results and metrics
- Key technologies mentioned`

	case "business":
		return `Emphasize:
- Business impact and value
- Stakeholders and decisions
- Financial implications
- Strategic considerations`

	case "academic":
		return `Emphasize:
- Research questions and hypotheses
- Methodology and findings
- Contributions to the field
- Theoretical framework`

	case "general":
		fallthrough
	default:
		return `Emphasize:
- Main ideas and key points
- Important facts and details
- Logical flow and connections`
	}
}

// InputSchema implements the AIFunction interface.
func (f *SummarizeFunction) InputSchema() *function.Schema {
	return schema.MustGenerateSchema(reflect.TypeOf(Input{}))
}

// Metadata implements the AIFunction interface.
func (f *SummarizeFunction) Metadata() *function.Metadata {
	return function.NewMetadata("SummarizeFunction").
		WithName("summarize").
		WithDescription("Generate summaries of text using LLM").
		WithToolName("summarize_text").
		WithToolDescription("Summarizes text using a language model with configurable style, focus, and length").
		WithCLIName("summarize").
		WithCLIDescription("Summarize text with AI").
		WithCategory("text").
		WithTags("text", "summarization", "llm", "nlp", "ai").
		WithExamples(
			function.Example{
				Input: map[string]interface{}{
					"text":       "Long article about artificial intelligence...",
					"max_length": 50,
					"style":      "bullets",
					"focus":      "technical",
				},
				Output: &Output{
					Summary:     "• AI systems use machine learning...\n• Recent advances in deep learning...\n• Applications include computer vision...",
					OriginalLen: 250,
					SummaryLen:  45,
					Compression: 0.18,
				},
				Description: "Summarize technical content as bullet points",
			},
			function.Example{
				Input: map[string]interface{}{
					"text":       "Quarterly earnings report...",
					"max_length": 75,
					"style":      "paragraph",
					"focus":      "business",
				},
				Description: "Summarize business report as paragraph",
			},
		)
}
