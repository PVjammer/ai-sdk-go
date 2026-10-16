package hooks

import (
	"context"
	"strings"
	"sync"

	"github.com/pvjammer/ai-sdk-go/pkg/agent"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
	"github.com/pvjammer/ai-sdk-go/pkg/skill"
)

// compactionMarker prefixes a compacted-summary message's content, so a later compaction call
// can recognize and update it rather than stack a second summary on top (docs/design/
// compaction-v2.md's "update, not append" invariant, ported from baish's compaction.go).
const compactionMarker = "[Compacted — session summary follows]"

// historyReplacer is implemented by AgentBase. Compaction needs it because, unlike TokenLimit
// or SlidingWindow (cheap, idempotent, safe to redo identically every round from the PreModel
// hook's return value alone), LLM summarization is expensive and must not re-run every round
// once applied — it has to write back to the agent's PERSISTENT session history, which a
// PreModel hook's return value alone cannot do.
type historyReplacer interface {
	SessionHistory(sessionID string) []llm.ChatMessage
	SeedSession(sessionID string, history []llm.ChatMessage)
}

// CompactionDetails tracks file-touched bookkeeping across arbitrarily many compaction rounds as
// structured data, not prose the model has to re-parse — ported from pi's CompactionDetails
// (compaction.ts:51-55). Meant to grow per agent: add fields as a particular agent's own tools
// warrant tracking something else structurally (docs/design/compaction-v2.md).
type CompactionDetails struct {
	ReadFiles     []string
	ModifiedFiles []string
}

func (d CompactionDetails) merge(read, modified []string) CompactionDetails {
	return CompactionDetails{
		ReadFiles:     mergeUnique(d.ReadFiles, read),
		ModifiedFiles: mergeUnique(d.ModifiedFiles, modified),
	}
}

// CompactionResult is the structured outcome of one compaction run — pi's CompactionResult<T>,
// ported (compaction.ts:104-114). Summary is still prose (compaction is fundamentally an LLM
// summarization step), but Details is typed data that survives arbitrarily many rounds without
// being re-parsed out of a paragraph, unlike baish's plain "**Relevant Files:**" bullet.
//
// Deliberately missing pi's FirstKeptEntryID: this compaction operates on AgentBase's flat
// session history, not a SessionLog, so there is no entry ID to report. A SessionLog-aware
// compaction path (operating on a Session's spine instead of a flat per-session history) would
// add it — not built this pass, see docs/roadmap.md Phase 6's note about re-verifying compaction
// is spine-generic.
type CompactionResult struct {
	Summary              string
	TokensBefore         int
	EstimatedTokensAfter int
	Usage                llm.Usage
	Details              CompactionDetails
}

// FileOpsExtractor inspects one tool call and returns any file paths it read or modified, for
// CompactionDetails. DefaultFileOpsExtractor is a reasonable starting convention, not a mandate
// — pass a custom one via CompactionOptions.ExtractFileOps for an agent whose tools don't match
// it, or whose tools track something other than files.
type FileOpsExtractor func(tc llm.ToolCall) (read []string, modified []string)

// DefaultFileOpsExtractor recognizes common tool-name conventions — anything containing "write",
// "edit", or "create" counts as modified; "read" or "list" counts as read — and looks for a
// "path", "file", or "filename" string argument.
func DefaultFileOpsExtractor(tc llm.ToolCall) (read []string, modified []string) {
	path := firstStringArg(tc.Args, "path", "file", "filename")
	if path == "" {
		return nil, nil
	}
	name := strings.ToLower(tc.Name)
	switch {
	case strings.Contains(name, "write"), strings.Contains(name, "edit"), strings.Contains(name, "create"):
		return nil, []string{path}
	case strings.Contains(name, "read"), strings.Contains(name, "list"):
		return []string{path}, nil
	}
	return nil, nil
}

func firstStringArg(args map[string]interface{}, keys ...string) string {
	for _, k := range keys {
		if v, ok := args[k].(string); ok && v != "" {
			return v
		}
	}
	return ""
}

func mergeUnique(a, b []string) []string {
	if len(a) == 0 && len(b) == 0 {
		return nil
	}
	seen := make(map[string]bool, len(a)+len(b))
	out := make([]string, 0, len(a)+len(b))
	for _, s := range append(append([]string{}, a...), b...) {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return out
}

// CompactionOptions configures Compaction. Zero values take the defaults noted per field.
type CompactionOptions struct {
	// MaxTokens is the token budget compaction protects (default 8000). Estimated at 4
	// characters per token, same heuristic TokenLimit already uses — NOT yet the provider's
	// own reported usage (docs/design/compaction-v2.md's TokenCounter idea); that needs
	// AgentBase to persist per-call usage somewhere a hook can read it, which it doesn't yet.
	// Flagged as a known gap versus the design doc, not silently done.
	MaxTokens int

	// Threshold is the fraction of MaxTokens that triggers compaction (default 0.75).
	//
	// If you're computing MaxTokens yourself as some fraction of a model's real context
	// window, set Threshold to 1.0 explicitly rather than leaving it at the 0.75 default.
	// Otherwise the two fractions multiply — found live: examples/tui-agent originally
	// computed MaxTokens as contextWindow*0.75, then this field's own default 0.75 applied on
	// top of that, so compaction actually triggered at ~0.56 of the real window, and for one
	// run that was low enough to be under 10% of what the model actually had room for. Fixed
	// at that one call site (Threshold: 1.0), not here, since MaxTokens already being a
	// deliberately-conservative sub-budget (the TokenLimit hard-backstop pattern) is a
	// legitimate use of this field as documented — the bug was stacking two independent
	// safety margins without realizing it, not this field's default being wrong.
	Threshold float64

	// ExtractFileOps builds CompactionDetails from each tool call in the span being compacted.
	// Default: DefaultFileOpsExtractor.
	ExtractFileOps FileOpsExtractor

	// OnCompaction, if set, is called with the structured result every time compaction actually
	// runs — the way a caller (a TUI, a log, a test) observes CompactionResult without having to
	// go hunt it out of the hook's internal state.
	OnCompaction func(sessionID string, result CompactionResult)
}

// Compaction returns an AgentHooks whose PreModel hook summarizes old conversation history via
// an LLM call once estimated tokens cross Threshold*MaxTokens — baish's compaction strategy,
// ported onto this SDK's hook system, with pi's typed output layered on top. summarizer is the
// provider used for the (cheap, one-call) summarization request; it can be the same provider the
// agent itself uses, or a smaller/cheaper one.
//
// Hard invariants, not configurable:
//   - never compacts the in-flight turn — everything from the current turn's user message
//     onward stays verbatim. A turn boundary is detected the same way baish's conceptually is:
//     when the newest entry in persistent history is role=="user", that index becomes the new
//     turn boundary; later rounds of the same turn (which never end in a bare user message
//     again until the next real turn) reuse the previously recorded boundary.
//   - non-destructive on failure: a failed summarization call leaves history untouched and the
//     hook tries again on a later round.
//   - update, not append: re-compacting an already-compacted span folds the previous summary
//     AND the previous CompactionDetails back into the new call and replaces both, rather than
//     stacking a second summary or losing previously-tracked file operations.
func Compaction(base historyReplacer, summarizer llm.Provider, opts CompactionOptions) agent.AgentHooks {
	if opts.MaxTokens <= 0 {
		opts.MaxTokens = 8000
	}
	if opts.Threshold <= 0 {
		opts.Threshold = 0.75
	}
	if opts.ExtractFileOps == nil {
		opts.ExtractFileOps = DefaultFileOpsExtractor
	}

	var mu sync.Mutex
	turnStart := make(map[string]int)
	lastResult := make(map[string]CompactionResult) // in-memory only — lost on process restart,
	// same durability as everything else AgentBase stores today (see docs/design/
	// session-dispatch.md's snapshot proposal, not yet implemented). Falls back to parsing the
	// compactionMarker text itself when no in-memory record exists, so a restart degrades to
	// "no structured Details, summary text still updates correctly" rather than stacking a
	// duplicate summary.

	return agent.AgentHooks{
		PreModel: func(ctx context.Context, messages []llm.ChatMessage) ([]llm.ChatMessage, error) {
			sessionID := skill.SessionIDFromCtx(ctx)
			if sessionID == "" {
				return messages, nil
			}

			history := base.SessionHistory(sessionID)
			if len(history) == 0 {
				return messages, nil
			}

			mu.Lock()
			if history[len(history)-1].Role == "user" {
				turnStart[sessionID] = len(history) - 1
			}
			ts := turnStart[sessionID]
			prev, hasPrev := lastResult[sessionID]
			mu.Unlock()

			if ts <= 0 {
				return messages, nil // nothing before the current turn exists to compact
			}
			tokensBefore := estimateMsgTokens(history)
			if tokensBefore < int(float64(opts.MaxTokens)*opts.Threshold) {
				return messages, nil
			}

			compactable := history[:ts]

			rest := compactable
			var previousSummary string
			var previousDetails CompactionDetails
			if parsed, ok := splitExistingSummaryMarker(compactable); ok {
				rest = compactable[1:]
				previousSummary = parsed // fallback if no in-memory record (e.g. after a restart)
			}
			if hasPrev {
				previousSummary = prev.Summary
				previousDetails = prev.Details
			}

			// Nothing new since the last compaction — compactable is just the marker itself,
			// nothing followed it. Re-running the summarizer here would produce the same
			// summary over unchanged input, for no benefit, every single round. That's not
			// just waste: found live on a real task, repeated no-op compaction on a turn whose
			// OWN in-flight content already exceeds the budget (which compaction, by design,
			// never touches — see the turn-boundary comment below) burned an extra LLM call
			// every iteration of a stuck loop, which is what actually exhausted
			// ErrMaxIterations faster than the stuck loop alone would have. Skipping it doesn't
			// fix a model getting stuck, but it stops compaction from making a stuck turn worse.
			if len(rest) == 0 {
				return messages, nil
			}

			read, modified := extractFileOps(rest, opts.ExtractFileOps)
			details := previousDetails.merge(read, modified)

			summary, usage, err := summarizeSpan(ctx, summarizer, previousSummary, rest)
			if err != nil {
				return messages, nil // non-destructive: unchanged, retry next round
			}

			newHistory := make([]llm.ChatMessage, 0, len(history)-ts+1)
			newHistory = append(newHistory, llm.ChatMessage{
				Role:    "assistant",
				Content: compactionMarker + "\n\n" + summary,
			})
			newHistory = append(newHistory, history[ts:]...)
			base.SeedSession(sessionID, newHistory)

			result := CompactionResult{
				Summary:              summary,
				TokensBefore:         tokensBefore,
				EstimatedTokensAfter: estimateMsgTokens(newHistory),
				Usage:                usage,
				Details:              details,
			}

			mu.Lock()
			turnStart[sessionID] = len(newHistory) - (len(history) - ts)
			lastResult[sessionID] = result
			mu.Unlock()

			if opts.OnCompaction != nil {
				opts.OnCompaction(sessionID, result)
			}

			// Also return the compacted view for the call about to happen, so this round
			// benefits immediately rather than waiting until the next one.
			sys, _ := splitSystem(messages)
			return append(sys, newHistory...), nil
		},
	}
}

// splitExistingSummaryMarker reports whether compactable's first message is already a
// compacted-summary marker, and if so, returns its summary text.
func splitExistingSummaryMarker(compactable []llm.ChatMessage) (string, bool) {
	if len(compactable) == 0 || compactable[0].Role != "assistant" || !strings.HasPrefix(compactable[0].Content, compactionMarker) {
		return "", false
	}
	return strings.TrimPrefix(compactable[0].Content, compactionMarker+"\n\n"), true
}

// extractFileOps walks every tool call in span (assistant messages' ToolCalls field) through
// extract, accumulating read/modified file paths.
func extractFileOps(span []llm.ChatMessage, extract FileOpsExtractor) (read, modified []string) {
	for _, m := range span {
		for _, tc := range m.ToolCalls {
			r, w := extract(tc)
			read = append(read, r...)
			modified = append(modified, w...)
		}
	}
	return read, modified
}

// summarizeSpan asks summarizer for a structured summary of span, folding previousSummary back
// in (updating it) when one exists.
func summarizeSpan(ctx context.Context, summarizer llm.Provider, previousSummary string, span []llm.ChatMessage) (string, llm.Usage, error) {
	var sb strings.Builder
	if previousSummary != "" {
		sb.WriteString("Existing summary so far:\n")
		sb.WriteString(previousSummary)
		sb.WriteString("\n\n")
	}
	sb.WriteString("Conversation span to fold into the summary:\n")
	for _, m := range span {
		sb.WriteString(m.Role)
		sb.WriteString(": ")
		sb.WriteString(m.Content)
		sb.WriteString("\n")
	}

	systemPrompt := "Summarize the conversation span below into a structured summary with these " +
		"sections: Goal, Progress, Key Decisions, Relevant Files, Errors & Fixes, Open Questions, " +
		"Next Steps. If an existing summary is given, UPDATE it in place — move completed " +
		"Next-Steps items into Progress, fold in new information — rather than writing a second, " +
		"separate summary alongside it. Output only the updated summary text, no preamble."

	resp, err := summarizer.Complete(ctx, sb.String(), llm.DefaultOptions().WithSystemPrompt(systemPrompt).WithTemperature(0.2))
	if err != nil {
		return "", llm.Usage{}, err
	}
	return strings.TrimSpace(resp.Text), llm.UsageFromResponse(resp), nil
}
