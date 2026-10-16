package experiment

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/pvjammer/ai-sdk-go/pkg/agent"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// sessionFile mirrors examples/tui-agent/main.go's on-disk session shape.
type sessionFile struct {
	Branching bool              `json:"branching"`
	Leaf      string            `json:"leaf,omitempty"`
	Entries   []agent.Entry     `json:"entries,omitempty"`
	History   []llm.ChatMessage `json:"history,omitempty"`
}

type ContextSize struct {
	Turn    int `json:"turn"`
	Entries int `json:"entries"`
	Chars   int `json:"chars"`
}

type Violation struct {
	TurnIndex int    `json:"turn_index"` // 1-based
	Tool      string `json:"tool"`
}

// TurnToolCalls is every tool call made on one turn, in request order, with full args — e.g.
// to see whether a "revisit this file" turn retrieved a branch, re-explored from scratch,
// answered from memory with no calls at all, or regressed to an earlier wrong path.
type TurnToolCalls struct {
	Turn  int              `json:"turn"`
	Calls []ToolCallMetric `json:"calls"`
}

// TurnTokens is one turn's own prompt/completion token cost — directly from the provider's
// usage field on every LLM call that turn made, summed across tool-call rounds, not estimated
// from character counts.
type TurnTokens struct {
	Turn             int `json:"turn"`
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

type CaseReport struct {
	Name                    string                  `json:"name"`
	Flags                   map[string]interface{}  `json:"flags"`
	TurnsRun                int                     `json:"turns_run"`
	Errors                  []string                `json:"errors,omitempty"`
	CompactionEvents        int                     `json:"compaction_events"`
	ForbiddenToolViolations []Violation             `json:"forbidden_tool_violations,omitempty"`
	TotalDurationMS         int64                   `json:"total_duration_ms"`
	TotalPromptTokens       int                     `json:"total_prompt_tokens"`
	TotalCompletionTokens   int                     `json:"total_completion_tokens"`
	ContextByTurn           []ContextSize           `json:"context_by_turn"`
	ToolCallsByTurn         []TurnToolCalls         `json:"tool_calls_by_turn"`
	TokensByTurn            []TurnTokens            `json:"tokens_by_turn"`
}

type Report struct {
	RunMeta RunMeta      `json:"run_meta"`
	Cases   []CaseReport `json:"cases"`
}

// BuildReport reads back everything Run wrote for runID — metrics.jsonl, flags.json, and the
// per-turn session snapshots — and computes mechanical metrics only (error/compaction counts,
// forbidden-tool violations, context growth). It does not judge correctness; that's left to
// whatever reviews the saved transcripts and session files afterward.
func BuildReport(resultsRoot, runID string, turns []Turn) (*Report, error) {
	runDir := filepath.Join(resultsRoot, runID)
	var meta RunMeta
	if err := readJSON(filepath.Join(runDir, "run.json"), &meta); err != nil {
		return nil, fmt.Errorf("read run.json: %w", err)
	}

	rep := &Report{RunMeta: meta}
	for _, name := range meta.Cases {
		caseDir := filepath.Join(runDir, name)
		cr, err := buildCaseReport(caseDir, name, turns)
		if err != nil {
			return nil, fmt.Errorf("case %s: %w", name, err)
		}
		rep.Cases = append(rep.Cases, *cr)
	}
	return rep, nil
}

func buildCaseReport(caseDir, name string, turns []Turn) (*CaseReport, error) {
	cr := &CaseReport{Name: name}

	var flags map[string]interface{}
	if err := readJSON(filepath.Join(caseDir, "flags.json"), &flags); err == nil {
		cr.Flags = flags
	}

	metrics, err := readMetrics(filepath.Join(caseDir, "metrics.jsonl"))
	if err != nil {
		return nil, err
	}
	cr.TurnsRun = len(metrics)
	for i, m := range metrics {
		cr.TotalDurationMS += m.DurationMS
		cr.CompactionEvents += len(m.Compactions)
		cr.ToolCallsByTurn = append(cr.ToolCallsByTurn, TurnToolCalls{Turn: i + 1, Calls: m.ToolCalls})
		cr.TokensByTurn = append(cr.TokensByTurn, TurnTokens{Turn: i + 1, PromptTokens: m.PromptTokens, CompletionTokens: m.CompletionTokens})
		cr.TotalPromptTokens += m.PromptTokens
		cr.TotalCompletionTokens += m.CompletionTokens
		if m.Error != "" {
			cr.Errors = append(cr.Errors, fmt.Sprintf("turn %d: %s", i+1, m.Error))
		}
		if i < len(turns) && len(turns[i].ForbiddenTools) > 0 {
			forbidden := make(map[string]bool, len(turns[i].ForbiddenTools))
			for _, t := range turns[i].ForbiddenTools {
				forbidden[t] = true
			}
			for _, tc := range m.ToolCalls {
				if forbidden[tc.Name] {
					cr.ForbiddenToolViolations = append(cr.ForbiddenToolViolations, Violation{TurnIndex: i + 1, Tool: tc.Name})
				}
			}
		}
	}

	for i := 1; i <= cr.TurnsRun; i++ {
		size, err := contextSizeAtTurn(filepath.Join(caseDir, fmt.Sprintf("turn-%d.json", i)))
		if err != nil {
			continue // snapshot missing (e.g. the turn's process never started) — skip, not fatal
		}
		size.Turn = i
		cr.ContextByTurn = append(cr.ContextByTurn, size)
	}
	return cr, nil
}

func readMetrics(path string) ([]TurnMetric, error) {
	f, err := os.Open(path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var out []TurnMetric
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var m TurnMetric
		if err := json.Unmarshal([]byte(line), &m); err != nil {
			continue // one malformed line shouldn't sink the whole report
		}
		out = append(out, m)
	}
	return out, scanner.Err()
}

func contextSizeAtTurn(snapshotPath string) (ContextSize, error) {
	var sf sessionFile
	if err := readJSON(snapshotPath, &sf); err != nil {
		return ContextSize{}, err
	}
	if sf.Branching {
		log := agent.LoadSessionLog(sf.Entries)
		path := log.Path(sf.Leaf)
		chars := 0
		for _, e := range path {
			chars += len(e.Content)
		}
		return ContextSize{Entries: len(path), Chars: chars}, nil
	}
	chars := 0
	for _, m := range sf.History {
		chars += len(m.Content)
	}
	return ContextSize{Entries: len(sf.History), Chars: chars}, nil
}

// WriteReport writes both the machine-readable report.json and a human-skimmable report.md.
func WriteReport(resultsRoot, runID string, rep *Report) error {
	runDir := filepath.Join(resultsRoot, runID)
	if err := writeJSON(filepath.Join(runDir, "report.json"), rep); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(runDir, "report.md"), []byte(renderMarkdown(rep)), 0o644)
}

func renderMarkdown(rep *Report) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# %s — %s\n\n", rep.RunMeta.ConfigName, rep.RunMeta.RunID)
	if rep.RunMeta.GitSHA != "" {
		fmt.Fprintf(&b, "Source: `%s` @ `%s`\n\n", rep.RunMeta.SourceRepo, rep.RunMeta.GitSHA)
	}
	fmt.Fprintf(&b, "| case | turns | errors | compactions | forbidden-tool violations | duration | final context (entries/chars) | prompt tokens | completion tokens | total tokens |\n")
	fmt.Fprintf(&b, "|---|---|---|---|---|---|---|---|---|---|\n")
	for _, c := range rep.Cases {
		final := "—"
		if n := len(c.ContextByTurn); n > 0 {
			f := c.ContextByTurn[n-1]
			final = strconv.Itoa(f.Entries) + "/" + strconv.Itoa(f.Chars)
		}
		fmt.Fprintf(&b, "| %s | %d | %d | %d | %d | %dms | %s | %d | %d | %d |\n",
			c.Name, c.TurnsRun, len(c.Errors), c.CompactionEvents, len(c.ForbiddenToolViolations), c.TotalDurationMS, final,
			c.TotalPromptTokens, c.TotalCompletionTokens, c.TotalPromptTokens+c.TotalCompletionTokens)
	}
	fmt.Fprintln(&b)

	for _, c := range rep.Cases {
		if len(c.Errors) == 0 && len(c.ForbiddenToolViolations) == 0 {
			continue
		}
		fmt.Fprintf(&b, "## %s — issues\n\n", c.Name)
		for _, e := range c.Errors {
			fmt.Fprintf(&b, "- error: %s\n", e)
		}
		for _, v := range c.ForbiddenToolViolations {
			fmt.Fprintf(&b, "- turn %d called forbidden tool %q\n", v.TurnIndex, v.Tool)
		}
		fmt.Fprintln(&b)
	}

	fmt.Fprintln(&b, "## Context growth by turn")
	fmt.Fprintln(&b)
	for _, c := range rep.Cases {
		fmt.Fprintf(&b, "**%s**\n\n", c.Name)
		for _, s := range c.ContextByTurn {
			fmt.Fprintf(&b, "- turn %d: %d entries, %d chars\n", s.Turn, s.Entries, s.Chars)
		}
		fmt.Fprintln(&b)
	}

	fmt.Fprintln(&b, "## Tool calls by turn")
	fmt.Fprintln(&b)
	for _, c := range rep.Cases {
		fmt.Fprintf(&b, "**%s**\n\n", c.Name)
		for _, t := range c.ToolCallsByTurn {
			fmt.Fprintf(&b, "- turn %d: %s\n", t.Turn, formatToolCalls(t.Calls))
		}
		fmt.Fprintln(&b)
	}

	fmt.Fprintln(&b, "## Tokens by turn (prompt/completion, direct from provider usage)")
	fmt.Fprintln(&b)
	for _, c := range rep.Cases {
		fmt.Fprintf(&b, "**%s** (total: %d prompt + %d completion = %d)\n\n",
			c.Name, c.TotalPromptTokens, c.TotalCompletionTokens, c.TotalPromptTokens+c.TotalCompletionTokens)
		for _, t := range c.TokensByTurn {
			fmt.Fprintf(&b, "- turn %d: %d prompt, %d completion\n", t.Turn, t.PromptTokens, t.CompletionTokens)
		}
		fmt.Fprintln(&b)
	}
	return b.String()
}

// formatToolCalls renders one turn's tool calls compactly for the markdown report — e.g.
// "read_branch(branch_id=B5), write_file(path=pkg/llm/usage.go)" — enough to see at a glance
// whether a turn retrieved a branch, re-explored from scratch, or made no calls at all
// (answered from memory, or declined).
func formatToolCalls(calls []ToolCallMetric) string {
	if len(calls) == 0 {
		return "(none)"
	}
	parts := make([]string, 0, len(calls))
	for _, c := range calls {
		parts = append(parts, c.Name+"("+formatArgs(c.Args)+")")
	}
	return strings.Join(parts, ", ")
}

// formatArgs renders a tool call's args as "key=value, key2=value2", sorted by key for
// stable output, truncating any single value over 60 chars so a large write_file content
// argument doesn't blow out the report.
func formatArgs(args map[string]interface{}) string {
	if len(args) == 0 {
		return ""
	}
	keys := make([]string, 0, len(args))
	for k := range args {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		v := fmt.Sprintf("%v", args[k])
		if len(v) > 60 {
			v = v[:60] + "..."
		}
		parts = append(parts, k+"="+v)
	}
	return strings.Join(parts, ", ")
}
