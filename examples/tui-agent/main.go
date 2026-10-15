// Command tui-agent is a small terminal test harness for the agent-loop-v2 work
// (docs/design/{session-dispatch,compaction-v2,turn-branching}.md): Session with steering,
// turn-scratch branching over a SessionLog, and LLM-summarization compaction, all wired
// together against a real OpenAI-compatible endpoint. Not a polished product — just enough UI
// to exercise and watch these mechanisms work.
//
// One-shot, straight from the command line (a message as a positional arg runs exactly one turn
// and exits — the same pattern cmd/ai/chat.go uses):
//
//	export AI_SDK_API_KEY=...    # optional — most local servers don't need one
//	go run ./examples/tui-agent --session alice "what time is it?"
//	go run ./examples/tui-agent --session alice "and what did I just ask you?"
//
// Interactive REPL (omit the message):
//
//	go run ./examples/tui-agent --session alice
//
// Every turn, in either mode, is saved to --session-dir (default
// ~/.local/share/ai-sdk-go/tui-agent/sessions/<session>.json) and reloaded at startup — a later
// invocation with the same --session continues it. Persistence follows --branching: when
// branching is on, the whole SessionLog (every branch, not just the trunk) is saved, so
// read_branch still works on branches created in an earlier process; when it's off, it's
// AgentBase's own flat history, which is the only state that path actually uses. Switching
// --branching for a --session that was saved under the other mode starts that session fresh
// (a warning is printed) rather than guessing how to convert one shape into the other.
//
// REPL-only commands (steering and follow-up are inherently "inject mid-turn" — they don't mean
// anything across separate process invocations, so one-shot mode doesn't have them):
//
//	/quit                 exit
//	/steer <text>         inject <text> mid-turn, at the next tool-call boundary
//	/followup <text>      queue <text> to run as a new turn once the current one finishes
//	/log                  print the trunk's current path (one line per entry)
//	/branch <id>          print a branch's full content by its entry ID (the ref: id shown in /log)
//	anything else         sent to the agent as a normal message
//
// Tools available to the model: get_time, write_note/read_note (a session-scoped notes
// directory, persisted alongside the session — write_note/read_note's names feed
// DefaultFileOpsExtractor so a [compacted] line's read/modified file lists have something real
// to show), and list_branches/read_branch when branching is on. A [compacted] line prints live whenever the
// Compaction hook actually fires.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/pvjammer/ai-sdk-go/pkg/agent"
	"github.com/pvjammer/ai-sdk-go/pkg/agent/hooks"
	"github.com/pvjammer/ai-sdk-go/pkg/function"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// tuiAgent is the concrete AIAgent this example runs — AgentBase plus the Metadata/Config/Run
// methods it doesn't provide on its own (see pkg/builtin/agents/simple for the same pattern).
type tuiAgent struct {
	agent.AgentBase
	branchAware bool // true when branching is on and read_branch is actually registered
}

func (a *tuiAgent) Metadata() *agent.AgentMetadata {
	return &agent.AgentMetadata{Name: "TUI Agent", CLIName: "tui-agent"}
}
func (a *tuiAgent) Config() function.Config { return function.NilConfig{} }
func (a *tuiAgent) Run(ctx context.Context, sessionID, message string) <-chan agent.Event {
	return a.BaseRun(ctx, sessionID, message, a.systemPrompt(), 15, false)
}

func (a *tuiAgent) systemPrompt() string {
	p := "You are a helpful assistant being used to test an agent SDK. Use the " +
		"available tools when they'd genuinely help; don't call a tool just to demonstrate it."
	if !a.branchAware {
		return p
	}
	// Round 1 (bare "ref:<id>", no explanation) failed outright: asked for a detail it had
	// deliberately kept out of its own final answer, the model said it didn't have it, rather
	// than considering retrieval at all — on both a small and a large local model, so this read
	// as a missing affordance, not a capability gap worth solving with a bigger model. Round 2
	// added a breadcrumb and this same paragraph, keyed on the raw ID as the thing to use —
	// which produced a NEW failure: a model that did attempt retrieval copied a literal example
	// ID from this very prompt instead of a real one from its own context. Round 3 (current):
	// short sequential labels instead of raw IDs, and list_branches as the primary discovery
	// tool instead of leaning on the model noticing and re-parsing inline breadcrumb text.
	return p + " Some of your own earlier conversation history has been moved out of view to " +
		"keep this context small, to make room for more conversation. If you need a specific " +
		"fact, value, or detail from earlier and you don't see it stated plainly in your " +
		"current context, call list_branches BEFORE concluding you don't have the information " +
		"or redoing the original work — it shows you everything that's been moved out of view " +
		"and a short label (like \"B2\") for each one. Then call read_branch with that exact " +
		"label to pull back the full original exchange. Never guess, invent, or reuse a label " +
		"or ID from an example — only ones list_branches actually shows you are real."
}

func defaultSessionDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ".tui-agent-sessions"
	}
	return filepath.Join(home, ".local", "share", "ai-sdk-go", "tui-agent", "sessions")
}

func main() {
	endpoint := flag.String("endpoint", "http://192.168.1.88:4000/v1", "OpenAI-compatible endpoint")
	model := flag.String("model", "Qwen3.6-35B-A3B", "model name as the endpoint reports it")
	branching := flag.Bool("branching", true, "enable turn-scratch branching (docs/design/turn-branching.md)")
	compact := flag.Bool("compaction", true, "enable LLM-summarization compaction (docs/design/compaction-v2.md)")
	contextWindow := flag.Int("context-window", 128000, "the model's REAL served context size — verify this against the "+
		"endpoint/server, don't assume the architecture's advertised max (see ai-sdk-go session notes on Ollama's "+
		"num_ctx vs. a model's reported context_length). Compaction's protected budget is a fraction of this, not a "+
		"flat number, so it scales with whatever model you actually point --endpoint/--model at.")
	compactionFraction := flag.Float64("compaction-fraction", 0.75, "fraction of --context-window compaction's "+
		"protected budget uses (pi's analogous default reserves a fixed ~16K-token margin instead of a fraction; "+
		"a fraction is simpler to reason about and scales the same way across very different context sizes)")
	experimentTools := flag.Bool("experiment-tools", false, "register lookup_fact, a test-only tool for the "+
		"branching-on-vs-off comparison (see experiment_tools.go)")
	codingTools := flag.Bool("coding-tools", false, "register write_file/read_file/list_files/run_go_test, "+
		"scoped to a per-session scratch Go module, for a real multi-turn implementation task (see coding_tools.go)")
	sessionID := flag.String("session", "tui", "session identifier — reused across invocations to continue it")
	sessionDir := flag.String("session-dir", defaultSessionDir(), "directory session files are saved/loaded from")
	flag.Parse()
	oneShotMessage := strings.Join(flag.Args(), " ")

	apiKey := os.Getenv("AI_SDK_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("DGX_API_KEY")
	}

	provider, err := llm.NewOpenAIProvider(*endpoint, *model, llm.WithAPIKey(apiKey))
	if err != nil {
		fmt.Fprintf(os.Stderr, "provider: %v\n", err)
		os.Exit(1)
	}

	a := &tuiAgent{branchAware: *branching}
	a.InitBase(function.NilConfig{})
	a.SetProvider(provider)

	if err := a.RegisterTool(newGetTimeTool()); err != nil {
		fmt.Fprintf(os.Stderr, "register get_time: %v\n", err)
		os.Exit(1)
	}

	// Session-scoped, not a one-off temp dir: notes need to survive across invocations just
	// like the conversation does, or "continuing a session" would silently lose them.
	notesDir := filepath.Join(*sessionDir, *sessionID+"-notes")
	if err := os.MkdirAll(notesDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "mkdir notes dir: %v\n", err)
		os.Exit(1)
	}
	if err := a.RegisterTool(newWriteNoteTool(notesDir)); err != nil {
		fmt.Fprintf(os.Stderr, "register write_note: %v\n", err)
		os.Exit(1)
	}
	if err := a.RegisterTool(newReadNoteTool(notesDir)); err != nil {
		fmt.Fprintf(os.Stderr, "register read_note: %v\n", err)
		os.Exit(1)
	}
	if *experimentTools {
		if err := a.RegisterTool(newLookupFactTool(notesDir)); err != nil {
			fmt.Fprintf(os.Stderr, "register lookup_fact: %v\n", err)
			os.Exit(1)
		}
	}
	if *codingTools {
		workDir := filepath.Join(*sessionDir, *sessionID+"-workdir")
		if err := initCodingWorkdir(workDir); err != nil {
			fmt.Fprintf(os.Stderr, "init coding workdir: %v\n", err)
			os.Exit(1)
		}
		for _, err := range []error{
			a.RegisterTool(newWriteFileTool(workDir)),
			a.RegisterTool(newReadFileTool(workDir)),
			a.RegisterTool(newListFilesTool(workDir)),
			a.RegisterTool(newRunGoTestTool(workDir)),
		} {
			if err != nil {
				fmt.Fprintf(os.Stderr, "register coding tool: %v\n", err)
				os.Exit(1)
			}
		}
	}

	loaded, found := loadSession(*sessionDir, *sessionID)
	if found && loaded.branching != *branching {
		fmt.Fprintf(os.Stderr,
			"warning: session %q was saved with branching=%v, this run has branching=%v — starting fresh\n",
			*sessionID, loaded.branching, *branching)
		found = false
	}

	var log *agent.SessionLog
	leaf := ""
	if *branching {
		if found {
			log = agent.LoadSessionLog(loaded.entries)
			leaf = loaded.leaf
		} else {
			log = agent.NewSessionLog()
		}
		if err := a.RegisterTool(newReadBranchTool(log)); err != nil {
			fmt.Fprintf(os.Stderr, "register read_branch: %v\n", err)
			os.Exit(1)
		}
		if err := a.RegisterTool(newListBranchesTool(log)); err != nil {
			fmt.Fprintf(os.Stderr, "register list_branches: %v\n", err)
			os.Exit(1)
		}
	} else if found {
		a.SeedSession(*sessionID, loaded.history)
	}

	// compactionBudget is the protected budget compaction triggers against — a FRACTION of the
	// model's real context window, not a flat number, so it scales with whatever you actually
	// point --endpoint/--model at instead of silently becoming a tiny sliver of a 96K+ model's
	// real room (see ai-sdk-go session notes: the old flat 4000-token default was triggering
	// compaction on nearly every round for any model with real headroom, which is what caused
	// the repeated-compaction tool-call loop found during stress testing).
	compactionBudget := int(float64(*contextWindow) * *compactionFraction)
	// hardLimit is the backstop if compaction can't keep up — just under the model's REAL
	// ceiling (not a multiple of compactionBudget, which would exceed contextWindow and defeat
	// the point of a backstop), leaving a little headroom for the response itself.
	hardLimit := int(float64(*contextWindow) * 0.95)

	var hs []agent.AgentHooks
	if *compact {
		hs = append(hs, hooks.Compaction(&a.AgentBase, provider, hooks.CompactionOptions{
			MaxTokens: compactionBudget,
			OnCompaction: func(sessionID string, result hooks.CompactionResult) {
				fmt.Printf("\n[compacted] %d -> ~%d tokens", result.TokensBefore, result.EstimatedTokensAfter)
				if len(result.Details.ReadFiles) > 0 {
					fmt.Printf("  read: %v", result.Details.ReadFiles)
				}
				if len(result.Details.ModifiedFiles) > 0 {
					fmt.Printf("  modified: %v", result.Details.ModifiedFiles)
				}
				fmt.Println()
			},
		}))
	}
	hs = append(hs, hooks.TokenLimit(hardLimit))
	hs = append(hs, hooks.LargeResult(8192, "", "read_branch"))
	a.SetHooks(hooks.Combine(hs...))

	sess := agent.NewSession(*sessionID, a, agent.SessionConfig{Log: log, Leaf: leaf})

	persist := func() {
		if err := saveSession(*sessionDir, *sessionID, *branching, sess, a); err != nil {
			fmt.Fprintf(os.Stderr, "warning: could not save session: %v\n", err)
		}
	}

	if oneShotMessage != "" {
		fmt.Fprintf(os.Stderr, "[config] model=%s branching=%v compaction=%v budget=%d window=%d\n",
			*model, *branching, *compact, compactionBudget, *contextWindow)
		fmt.Print("agent> ")
		runTurn(context.Background(), sess, oneShotMessage)
		persist()
		return
	}

	fmt.Printf("=== TUI Agent === endpoint=%s model=%s branching=%v compaction=%v (budget=%d/window=%d) session=%s\n",
		*endpoint, *model, *branching, *compact, compactionBudget, *contextWindow, *sessionID)
	if found {
		fmt.Println("(continuing a saved session)")
	}
	fmt.Println("Type /quit to exit, or a message to chat. See the file header for other commands.")
	fmt.Println()

	runREPL(sess, log, persist)
}

func runREPL(sess *agent.Session, log *agent.SessionLog, persist func()) {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	ctx := context.Background()

	for {
		fmt.Print("you> ")
		if !scanner.Scan() {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case line == "/quit" || line == "/exit":
			return

		case strings.HasPrefix(line, "/steer "):
			sess.Steer(strings.TrimPrefix(line, "/steer "))
			fmt.Println("[steered]")
			continue

		case strings.HasPrefix(line, "/followup "):
			sess.SendFollowUp(strings.TrimPrefix(line, "/followup "))
			fmt.Println("[follow-up queued]")
			continue

		case line == "/log":
			printLog(log, sess.Leaf())
			continue

		case strings.HasPrefix(line, "/branch "):
			printBranch(log, strings.TrimPrefix(line, "/branch "))
			continue
		}

		fmt.Print("agent> ")
		runTurn(ctx, sess, line)
		persist()
	}
}

func runTurn(ctx context.Context, sess *agent.Session, message string) {
	for ev := range sess.Send(ctx, message) {
		switch ev.Type {
		case agent.EventToolCall:
			tc := ev.Data.(agent.ToolCallEvent)
			fmt.Printf("\n  [tool call] %s(%v)\n", tc.ToolName, tc.Input)

		case agent.EventToolResult:
			tr := ev.Data.(agent.ToolResultEvent)
			if tr.Error != nil {
				fmt.Printf("  [tool result] %s: error: %v\n", tr.ToolName, tr.Error)
			} else {
				fmt.Printf("  [tool result] %s: %v\n", tr.ToolName, tr.Output)
			}

		case agent.EventResponse:
			r := ev.Data.(agent.ResponseEvent)
			if r.Done {
				fmt.Println(r.Content)
			}

		case agent.EventError:
			ee := ev.Data.(agent.ErrorEvent)
			fmt.Fprintf(os.Stderr, "\n[error] %v\n", ee.Error)
		}
	}
}

func printLog(log *agent.SessionLog, leaf string) {
	if log == nil {
		fmt.Println("(branching disabled — no log)")
		return
	}
	path := log.Path(leaf)
	if len(path) == 0 {
		fmt.Println("(empty)")
		return
	}
	for _, e := range path {
		switch {
		case e.Role == "tool" && e.ToolName == "turn_scratch":
			id, _ := agent.ParseBranchRef(e.Content)
			fmt.Printf("  [%s] branch reference -> %s  (see: /branch %s)\n", e.Role, e.Content, id)
		default:
			preview := e.Content
			if len(preview) > 100 {
				preview = preview[:100] + "..."
			}
			fmt.Printf("  [%s] %s\n", e.Role, preview)
		}
	}
}

func printBranch(log *agent.SessionLog, branchID string) {
	if log == nil {
		fmt.Println("(branching disabled — no log)")
		return
	}
	subtree := log.Subtree(branchID)
	if len(subtree) == 0 {
		fmt.Printf("(no such branch: %s)\n", branchID)
		return
	}
	for _, e := range subtree {
		preview := e.Content
		if len(preview) > 200 {
			preview = preview[:200] + "..."
		}
		fmt.Printf("  [%s] %s\n", e.Role, preview)
	}
}

// ── persistence ──────────────────────────────────────────────────────────────

// sessionFile is the on-disk shape for one --session. Only one of Entries (branching on) or
// History (branching off) is ever populated — see main()'s loadSession/saveSession call sites.
type sessionFile struct {
	Branching bool              `json:"branching"`
	Leaf      string            `json:"leaf,omitempty"`
	Entries   []agent.Entry     `json:"entries,omitempty"`
	History   []llm.ChatMessage `json:"history,omitempty"`
}

type loadedSession struct {
	branching bool
	leaf      string
	entries   []agent.Entry
	history   []llm.ChatMessage
}

func sessionFilePath(dir, sessionID string) string {
	return filepath.Join(dir, sessionID+".json")
}

func loadSession(dir, sessionID string) (loadedSession, bool) {
	data, err := os.ReadFile(sessionFilePath(dir, sessionID))
	if err != nil {
		return loadedSession{}, false
	}
	var sf sessionFile
	if err := json.Unmarshal(data, &sf); err != nil {
		fmt.Fprintf(os.Stderr, "warning: could not parse session file %q, starting fresh: %v\n", sessionFilePath(dir, sessionID), err)
		return loadedSession{}, false
	}
	return loadedSession{branching: sf.Branching, leaf: sf.Leaf, entries: sf.Entries, history: sf.History}, true
}

func saveSession(dir, sessionID string, branching bool, sess *agent.Session, a *tuiAgent) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	sf := sessionFile{Branching: branching}
	if branching {
		sf.Leaf = sess.Leaf()
		sf.Entries = sess.Log().Entries()
	} else {
		sf.History = a.SessionHistory(sessionID)
	}
	data, err := json.MarshalIndent(sf, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(sessionFilePath(dir, sessionID), data, 0o644)
}

// ── tools ────────────────────────────────────────────────────────────────────

type getTimeTool struct{ function.Base }

func newGetTimeTool() *getTimeTool {
	return &getTimeTool{Base: function.NewBase(function.NilConfig{})}
}
func (t *getTimeTool) Execute(ctx context.Context, _ interface{}) (interface{}, error) {
	return time.Now().UTC().Format("2006-01-02 15:04:05 MST"), nil
}
func (t *getTimeTool) InputSchema() *function.Schema {
	return &function.Schema{Type: "object", Properties: map[string]*function.Property{}}
}
func (t *getTimeTool) Metadata() *function.Metadata {
	m := function.NewMetadata("get_time")
	m.Description = "Returns the current date and time."
	return m
}

// writeNoteTool and readNoteTool are a tiny read/write pair confined to a session-scoped
// directory, purely so DefaultFileOpsExtractor (compaction.go) has real file operations to
// track — "write_note"/"read_note" match its write/read naming convention, and a "path" arg.

type writeNoteTool struct {
	function.Base
	dir string
}

func newWriteNoteTool(dir string) *writeNoteTool {
	return &writeNoteTool{Base: function.NewBase(function.NilConfig{}), dir: dir}
}
func (t *writeNoteTool) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	args, _ := input.(map[string]interface{})
	path, _ := args["path"].(string)
	content, _ := args["content"].(string)
	if path == "" {
		return nil, fmt.Errorf("path is required")
	}
	if err := os.WriteFile(t.dir+"/"+path, []byte(content), 0o644); err != nil {
		return nil, err
	}
	return fmt.Sprintf("wrote %d bytes to %s", len(content), path), nil
}
func (t *writeNoteTool) InputSchema() *function.Schema {
	return &function.Schema{
		Type: "object",
		Properties: map[string]*function.Property{
			"path":    {Type: "string", Description: "Note filename, e.g. \"ideas.md\"."},
			"content": {Type: "string", Description: "Text to write."},
		},
		Required: []string{"path", "content"},
	}
}
func (t *writeNoteTool) Metadata() *function.Metadata {
	m := function.NewMetadata("write_note")
	m.Description = "Write a text note to a scratch notes directory for this session."
	return m
}

type readNoteTool struct {
	function.Base
	dir string
}

func newReadNoteTool(dir string) *readNoteTool {
	return &readNoteTool{Base: function.NewBase(function.NilConfig{}), dir: dir}
}
func (t *readNoteTool) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	args, _ := input.(map[string]interface{})
	path, _ := args["path"].(string)
	if path == "" {
		return nil, fmt.Errorf("path is required")
	}
	content, err := os.ReadFile(t.dir + "/" + path)
	if err != nil {
		return nil, err
	}
	return string(content), nil
}
func (t *readNoteTool) InputSchema() *function.Schema {
	return &function.Schema{
		Type: "object",
		Properties: map[string]*function.Property{
			"path": {Type: "string", Description: "Note filename to read back."},
		},
		Required: []string{"path"},
	}
}
func (t *readNoteTool) Metadata() *function.Metadata {
	m := function.NewMetadata("read_note")
	m.Description = "Read back a note previously written with write_note."
	return m
}

// readBranchTool lets the model retrieve an earlier turn-scratch branch's full content by the
// ID shown in its tool-message reference — see docs/design/turn-branching.md.
type readBranchTool struct {
	function.Base
	log *agent.SessionLog
}

func newReadBranchTool(log *agent.SessionLog) *readBranchTool {
	return &readBranchTool{Base: function.NewBase(function.NilConfig{}), log: log}
}
func (t *readBranchTool) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	args, _ := input.(map[string]interface{})
	id, _ := args["branch_id"].(string)
	if id == "" {
		return nil, fmt.Errorf("branch_id is required")
	}
	rootID, ok := agent.ResolveBranch(t.log, id)
	if !ok {
		return nil, fmt.Errorf("no branch found for %q — call list_branches to see valid labels", id)
	}
	subtree := t.log.Subtree(rootID)
	if len(subtree) == 0 {
		return nil, fmt.Errorf("no branch found for %q", id)
	}
	var sb strings.Builder
	for _, e := range subtree {
		fmt.Fprintf(&sb, "%s: %s\n", e.Role, e.Content)
	}
	return sb.String(), nil
}
func (t *readBranchTool) InputSchema() *function.Schema {
	return &function.Schema{
		Type: "object",
		Properties: map[string]*function.Property{
			"branch_id": {Type: "string", Description: "A branch label like \"B2\" — call list_branches first " +
				"if you don't already know which one you need. Never guess or invent a label or ID."},
		},
		Required: []string{"branch_id"},
	}
}
func (t *readBranchTool) Metadata() *function.Metadata {
	m := function.NewMetadata("read_branch")
	m.Description = "Retrieve the full original content of an earlier turn's hidden scratch work (every tool call " +
		"and result from that turn, not just its summary) by its branch label (call list_branches to see valid " +
		"labels). Use this whenever you need a specific fact, value, or detail from earlier that isn't plainly " +
		"visible in your current context — check list_branches BEFORE concluding you don't have the information, " +
		"and before redoing the work by calling the original tool again."
	return m
}

// listBranchesTool lets the model discover what's retrievable without having to notice and
// re-parse an inline breadcrumb from potentially-distant or compacted-away history — a tool's
// presence in the tool list doesn't degrade with context length the way re-reading history
// does, so this is the primary, always-available discovery path; the inline breadcrumbs on the
// trunk are a secondary, passive hint for a model that happens to be re-reading nearby history.
type listBranchesTool struct {
	function.Base
	log *agent.SessionLog
}

func newListBranchesTool(log *agent.SessionLog) *listBranchesTool {
	return &listBranchesTool{Base: function.NewBase(function.NilConfig{}), log: log}
}
func (t *listBranchesTool) Execute(ctx context.Context, _ interface{}) (interface{}, error) {
	branches := agent.ListBranches(t.log)
	if len(branches) == 0 {
		return "No branches exist yet in this session.", nil
	}
	var sb strings.Builder
	for _, b := range branches {
		fmt.Fprintf(&sb, "%s: %s\n", b.Label, b.Content)
	}
	return sb.String(), nil
}
func (t *listBranchesTool) InputSchema() *function.Schema {
	return &function.Schema{Type: "object", Properties: map[string]*function.Property{}}
}
func (t *listBranchesTool) Metadata() *function.Metadata {
	m := function.NewMetadata("list_branches")
	m.Description = "List every piece of earlier scratch work (tool calls and results) that's been moved out of " +
		"your visible context, with a short note on what each one contains. Call this whenever you need a fact, " +
		"value, or detail from earlier that you don't see stated plainly in front of you — before assuming it's " +
		"lost, and before redoing the original work. Follow up with read_branch using the label shown here."
	return m
}
