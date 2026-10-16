package experiment

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// Options configures a Run.
type Options struct {
	ConfigPath  string // for provenance in run.json; not re-read during the run
	ResultsRoot string // e.g. <module>/experiments/results
	ModuleRoot  string // ai-sdk-go repo root — where `go build ./examples/tui-agent` runs
}

// RunMeta is written to run.json — everything needed to find, clean, or re-report a run later
// without re-reading the original YAML (which may have changed since).
type RunMeta struct {
	RunID      string    `json:"run_id"`
	ConfigName string    `json:"config_name"`
	ConfigPath string    `json:"config_path"`
	StartedAt  time.Time `json:"started_at"`
	FinishedAt time.Time `json:"finished_at"`
	SourceType string    `json:"source_type"`
	SourceRepo string    `json:"source_repo,omitempty"`
	GitSHA     string    `json:"git_sha,omitempty"` // the exact commit every case started from
	Cases      []string  `json:"cases"`
	Worktrees  []string  `json:"worktrees,omitempty"` // for clean to remove alongside the run dir
}

// Run executes every case in cfg sequentially — not concurrently: cases may share one local
// model server, and interleaving them would confound timing and compaction behavior in ways
// that are hard to untangle from a transcript afterward. Each case starts from the same
// checkpoint: a freshly built tui-agent binary (so a run always tests the CURRENT code, never
// a stale binary left over from an earlier session) and, for a git source, a brand new
// worktree pinned to the same resolved commit — never a worktree left over from a previous
// case or a previous run of this same config.
func Run(ctx context.Context, cfg *Config, opts Options) (string, error) {
	runID := time.Now().UTC().Format("20060102-150405")
	runDir := filepath.Join(opts.ResultsRoot, runID)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		return "", err
	}

	binPath := filepath.Join(runDir, "tui-agent")
	if err := buildBinary(opts.ModuleRoot, binPath); err != nil {
		return runID, fmt.Errorf("build tui-agent: %w", err)
	}

	meta := RunMeta{
		RunID:      runID,
		ConfigName: cfg.Name,
		ConfigPath: opts.ConfigPath,
		StartedAt:  time.Now().UTC(),
		SourceType: cfg.Source.Type,
		SourceRepo: cfg.Source.Repo,
	}

	if cfg.Source.Type == "git" {
		sha, err := resolveRef(cfg.Source.Repo, cfg.Source.Ref)
		if err != nil {
			return runID, fmt.Errorf("resolve git ref %q: %w", cfg.Source.Ref, err)
		}
		meta.GitSHA = sha
		fmt.Printf("source pinned to %s (%s)\n", sha, cfg.Source.Ref)
	}

	// Merge each case's flags once, up front, sequentially — cheap, and every later step
	// (grouping by model, writing flags.json, actually running) reads from this instead of
	// re-merging. Worktree creation happens here too, one at a time, before any concurrent
	// work starts — not because concurrent `git worktree add` is known to be unsafe, but
	// because there's no benefit to parallelizing a few seconds of setup against the risk of
	// finding out it isn't. This whole loop is single-goroutine, so meta.Cases/Worktrees need
	// no lock here — only the later per-case goroutines run concurrently, and they never touch
	// meta.
	merged := make(map[string]map[string]interface{}, len(cfg.Cases))
	for _, c := range cfg.Cases {
		caseDir := filepath.Join(runDir, c.Name)
		if err := os.MkdirAll(caseDir, 0o755); err != nil {
			return runID, err
		}
		m := cfg.merged(c)
		merged[c.Name] = m

		if cfg.Source.Type == "git" {
			workDir := filepath.Join(caseDir, "run-workdir")
			if err := addWorktree(cfg.Source.Repo, meta.GitSHA, workDir); err != nil {
				return runID, fmt.Errorf("case %s: %w", c.Name, err)
			}
			meta.Worktrees = append(meta.Worktrees, workDir)
		}
		if err := writeJSON(filepath.Join(caseDir, "flags.json"), m); err != nil {
			return runID, fmt.Errorf("case %s: write flags.json: %w", c.Name, err)
		}
		meta.Cases = append(meta.Cases, c.Name)
	}

	// Group cases by their already-resolved model and run groups one at a time — different
	// models must never run concurrently against one local GPU serving all of them (loading a
	// second model while the first is still mid-run would thrash VRAM, or fail outright).
	// Cases WITHIN a group share a model, so running them concurrently costs no reload at all
	// — Ollama serves concurrent requests to an already-loaded model directly.
	groups := make(map[string][]Case, len(cfg.Cases))
	for _, c := range cfg.Cases {
		model := fmt.Sprint(merged[c.Name]["model"])
		groups[model] = append(groups[model], c)
	}
	modelNames := make([]string, 0, len(groups))
	for m := range groups {
		modelNames = append(modelNames, m)
	}
	sort.Strings(modelNames) // deterministic group order run to run

	for _, model := range modelNames {
		fmt.Printf("=== model group: %s ===\n", model)
		group := groups[model]

		var wg sync.WaitGroup
		for _, c := range group {
			wg.Add(1)
			go func(c Case) {
				defer wg.Done()
				caseDir := filepath.Join(runDir, c.Name)
				fmt.Printf("=== case %s ===\n", c.Name)
				if err := runCase(ctx, binPath, caseDir, merged[c.Name], cfg.Turns); err != nil {
					// A case-level failure (binary wouldn't start at all, etc.) is logged and
					// the run moves on rather than aborting the whole comparison — a failure
					// mid-turn inside a case (ErrMaxIterations, a provider error) is NOT this:
					// that's recorded in metrics.jsonl and surfaces in the report, and runCase
					// keeps going on its own either way since one bad turn doesn't stop the
					// rest of that same case's turns from running.
					fmt.Fprintf(os.Stderr, "case %s: %v\n", c.Name, err)
				}
			}(c)
		}
		wg.Wait()
	}

	meta.FinishedAt = time.Now().UTC()
	if err := writeJSON(filepath.Join(runDir, "run.json"), meta); err != nil {
		return runID, err
	}
	return runID, nil
}

func buildBinary(moduleRoot, outPath string) error {
	cmd := exec.Command("go", "build", "-o", outPath, "./examples/tui-agent")
	cmd.Dir = moduleRoot
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("%w: %s", err, out)
	}
	return nil
}

// runCase runs every turn against one tui-agent process invocation each (one-shot mode, so
// each turn is its own process — exactly how every experiment this project has run by hand
// worked, now scripted), appending to a shared transcript and metrics file, and snapshotting
// the session after each turn so the report can chart context growth turn by turn.
func runCase(ctx context.Context, binPath, caseDir string, merged map[string]interface{}, turns []Turn) error {
	transcript, err := os.Create(filepath.Join(caseDir, "transcript.log"))
	if err != nil {
		return err
	}
	defer transcript.Close()

	const sessionName = "run"
	metricsPath := filepath.Join(caseDir, "metrics.jsonl")

	baseArgs := toFlags(merged)
	baseArgs = append(baseArgs,
		"--session-dir", caseDir,
		"--session", sessionName,
		"--metrics-file", metricsPath,
	)

	sessionFile := filepath.Join(caseDir, sessionName+".json")
	for i, turn := range turns {
		fmt.Fprintf(transcript, "\n### turn %d ###\n", i+1)
		args := append(append([]string{}, baseArgs...), turn.Prompt)
		cmd := exec.CommandContext(ctx, binPath, args...)
		cmd.Stdout = transcript
		cmd.Stderr = transcript
		if err := cmd.Run(); err != nil {
			// Only a real process-level failure lands here (binary missing, panic, killed) —
			// an agent-level error (max iterations, provider error) exits 0 and is recorded in
			// metrics.jsonl by tui-agent itself instead. Note it and keep going; later turns
			// in this case may still run fine against whatever session state survived.
			//
			// tui-agent never got to write its own metrics.jsonl line for this turn, since the
			// process died before reaching that code — write one here instead. Without this,
			// report.go's `for i, m := range metrics` would treat line i as turn i+1 for every
			// line that DOES exist, silently misattributing every turn after this one to the
			// wrong number in the report (wrong turn blamed for an error, forbidden-tool
			// checks matched against the wrong turn's rules).
			fmt.Fprintf(transcript, "\n### turn %d process error: %v ###\n", i+1, err)
			if werr := appendMetricsLine(metricsPath, TurnMetric{Error: "process error: " + err.Error()}); werr != nil {
				fmt.Fprintf(transcript, "\n### turn %d: also failed to record a placeholder metrics line: %v ###\n", i+1, werr)
			}
		}
		if err := copyFile(sessionFile, filepath.Join(caseDir, fmt.Sprintf("turn-%d.json", i+1))); err != nil {
			fmt.Fprintf(transcript, "\n### turn %d: could not snapshot session: %v ###\n", i+1, err)
		}
	}
	return nil
}

// appendMetricsLine appends one JSON line to a case's metrics.jsonl — the runner's own
// equivalent of examples/tui-agent's appendMetrics, used only for the placeholder line written
// when tui-agent's process dies before it can write its own.
func appendMetricsLine(path string, m TurnMetric) error {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewEncoder(f).Encode(m)
}

func writeJSON(path string, v interface{}) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func readJSON(path string, v interface{}) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, v)
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, in)
	return err
}
