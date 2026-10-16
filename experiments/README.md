# experiments

Scripted, reproducible A/B runs of `examples/tui-agent` across a matrix of configs (model,
branching on/off, context size, ...), implemented in `internal/experiment` +
`cmd/experiment-runner`.

## Usage

```bash
go run ./cmd/experiment-runner run experiments/configs/repo-feature.yaml
go run ./cmd/experiment-runner list
go run ./cmd/experiment-runner report <run-id>    # regenerate report.md/.json without rerunning
go run ./cmd/experiment-runner clean <run-id>      # remove a run's results AND its git worktrees
go run ./cmd/experiment-runner clean --all
```

Each `run` prints a run ID (`experiments/results/<run-id>/`) and writes `report.md` there.

## Config format

```yaml
name: my-suite
defaults:            # forwarded to tui-agent as CLI flags, one per key
  model: qwen3.8-96k  #   -> --model qwen3.8-96k
  context_window: 98304
  branching: true     # bools use the --key=value form tui-agent's flag package requires

source:
  type: scratch        # a fresh empty directory per case, every run
  # type: git
  # repo: /abs/path/to/a/repo
  # ref: HEAD           # resolved to an exact commit SHA once per run, pinned in run.json —
                         # every case, and every later rerun of this same config, starts from
                         # that exact commit in a brand-new worktree, never a leftover one.

turns:
  - prompt: "..."
  - prompt: "..."
    forbidden_tools: ["some_tool"]   # flags a violation in the report if called on this turn

cases:
  - name: branch-on
    branching: true
  - name: branch-off
    branching: false
    model: a-different-model          # any default can be overridden per case
```

Any YAML key under `defaults` or a case becomes a `--key-with-dashes value` flag to
`examples/tui-agent` — nothing here hardcodes which flags exist, so a new `tui-agent` flag
works from a config immediately.

## What the report covers, and what it doesn't

`report.md`/`report.json` are mechanical only: errors, compaction events, forbidden-tool
violations, context size by turn (entries/chars — computed from the real saved session state,
not scraped from printed text). They do **not** judge whether an answer was actually correct —
that's left to reading the saved `transcript.log`/session files afterward, by a human or a
separate LLM-judge pass.

## Existing suites

- `repo-feature.yaml` — cold-start feature implementation against a real, unfamiliar repo
  (this project itself, via a git worktree). Tests repo-exploration survival and longer spine
  growth. Found two real bugs when this was first run by hand (see
  `pkg/agent/hooks/compaction.go`, `pkg/agent/session.go`).
- `daily-driver.yaml` — many quick single-tool-call lookups (docs, a simulated web search,
  notes) rather than few tool-heavy turns — the "personal assistant" shape.

## Known limits

- Cases run sequentially, not concurrently — intentional: they may share one local model
  server, and interleaving them would confound timing and compaction behavior in ways that are
  hard to untangle from a transcript afterward. It also means mixing models across cases in one
  config is safe (only one is ever actively loaded at a time), just not faster for it.
- A case directory's `run-workdir` is the literal git worktree (or scratch dir) `coding_tools`
  operates in — if you want to inspect the actual code a case produced, it's right there, not
  just in the transcript.
