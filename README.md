# ai-sdk-go

A Go SDK for building AI agents: a provider-agnostic LLM interface, an agent loop with
lifecycle hooks, a tool/function registry, and a CLI that auto-generates commands from function
schemas. This branch (`agent-loop-v2`) adds session-level context management on top of that —
turn-scratch branching and LLM-summarization compaction — plus the tooling used to evaluate it
against real models.

## What's on this branch

- **Turn-scratch branching** (`pkg/agent/session.go`, `pkg/agent/sessionlog.go`) — each
  tool-using turn runs on a disposable scratch session seeded from the real conversation so
  far; only a short reference and the final answer get spliced back into the main history,
  with the raw tool-call detail retrievable on demand (`read_branch`/`list_branches`) rather
  than carried forever. See `docs/design/turn-branching.md` for the design,
  `docs/design/branching-evaluation.md` for what happened when it was actually measured against
  real models — the short version: real context and token savings, at the cost of more tool-call
  round trips.
- **Compaction** (`pkg/agent/hooks/compaction.go`) — LLM-summarization compaction with typed,
  structured output (not prose), update-not-append across repeated rounds, non-destructive on
  summarizer failure. See `docs/design/compaction-v2.md`.
- **`examples/tui-agent`** — a terminal harness exercising both of the above against a real
  OpenAI-compatible endpoint, as a one-shot CLI command or an interactive REPL, with session
  persistence across process invocations.
- **`cmd/experiment-runner`** (`internal/experiment`) — scripts reproducible A/B runs of
  `tui-agent` across a matrix of configs (model, branching on/off, context size, replicates)
  from a YAML file, with a fresh git worktree per case for anything touching a real repo, and a
  mechanical (not LLM-judged) report: errors, compaction events, context growth, token usage,
  and the full tool-call sequence per turn. See `experiments/README.md`.

## Architecture

```
cmd/ai/                   CLI entrypoint — cobra root, init, config, function subcommands
cmd/experiment-runner/    Runs experiments/configs/*.yaml — see experiments/README.md
pkg/agent/                Agent loop: Plan -> Act -> Respond (event-driven via channels)
pkg/agent/session.go      Session: steering, follow-up queuing, turn-scratch branching
pkg/agent/sessionlog.go   SessionLog: the tree-structured log branching runs on
pkg/agent/hooks/          Built-in AgentHooks: Compaction, LargeResult, TokenLimit, Logging, Combine
pkg/planner/              SimplePlanner (USE_TOOL: pattern) and ReActPlanner (JSON)
pkg/llm/                  Provider interface + OpenAI-compatible and Ollama-native providers
pkg/tools/                ToolManager — registers AIFunctions for agent use
pkg/function/             AIFunction interface, Base, Config, Schema, Metadata
pkg/registry/             Global function registry with ConfigParam declarations
pkg/config/               TOML AppConfig + Viper bridge (BridgeToViper)
pkg/cli/                  Auto-generates Cobra commands from AIFunction InputSchema
pkg/agentcontext/         InMemory and Extended context with compression hooks
pkg/builtin/              Built-in functions: summarize, websearch, extract, text_transform
internal/experiment/      Config loading, git worktree management, running, reporting
examples/chat/            Interactive chat REPL demonstrating the agent
examples/tui-agent/       Branching + compaction test harness (see above)
examples/personal-assistant/  Extended context with personalization and compression
experiments/              YAML configs + README for cmd/experiment-runner
docs/design/              Design docs: turn-branching, compaction-v2, session-dispatch,
                          working-memory, branching-evaluation (the empirical results)
```

## Development

```bash
# Build everything
go build ./...

# Run tests
go test ./...
go test -race ./...   # branching/session code is concurrency-sensitive; always run with -race

# Lint
go vet ./...

# Install the `ai` binary
make install

# Run the chat example
go run ./examples/chat

# Run the branching/compaction test harness
go run ./examples/tui-agent --session alice "what time is it?"

# Reproduce the experiments in docs/design/branching-evaluation.md
go run ./cmd/experiment-runner run experiments/configs/repo-feature.yaml
go run ./cmd/experiment-runner run experiments/configs/daily-driver.yaml
```

Full CLI usage, config format, and existing patterns (function registration, hooks, the
planner's tool-call parsing) are in `CLAUDE.md` — written for an agent working in this repo, but
equally useful for a human. Note it predates this branch's additions; `docs/design/` is the
source of truth for everything added here.

## Status

Turn-scratch branching and compaction are implemented, tested (including with `-race`), and
evaluated against three real local models on two different task shapes with replication — see
`docs/design/branching-evaluation.md` for the full results and the honest gaps that are still
open. Short version: the branching mechanism itself holds up well; the bigger open risk is that
reliable multi-step coding work with this SDK currently needs a fairly large model (validated
only down to a 27B-class model so far) — smaller models tested did not reliably produce working
code, independent of whether branching was on.
