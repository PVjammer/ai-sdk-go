# CLAUDE.md — ai-sdk-go

## Overview

Go SDK providing an AI agent framework with a unified CLI (`ai`). Functions auto-register, are configurable via TOML, and work as CLI commands, LLM tools, or Go API calls.

## Development Commands

```bash
# Build everything
go build ./...

# Install `ai` binary to ~/.local/bin
make install

# Run tests
go test ./...

# Lint
go vet ./...

# Run the chat example
go run ./examples/chat

# Run a function via CLI
echo "some text" | ai summarize
ai search --query "golang tutorials" --max_results 5
```

## Architecture

```
cmd/ai/              CLI entrypoint — cobra root, init, config, function subcommands
pkg/agent/           Agent loop: Plan → Act → Respond (event-driven via channels)
pkg/agent/hooks/     Built-in AgentHooks: LargeResult, Logging, Combine
pkg/planner/         SimplePlanner (USE_TOOL: pattern) and ReActPlanner (JSON)
pkg/llm/             Provider interface + OllamaProvider (streaming, conversation)
pkg/tools/           ToolManager — registers AIFunctions for agent use
pkg/function/        AIFunction interface, Base, Config, Schema, Metadata
pkg/registry/        Global function registry with ConfigParam declarations
pkg/config/          TOML AppConfig + Viper bridge (BridgeToViper)
pkg/cli/             Auto-generates Cobra commands from AIFunction InputSchema
pkg/agentcontext/    InMemory and Extended context with compression hooks
pkg/builtin/         Built-in functions: summarize, websearch, extract, text_transform
examples/chat/       Interactive chat REPL demonstrating the agent
examples/personal-assistant/  Extended context with personalization and compression
```

## Config System

Config file: `~/.config/ai/config.toml`

```toml
[llm]
provider = "ollama"
endpoint = "http://localhost:11434"
model = "gpt-oss:20b"

[agent]
max_iterations = 10
temperature = 0.7
max_tokens = 2048

[functions.summarize]
enabled_cli = true
enabled_tool = true
requires_approval = false
max_length = 500
style = "bullets"
llm_provider = ""          # empty = fall back to global [llm]
llm_endpoint = ""
llm_model = "qwen3-64k:latest"
```

### Config priority chains

- **Input defaults:** hardcoded < TOML config (via Viper) < per-call CLI flag/argument
- **LLM provider:** per-function Viper key → global Viper key → hardcoded default

### How config flows

1. `LoadAppConfig` parses TOML into `AppConfig` struct
2. `BridgeToViper` injects all values into Viper (`summarize.style`, `llm.provider`, etc.)
3. Functions read via `f.Config().GetString("summarize.style")` at execution time
4. `_changed_flags` metadata in CLI input maps distinguishes user-set flags from Cobra defaults
5. LLM providers are resolved lazily in `Execute()` (not constructor) so BridgeToViper has run

## Key Patterns

### Function registration
Functions auto-register via `init()` in `pkg/builtin/*/register.go`:
```go
func init() {
    registry.Global().RegisterWithParams("summarize", New, []registry.ConfigParam{...})
}
```

### Factory pattern
All functions accept `function.Config`: `New(cfg function.Config) function.AIFunction`

### Agent event streaming
Agent.Run() returns `<-chan Event` with types: EventThinking, EventToolCall, EventToolResult, EventResponse, EventError, EventComplete.

### Agent hooks
`AgentBase` supports lifecycle hooks via `agent.SetHooks(hooks AgentHooks)`. Six hook points fire in this order per Run():

| Hook | When | Can modify |
|---|---|---|
| `PreInput` | before user message → history | message string |
| `PreModel` | before each LLM call | message slice |
| `PostModel` | after LLM response | StreamChunk |
| `PreTool` | after EventToolCall, before execution | ToolCall args + can deny |
| `PostTool` | after execution, before → history | result string (key for large-result handling) |
| `PostResponse` | after final response → history | (observe only, errors non-fatal) |

`EventToolResult` is emitted with the **raw** output before `PostTool` runs, so the UI always sees the real result. Only the model's history gets the PostTool-modified string.

Built-in hooks in `pkg/agent/hooks/`:
- `LargeResult(threshold, dir)` — redirects large results to temp files the model can read_file in chunks
- `Logging(w)` — logs all hook points to an io.Writer (dev/debug)
- `Combine(hooks...)` — composes multiple hooks into a pipeline

```go
import "github.com/pvjammer/ai-sdk-go/pkg/agent/hooks"

myAgent.SetHooks(hooks.Combine(
    hooks.Logging(os.Stderr),
    hooks.LargeResult(8192, ""),
))
```

### Planner tool parsing
SimplePlanner parses `USE_TOOL: tool_name {"param": "value"}` from raw LLM output. Works with any model (no native function calling required).

### CLI auto-generation
`pkg/cli/builder.go` generates Cobra commands from function InputSchema — flags, stdin piping, help text all derived from struct tags.

## Adding a New Built-in Function

1. Create `pkg/builtin/myfunction/myfunction.go` — implement `function.AIFunction`
2. Create `pkg/builtin/myfunction/register.go` — `init()` calls `registry.Global().RegisterWithParams()`
3. Add `_ "github.com/pvjammer/ai-sdk-go/pkg/builtin/myfunction"` to `cmd/ai/main.go` imports
4. Run `ai init` to regenerate config with new function's params

## Testing

```bash
# All tests
go test ./...

# Specific package
go test ./pkg/cli/

# With verbose output
go test -v ./pkg/config/
```

Tests use `NewWithProvider(mockProvider)` to bypass real LLM calls. The `llm.NewMockProvider()` returns canned responses for testing.

## Dependencies

Go >= 1.24, Cobra, Viper, go-toml/v2, Ollama Go SDK, charmbracelet/huh (TUI), go-playground/validator
