# Chat CLI Example

A simple interactive chat CLI that demonstrates the AI Agent SDK with automatic tool discovery from the ai-functions-go registry.

## Features

- **Interactive REPL** - Chat with the AI agent in a loop
- **Auto-discovery** - Automatically loads all registered functions from the registry
- **Tool Support** - Agent can use tools like text_transform, summarize, and extract
- **Meta Commands** - Built-in commands for help, listing tools, and more
- **Clean Output** - Shows tool usage inline for transparency

## Prerequisites

1. **Ollama** must be running:
   ```bash
   ollama serve
   ```

2. **llama3.2 model** must be downloaded:
   ```bash
   ollama pull llama3.2:latest
   ```

## Running the Example

```bash
cd examples/chat
go run main.go
```

Or build and run:
```bash
go build
./chat
```

## Usage

### Basic Chat

```
You: Hello! What can you do?
Assistant: I'm an AI assistant with access to several tools...
```

### Using Tools

The agent can automatically use tools when needed:

```
You: Transform 'hello world' to uppercase
Assistant: [Using text_transform] HELLO WORLD
```

### Meta Commands

- `/help` - Show available commands
- `/tools` - List all registered tools with descriptions
- `/clear` - Clear conversation history
- `/quit` or `/exit` - Exit the chat

### Example Session

```
$ go run main.go
=== Chat CLI with AI Agent ===
Model: llama3.2:latest
Loading 3 tools from registry...

Type /help for commands, /quit to exit

You: /tools
Available tools:
  extract - Extract structured data (emails, URLs, dates, etc.)
  summarize - Summarize long text using LLM
  text_transform - Transform text case (upper, lower, title, etc.)

You: transform 'hello agent' to title case
Assistant: [Using text_transform] Hello Agent

You: /quit
Goodbye!
```

## Available Tools

The chat CLI automatically discovers and registers all tools from the ai-functions-go registry:

- **text_transform** - Text case transformations (uppercase, lowercase, title case, etc.)
- **summarize** - LLM-based text summarization
- **extract** - Extract structured data like emails, URLs, dates, phone numbers

## Configuration

Default configuration:
- **Model**: llama3.2:latest
- **Ollama URL**: http://localhost:11434
- **Session ID**: chat-session (persistent across messages in same run)

To use a different model, modify the `main.go` file:
```go
provider, err := llm.NewOllamaProvider("http://localhost:11434", "your-model:latest")
```

## How It Works

1. **Initialization**: Creates Ollama provider and agent with default config
2. **Tool Discovery**: Uses `registry.Global().List()` to find all registered functions
3. **Registration**: Registers each discovered function as a tool with the agent
4. **Event Loop**: Processes user input and agent responses via event streaming
5. **Command Routing**: Routes `/` prefixed commands to command handler, others to agent

## Code Structure

- `main()` - Entry point, sets up provider and agent
- `registerAllTools()` - Discovers and registers functions from registry
- `runChatLoop()` - Main REPL loop for interactive chat
- `processMessage()` - Sends messages to agent and handles event stream
- `handleCommand()` - Processes meta commands like /help, /tools, /quit
- Helper functions for display (printWelcome, printHelp, listTools)

## Extending

To add more tools:

1. Create a new function in ai-functions-go/pkg/builtin/
2. Import it with `_` in main.go imports section
3. It will be automatically discovered and registered!

Example:
```go
import (
    _ "github.com/pvjammer/ai-functions-go/pkg/builtin/yourfunction"
)
```

## Troubleshooting

**Error: "connection refused"**
- Make sure Ollama is running: `ollama serve`

**Error: "model not found"**
- Pull the model: `ollama pull llama3.2:latest`

**No tools loaded**
- Verify ai-functions-go builtin imports are present
- Check `go.mod` has correct replace directive for local ai-functions-go

**Agent not using tools**
- The SimplePlanner uses prompt-based tool calling
- Agent will output `USE_TOOL: name {...}` which gets parsed
- If tools aren't being used, the LLM might not understand the prompt format
