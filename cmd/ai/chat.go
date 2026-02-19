package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/pvjammer/ai-sdk-go/pkg/agent"
	"github.com/pvjammer/ai-sdk-go/pkg/agentreg"
	"github.com/pvjammer/ai-sdk-go/pkg/config"
	"github.com/pvjammer/ai-sdk-go/pkg/registry"
	"github.com/spf13/cobra"

	// Register the simple agent
	_ "github.com/pvjammer/ai-sdk-go/pkg/builtin/agents/simple"
)

func createChatCommand() *cobra.Command {
	var verbose bool

	cmd := &cobra.Command{
		Use:   "chat [message]",
		Short: "Chat with an AI agent that can use tools",
		Long: `Start an interactive chat session with an AI agent.

If a message is provided as an argument, it runs in single-shot mode and exits.
Otherwise, an interactive REPL is started.

The agent uses Ollama's native tool-calling API for reliable tool use.

Commands available in REPL mode:
  /help   - Show available commands
  /tools  - List registered tools
  /clear  - Clear conversation history
  /quit   - Exit`,
		Args: cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runChatCommand(args, verbose)
		},
	}

	cmd.Flags().BoolVarP(&verbose, "verbose", "v", false, "Show tool calls and debug information")

	return cmd
}

// runChatCommand sets up and runs the chat session.
func runChatCommand(args []string, verbose bool) error {
	// Determine agent type (default "simple")
	agentType := "simple"

	// Create agent instance from registry
	cfg := config.GlobalConfig()
	a := agentreg.Global().Get(agentType, cfg)
	if a == nil {
		return fmt.Errorf("agent %q not registered (available: %v)", agentType, agentreg.Global().List())
	}

	// Register all enabled tools
	if err := loadToolsIntoChatAgent(a); err != nil {
		return fmt.Errorf("loading tools: %w", err)
	}

	// Wire approval callback for tools that require it
	if appCfg != nil {
		installApprovalFunc(a, appCfg)
	}

	// Single-shot mode: one message, then exit
	if len(args) == 1 {
		runOnce(context.Background(), a, "default", args[0], verbose)
		return nil
	}

	// Interactive REPL mode
	printChatWelcome()
	runChatREPL(a, verbose)
	return nil
}

// loadToolsIntoChatAgent registers all enabled-tool functions from the registry.
func loadToolsIntoChatAgent(a agent.AIAgent) error {
	reg := registry.Global()
	cfg := config.GlobalConfig()

	for _, name := range reg.List() {
		// Skip tools disabled in config
		if appCfg != nil {
			if fc, ok := appCfg.Functions[name]; ok && !fc.EnabledTool {
				continue
			}
		}

		fn := reg.Get(name, cfg)
		if fn == nil {
			continue
		}

		if err := a.RegisterTool(fn); err != nil {
			return fmt.Errorf("registering tool %s: %w", name, err)
		}
	}

	return nil
}

// installApprovalFunc sets up a prompt-based approval callback for tools
// that have requires_approval = true in the config.
func installApprovalFunc(a agent.AIAgent, appCfg *config.AppConfig) {
	needsApproval := make(map[string]bool)
	for name, fc := range appCfg.Functions {
		if fc.RequiresApproval {
			needsApproval[name] = true
		}
	}

	if len(needsApproval) == 0 {
		return
	}

	a.SetApprovalFunc(func(toolName string, input interface{}) (bool, error) {
		// Map tool names (e.g. "web_search") back to registry names (e.g. "websearch")
		registryName := toolNameToRegistry(toolName)
		if !needsApproval[registryName] {
			return true, nil
		}

		fmt.Printf("\nAllow %s? [y/N] ", toolName)
		reader := bufio.NewReader(os.Stdin)
		answer, err := reader.ReadString('\n')
		if err != nil {
			return false, err
		}

		answer = strings.TrimSpace(strings.ToLower(answer))
		return answer == "y" || answer == "yes", nil
	})
}

// toolNameToRegistry maps LLM tool names back to registry names.
func toolNameToRegistry(toolName string) string {
	mappings := map[string]string{
		"web_search":     "websearch",
		"text_transform": "texttransform",
		"summarize_text": "summarize",
		"extract_data":   "extract",
	}
	if name, ok := mappings[toolName]; ok {
		return name
	}
	return toolName
}

// printChatWelcome prints the welcome banner.
func printChatWelcome() {
	model := "unknown"
	if appCfg != nil && appCfg.LLM.Model != "" {
		model = appCfg.LLM.Model
	}

	fmt.Printf("=== AI Chat (model: %s) ===\n", model)
	fmt.Println("Type /help for commands, /quit to exit")
	fmt.Println()
}

// runChatREPL runs the interactive REPL loop.
func runChatREPL(a agent.AIAgent, verbose bool) {
	scanner := bufio.NewScanner(os.Stdin)
	ctx := context.Background()
	sessionID := "chat-session"

	for {
		fmt.Print("You: ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}

		if strings.HasPrefix(input, "/") {
			if !handleChatCommand(input, a, sessionID) {
				break
			}
			continue
		}

		fmt.Print("Assistant: ")
		runOnce(ctx, a, sessionID, input, verbose)
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
	}
}

// handleChatCommand processes REPL meta-commands (/help, /tools, /clear, /quit).
// Returns false when the user wants to exit.
func handleChatCommand(cmd string, a agent.AIAgent, sessionID string) bool {
	parts := strings.Fields(cmd)
	if len(parts) == 0 {
		return true
	}

	switch parts[0] {
	case "/quit", "/exit":
		fmt.Println("Goodbye!")
		return false

	case "/help":
		fmt.Println("\nAvailable commands:")
		fmt.Println("  /help   - Show this help message")
		fmt.Println("  /tools  - List available tools")
		fmt.Println("  /clear  - Clear conversation history")
		fmt.Println("  /quit   - Exit the chat")
		fmt.Println()

	case "/tools":
		printChatTools()

	case "/clear":
		// Type-assert to access ClearSession (AgentBase method)
		type sessionClearer interface {
			ClearSession(sessionID string)
		}
		if sc, ok := a.(sessionClearer); ok {
			sc.ClearSession(sessionID)
			fmt.Println("Conversation history cleared.")
		} else {
			fmt.Println("(Clear not supported by this agent)")
		}

	default:
		fmt.Printf("Unknown command: %s (type /help for commands)\n", parts[0])
	}

	return true
}

// printChatTools lists the tools registered in the function registry.
func printChatTools() {
	reg := registry.Global()
	cfg := config.GlobalConfig()

	fmt.Println("\nAvailable tools:")
	for _, name := range reg.List() {
		fn := reg.Get(name, cfg)
		if fn == nil {
			continue
		}
		meta := fn.Metadata()
		desc := meta.GetDescription("tool")
		if desc == "" {
			desc = "(no description)"
		}
		fmt.Printf("  %-20s %s\n", meta.GetName("tool"), desc)
	}
	fmt.Println()
}

// runOnce sends a single message to the agent and prints events to stdout.
func runOnce(ctx context.Context, a agent.AIAgent, sessionID, message string, verbose bool) {
	events := a.Run(ctx, sessionID, message)

	for event := range events {
		switch event.Type {
		case agent.EventToolCall:
			tc := event.Data.(agent.ToolCallEvent)
			if verbose {
				fmt.Printf("\n[tool call] %s(%v)\n", tc.ToolName, tc.Input)
			} else {
				fmt.Printf("[using %s] ", tc.ToolName)
			}

		case agent.EventToolResult:
			res := event.Data.(agent.ToolResultEvent)
			if verbose {
				if res.Error != nil {
					fmt.Printf("[tool result] %s: error: %v\n", res.ToolName, res.Error)
				} else {
					fmt.Printf("[tool result] %s: %v\n", res.ToolName, res.Output)
				}
			}

		case agent.EventThinking:
			thinking := event.Data.(agent.ThinkingEvent)
			if verbose {
				fmt.Printf("[thinking] %s\n", thinking.Content)
			}

		case agent.EventResponse:
			response := event.Data.(agent.ResponseEvent)
			if response.Done {
				fmt.Println(response.Content)
			}

		case agent.EventError:
			errEvent := event.Data.(agent.ErrorEvent)
			fmt.Fprintf(os.Stderr, "\n[error] %v\n", errEvent.Error)

		case agent.EventComplete:
			// Nothing to print; EventResponse already printed the content.
		}
	}
}
