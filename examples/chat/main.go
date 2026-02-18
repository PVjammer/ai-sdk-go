package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/pvjammer/ai-sdk-go/pkg/agent"
	"github.com/pvjammer/ai-sdk-go/pkg/config"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
	"github.com/pvjammer/ai-sdk-go/pkg/registry"

	// Auto-register builtin functions
	_ "github.com/pvjammer/ai-sdk-go/pkg/builtin/extract"
	_ "github.com/pvjammer/ai-sdk-go/pkg/builtin/summarize"
	_ "github.com/pvjammer/ai-sdk-go/pkg/builtin/texttransform"
	_ "github.com/pvjammer/ai-sdk-go/pkg/builtin/websearch"
)

var (
	commandMode = flag.String("c", "", "Execute a single command and exit")
	verbose     = flag.Bool("v", false, "Enable verbose logging")
)

// registryFunctionDefaults converts registry registrations into
// []config.FunctionDefaults for the config package.
func registryFunctionDefaults() []config.FunctionDefaults {
	reg := registry.Global()
	names := reg.List()
	result := make([]config.FunctionDefaults, 0, len(names))

	for _, name := range names {
		fd := config.FunctionDefaults{Name: name}
		registration := reg.GetRegistration(name)
		if registration != nil {
			for _, p := range registration.ConfigParams {
				fd.Params = append(fd.Params, config.FunctionDefaultParam{
					Key:     p.Key,
					Default: p.Default,
				})
			}
		}
		result = append(result, fd)
	}

	return result
}

func main() {
	flag.Parse()

	// Load structured TOML config
	appCfg, err := config.EnsureConfig(registryFunctionDefaults())
	if err != nil {
		appCfg = config.DefaultAppConfig(registryFunctionDefaults())
	}

	// Bridge to Viper for backward compatibility
	config.BridgeToViper(appCfg, config.GlobalConfig())

	// Determine endpoint and model from config
	endpoint := appCfg.LLM.Endpoint
	if endpoint == "" {
		endpoint = "http://localhost:11434"
	}
	model := appCfg.LLM.Model
	if model == "" {
		model = "gpt-oss:20b"
	}

	// Create Ollama provider
	provider, err := llm.NewOllamaProvider(endpoint, model)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating Ollama provider: %v\n", err)
		fmt.Fprintf(os.Stderr, "Make sure Ollama is running: ollama serve\n")
		os.Exit(1)
	}

	// Configure agent
	agentConfig := agent.DefaultConfig()
	agentConfig.Name = "ChatBot"
	agentConfig.SystemPrompt = buildSystemPrompt()
	agentConfig.ThinkingVisible = *verbose

	// Apply config values
	if appCfg.Agent.MaxIterations > 0 {
		agentConfig.MaxIterations = appCfg.Agent.MaxIterations
	}
	if appCfg.Agent.MaxTokens > 0 {
		agentConfig.MaxTokens = appCfg.Agent.MaxTokens
	}
	if appCfg.Agent.Temperature > 0 {
		agentConfig.Temperature = appCfg.Agent.Temperature
	}

	// Wire approval callback based on config
	agentConfig.ApprovalFunc = buildApprovalFunc(appCfg)

	// Create agent
	myAgent := agent.New(provider, agentConfig)
	defer myAgent.Close()

	// Load and register tools (filtered by enabled_tool)
	if err := registerAllTools(myAgent, appCfg); err != nil {
		fmt.Fprintf(os.Stderr, "Error loading tools: %v\n", err)
		os.Exit(1)
	}

	// Command mode (-c flag)
	if *commandMode != "" {
		if *verbose {
			fmt.Printf("[VERBOSE] Executing command: %s\n", *commandMode)
		}
		processMessage(context.Background(), myAgent, "single-command-session", *commandMode)
		return
	}

	// Print welcome message
	printWelcome()

	// Start chat loop
	runChatLoop(myAgent)
}

// buildApprovalFunc creates an approval callback that prompts the user
// for tools that have requires_approval = true in the config.
func buildApprovalFunc(appCfg *config.AppConfig) func(string, interface{}) (bool, error) {
	needsApproval := make(map[string]bool)
	for name, fc := range appCfg.Functions {
		if fc.RequiresApproval {
			needsApproval[name] = true
		}
	}

	if len(needsApproval) == 0 {
		return nil
	}

	return func(toolName string, input interface{}) (bool, error) {
		registryName := toolNameToRegistryName(toolName)

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
	}
}

// toolNameToRegistryName maps tool names (as used by the agent) back to
// registry names (as used in config).
func toolNameToRegistryName(toolName string) string {
	mappings := map[string]string{
		"web_search":     "websearch",
		"text_transform": "text_transform",
		"summarize_text": "summarize",
		"extract_data":   "extract",
	}
	if name, ok := mappings[toolName]; ok {
		return name
	}
	return toolName
}

// registerAllTools discovers and registers all functions from the registry,
// filtering by enabled_tool config.
func registerAllTools(myAgent *agent.Agent, appCfg *config.AppConfig) error {
	reg := registry.Global()
	cfg := config.GlobalConfig()

	toolNames := reg.List()
	fmt.Printf("Loading tools from registry...\n")

	loaded := 0
	for _, name := range toolNames {
		if fc, ok := appCfg.Functions[name]; ok && !fc.EnabledTool {
			if *verbose {
				fmt.Printf("[VERBOSE] Skipping disabled tool: %s\n", name)
			}
			continue
		}

		fn := reg.Get(name, cfg)
		if fn == nil {
			return fmt.Errorf("failed to instantiate %s", name)
		}

		if err := myAgent.RegisterTool(fn); err != nil {
			return fmt.Errorf("failed to register %s: %w", name, err)
		}

		meta := fn.Metadata()
		toolName := meta.ToolName
		if toolName == "" {
			toolName = name
		}

		if *verbose {
			fmt.Printf("[VERBOSE] Registered tool: %s (function: %s)\n", toolName, name)
			fmt.Printf("[VERBOSE]   Description: %s\n", meta.ToolDescription)
		} else {
			fmt.Printf("  + %s\n", toolName)
		}
		loaded++
	}

	fmt.Printf("Loaded %d tools\n", loaded)
	return nil
}

// handleCommand processes meta commands like /help, /tools, /quit
func handleCommand(cmd string, myAgent *agent.Agent, sessionID string) bool {
	parts := strings.Fields(cmd)
	if len(parts) == 0 {
		return true
	}

	switch parts[0] {
	case "/quit", "/exit":
		fmt.Println("Goodbye!")
		return false
	case "/help":
		printHelp()
	case "/tools":
		listTools()
	case "/clear":
		clearConversation(myAgent, sessionID)
	default:
		fmt.Printf("Unknown command: %s (type /help for commands)\n", parts[0])
	}

	return true
}

// runChatLoop implements the main REPL loop
func runChatLoop(myAgent *agent.Agent) {
	scanner := bufio.NewScanner(os.Stdin)
	ctx := context.Background()
	sessionID := "chat-session"

	for {
		fmt.Print("\nYou: ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}

		if strings.HasPrefix(input, "/") {
			if !handleCommand(input, myAgent, sessionID) {
				break
			}
			continue
		}

		fmt.Print("Assistant: ")
		processMessage(ctx, myAgent, sessionID, input)
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
	}
}

// processMessage sends a message to the agent and handles the event stream
func processMessage(ctx context.Context, myAgent *agent.Agent, sessionID, message string) {
	if *verbose {
		fmt.Printf("\n[VERBOSE] Processing message: %s\n", message)
		fmt.Printf("[VERBOSE] Session ID: %s\n", sessionID)
	}

	events := myAgent.Run(ctx, sessionID, message)
	responseReceived := false

	for event := range events {
		switch event.Type {
		case agent.EventToolCall:
			toolCall := event.Data.(agent.ToolCallEvent)
			if *verbose {
				fmt.Printf("\n[VERBOSE] Tool call detected:\n")
				fmt.Printf("[VERBOSE]   Tool: %s\n", toolCall.ToolName)
				fmt.Printf("[VERBOSE]   Input: %+v\n", toolCall.Input)
			} else {
				fmt.Printf("[Using %s] ", toolCall.ToolName)
			}

		case agent.EventToolResult:
			result := event.Data.(agent.ToolResultEvent)
			if *verbose {
				fmt.Printf("[VERBOSE] Tool result:\n")
				fmt.Printf("[VERBOSE]   Tool: %s\n", result.ToolName)
				fmt.Printf("[VERBOSE]   Success: %v\n", result.Error == nil)
				if result.Error != nil {
					fmt.Printf("[VERBOSE]   Error: %v\n", result.Error)
				} else {
					fmt.Printf("[VERBOSE]   Output: %+v\n", result.Output)
				}
			}

		case agent.EventThinking:
			thinking := event.Data.(agent.ThinkingEvent)
			if *verbose {
				fmt.Printf("[VERBOSE] Thinking: %s\n", thinking.Content)
			}

		case agent.EventResponse:
			response := event.Data.(agent.ResponseEvent)
			if *verbose {
				fmt.Printf("[VERBOSE] Response event (done=%v): %s\n", response.Done, response.Content)
			}
			if response.Done {
				if !*verbose {
					fmt.Println(response.Content)
				}
				responseReceived = true
			}

		case agent.EventError:
			errorEvent := event.Data.(agent.ErrorEvent)
			fmt.Fprintf(os.Stderr, "\n[ERROR] %v\n", errorEvent.Error)
			if *verbose {
				fmt.Fprintf(os.Stderr, "[VERBOSE] Error context: %+v\n", errorEvent)
			}
			responseReceived = true
		}
	}

	if !responseReceived {
		fmt.Println()
	}

	if *verbose {
		fmt.Printf("[VERBOSE] Message processing complete\n\n")
	}
}

// buildSystemPrompt creates the system prompt for the agent
func buildSystemPrompt() string {
	return `You are a helpful AI assistant with access to various tools.

IMPORTANT: To use a tool, you MUST output exactly in this format on a single line:
USE_TOOL: tool_name {"param1": "value1", "param2": "value2"}

Examples:
- To search the web: USE_TOOL: web_search {"query": "golang tutorials", "max_results": 5}
- To search and fetch page content: USE_TOOL: web_search {"query": "golang error handling best practices", "max_results": 3, "fetch_content": true, "content_count": 2}
- To transform text: USE_TOOL: text_transform {"text": "hello", "operation": "upper"}

When the user needs detailed information from web pages (not just links), use "fetch_content": true to retrieve actual page content. Set "content_count" to control how many pages to fetch.

If you don't need to use a tool, just respond normally to the user.
Always be concise and helpful.`
}

// printWelcome displays the welcome message
func printWelcome() {
	fmt.Println("=== Chat CLI with AI Agent ===")
	fmt.Println("Model: llama3.2:latest")
	fmt.Println("\nType /help for commands, /quit to exit")
	fmt.Println()
}

// printHelp displays available commands
func printHelp() {
	fmt.Println("\nAvailable commands:")
	fmt.Println("  /help   - Show this help message")
	fmt.Println("  /tools  - List available tools")
	fmt.Println("  /clear  - Clear conversation history")
	fmt.Println("  /quit   - Exit the chat")
	fmt.Println()
}

// listTools displays all registered tools with descriptions
func listTools() {
	reg := registry.Global()
	cfg := config.GlobalConfig()

	fmt.Println("\nAvailable tools:")

	for _, name := range reg.List() {
		fn := reg.Get(name, cfg)
		if fn == nil {
			continue
		}

		meta := fn.Metadata()
		description := meta.Description
		if description == "" {
			description = "No description available"
		}

		fmt.Printf("  %s - %s\n", name, description)
	}
	fmt.Println()
}

// clearConversation clears the conversation history for the current session
func clearConversation(myAgent *agent.Agent, sessionID string) {
	fmt.Println("Conversation cleared. Starting fresh!")
	fmt.Println("(Note: Session state is managed internally by the agent)")
}
