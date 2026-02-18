// Package main demonstrates a personal assistant with personalization and context switching.
package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/pvjammer/ai-sdk-go/pkg/agent"
	"github.com/pvjammer/ai-sdk-go/pkg/agentcontext"
	"github.com/pvjammer/ai-sdk-go/pkg/agentcontext/hooks"
	"github.com/pvjammer/ai-sdk-go/pkg/builtin/texttransform"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

func main() {
	fmt.Println("=== Personal Assistant with Context Switching ===")
	fmt.Println()

	// Check for Ollama
	ollamaURL := os.Getenv("OLLAMA_URL")
	if ollamaURL == "" {
		ollamaURL = "http://localhost:11434"
	}

	// Create Ollama provider
	provider, err := llm.NewOllamaProvider(ollamaURL, "llama3.2:latest")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create Ollama provider: %v\n", err)
		fmt.Fprintln(os.Stderr, "Make sure Ollama is running: ollama serve")
		os.Exit(1)
	}

	// Create Extended context with personalization and compression
	extConfig := &agentcontext.ExtendedConfig{
		DefaultContextID: "personal",
		MaxTokens:        2000, // Low limit to demonstrate compression
		AutoCompress:     true,
	}
	agentCtx := agentcontext.NewExtendedWithConfig(extConfig)

	// Set up personalization
	agentCtx.SetPersonalization("name", "Paul")
	agentCtx.SetPersonalization("preferences", map[string]interface{}{
		"greeting_style": "casual",
		"response_style": "concise",
	})

	// Register compression hooks
	agentCtx.RegisterCompressionHook(hooks.NewSummarization(provider, 2000))
	agentCtx.RegisterCompressionHook(hooks.NewSlidingWindow(15))

	// Create agent config with personalization in system prompt
	config := agent.DefaultConfig()
	config.Name = "PersonalAssistant"
	config.Context = agentCtx
	config.SystemPrompt = fmt.Sprintf(`You are a helpful personal assistant for %s.

Your user preferences:
- Greeting style: casual and friendly
- Response style: concise and to the point

You have access to tools to help with tasks. Be helpful and remember context from previous messages.`,
		agentCtx.GetPersonalization("name"))
	config.ThinkingVisible = false // Less verbose for interactive use

	// Create agent
	myAgent := agent.New(provider, config)

	// Register tools
	if err := myAgent.RegisterTool(texttransform.NewDefault()); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to register tools: %v\n", err)
		os.Exit(1)
	}

	// Print instructions
	fmt.Println("Commands:")
	fmt.Println("  /context <name>  - Switch context (e.g., 'work', 'personal', 'project')")
	fmt.Println("  /status          - Show current context and token usage")
	fmt.Println("  /quit            - Exit")
	fmt.Println()
	fmt.Printf("Current context: %s\n", agentCtx.GetCurrentContextID())
	fmt.Println()

	// Interactive loop
	scanner := bufio.NewScanner(os.Stdin)
	sessionID := "main-session"
	ctx := context.Background()

	for {
		fmt.Print("You: ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}

		// Handle commands
		if strings.HasPrefix(input, "/") {
			if !handleCommand(input, agentCtx) {
				break // Quit
			}
			continue
		}

		// Send to agent
		fmt.Print("Assistant: ")
		events := myAgent.Run(ctx, sessionID, input)

		responseReceived := false
		for event := range events {
			switch event.Type {
			case agent.EventResponse:
				response := event.Data.(agent.ResponseEvent)
				if response.Done {
					fmt.Println(response.Content)
					responseReceived = true
				}

			case agent.EventToolCall:
				toolCall := event.Data.(agent.ToolCallEvent)
				fmt.Printf("[Using %s]", toolCall.ToolName)

			case agent.EventError:
				errorEvent := event.Data.(agent.ErrorEvent)
				fmt.Fprintf(os.Stderr, "\nError: %v\n", errorEvent.Error)
				responseReceived = true
			}
		}

		if !responseReceived {
			fmt.Println()
		}
	}

	// Cleanup
	myAgent.Close()
	fmt.Println("\nGoodbye!")
}

// handleCommand processes special commands.
// Returns false if the user wants to quit.
func handleCommand(input string, agentCtx *agentcontext.Extended) bool {
	parts := strings.Fields(input)
	if len(parts) == 0 {
		return true
	}

	command := parts[0]

	switch command {
	case "/quit", "/exit", "/q":
		return false

	case "/context", "/ctx":
		if len(parts) < 2 {
			fmt.Println("Usage: /context <name>")
			fmt.Printf("Current context: %s\n", agentCtx.GetCurrentContextID())
			fmt.Printf("Available contexts: %v\n", agentCtx.GetAllContextIDs())
			return true
		}

		newContext := parts[1]
		if err := agentCtx.SwitchContext(newContext); err != nil {
			fmt.Printf("Error switching context: %v\n", err)
		} else {
			fmt.Printf("Switched to context: %s\n", newContext)
			fmt.Printf("Messages in this context: %d\n", agentCtx.GetContextMessageCount(newContext))
			fmt.Printf("Estimated tokens: %d\n", agentCtx.EstimateTokens())
		}

	case "/status", "/info":
		fmt.Printf("Current context: %s\n", agentCtx.GetCurrentContextID())
		fmt.Printf("All contexts: %v\n", agentCtx.GetAllContextIDs())
		fmt.Printf("Messages in current context: %d\n", len(agentCtx.GetMessages(0)))
		fmt.Printf("Estimated tokens: %d\n", agentCtx.EstimateTokens())
		fmt.Printf("Needs compression: %v\n", agentCtx.NeedsCompression())

		// Show personalization
		if name := agentCtx.GetPersonalization("name"); name != nil {
			fmt.Printf("User name: %v\n", name)
		}

	default:
		fmt.Printf("Unknown command: %s\n", command)
	}

	return true
}
