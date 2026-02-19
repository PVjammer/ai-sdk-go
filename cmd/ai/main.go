// Package main provides the ai CLI.
package main

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/pvjammer/ai-sdk-go/pkg/cli"
	"github.com/pvjammer/ai-sdk-go/pkg/config"
	"github.com/pvjammer/ai-sdk-go/pkg/registry"
	"github.com/spf13/cobra"

	// Import builtin functions to trigger auto-registration
	_ "github.com/pvjammer/ai-sdk-go/pkg/builtin/extract"
	_ "github.com/pvjammer/ai-sdk-go/pkg/builtin/summarize"
	_ "github.com/pvjammer/ai-sdk-go/pkg/builtin/texttransform"
	_ "github.com/pvjammer/ai-sdk-go/pkg/builtin/websearch"
)

const version = "0.1.0"

var (
	configFile  string
	showVersion bool
	appCfg      *config.AppConfig
)

func main() {
	rootCmd := &cobra.Command{
		Use:   "ai",
		Short: "AI - A unified CLI for AI-powered functions and agents",
		Long: `AI provides a unified interface for executing AI-powered functions and agents.
Functions can be invoked directly via the CLI, used as tools in LLM applications,
or called programmatically via the Go API.`,
		SilenceUsage: true,
		PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
			// Load Viper-based config (environment variables, YAML fallback)
			cfg := config.GlobalConfig()
			if err := cfg.LoadDefaults(); err != nil {
				// Non-fatal - config files are optional
			}

			// Load custom config file if specified via --config flag
			if configFile != "" {
				if err := cfg.LoadFile(configFile); err != nil {
					return fmt.Errorf("failed to load config file: %w", err)
				}
			}

			// Load structured TOML AppConfig
			var err error
			appCfg, err = loadAppConfig()
			if err != nil {
				// Non-fatal — fall back to defaults
				appCfg = config.DefaultAppConfig(registryFunctionDefaults())
			}

			// Bridge TOML values into Viper so functions read them
			config.BridgeToViper(appCfg, cfg)

			return nil
		},
		Run: func(cmd *cobra.Command, args []string) {
			if showVersion {
				fmt.Printf("ai version %s\n", version)
				return
			}
			cmd.Help()
		},
	}

	// Global flags
	rootCmd.PersistentFlags().StringVarP(&configFile, "config", "c", "", "config file (default is ~/.config/ai/config.toml)")
	rootCmd.Flags().BoolVarP(&showVersion, "version", "v", false, "show version information")

	// Add subcommands
	rootCmd.AddCommand(createListCommand())
	rootCmd.AddCommand(createInitCommand())
	rootCmd.AddCommand(createConfigCommand())
	rootCmd.AddCommand(createChatCommand())

	// Auto-discover and add function commands
	addFunctionCommands(rootCmd)

	// Execute
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

// registryFunctionDefaults converts registry registrations into the
// []config.FunctionDefaults needed by the config package (avoids import cycle).
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

// loadAppConfig loads the TOML AppConfig, checking (in order):
// 1. --config flag path
// 2. ./ai.toml (local override)
// 3. ~/.config/ai/config.toml (XDG default)
// Falls back to generated defaults if nothing is found.
func loadAppConfig() (*config.AppConfig, error) {
	// If --config flag was given and it's a .toml file, use it
	if configFile != "" && strings.HasSuffix(configFile, ".toml") {
		return config.LoadAppConfig(configFile)
	}

	// Check for local override
	if _, err := os.Stat("ai.toml"); err == nil {
		return config.LoadAppConfig("ai.toml")
	}

	// Check XDG default path
	return config.EnsureConfig(registryFunctionDefaults())
}

// addFunctionCommands discovers functions from the registry and adds them as commands.
func addFunctionCommands(rootCmd *cobra.Command) {
	reg := registry.Global()
	cfg := config.GlobalConfig()

	// Load app config early for filtering (may already be set by PersistentPreRunE)
	filterCfg := appCfg
	if filterCfg == nil {
		filterCfg, _ = loadAppConfig()
		if filterCfg == nil {
			filterCfg = config.DefaultAppConfig(registryFunctionDefaults())
		}
	}

	// Get all registered functions
	names := reg.List()

	for _, name := range names {
		// Check if function is enabled for CLI
		if fc, ok := filterCfg.Functions[name]; ok && !fc.EnabledCLI {
			continue
		}

		// Create function instance to get metadata
		fn := reg.Get(name, cfg)
		if fn == nil {
			continue
		}

		// Generate Cobra command from function
		cmd := cli.AsCommand(fn)

		// Add to root
		rootCmd.AddCommand(cmd)
	}
}

// createListCommand creates the "list" command.
func createListCommand() *cobra.Command {
	var (
		byCategory bool
		byTag      bool
		verbose    bool
	)

	cmd := &cobra.Command{
		Use:   "list",
		Short: "List available functions",
		Long: `List all available AI functions.

Functions can be filtered by category or tag, and displayed with verbose output
showing descriptions and metadata.`,
		Run: func(cmd *cobra.Command, args []string) {
			reg := registry.Global()

			if byCategory {
				listByCategory(reg)
			} else if byTag {
				listByTag(reg)
			} else if verbose {
				listVerbose(reg)
			} else {
				listSimple(reg)
			}
		},
	}

	cmd.Flags().BoolVar(&byCategory, "by-category", false, "group functions by category")
	cmd.Flags().BoolVar(&byTag, "by-tag", false, "show all tags")
	cmd.Flags().BoolVarP(&verbose, "verbose", "v", false, "show detailed information")

	return cmd
}

// listSimple lists function names.
func listSimple(reg *registry.Registry) {
	names := reg.List()
	for _, name := range names {
		fmt.Println(name)
	}
}

// listVerbose lists functions with descriptions.
func listVerbose(reg *registry.Registry) {
	regs := reg.ListWithMetadata()

	fmt.Println("Available Functions:")
	fmt.Println()

	for _, r := range regs {
		fmt.Printf("  %s\n", r.Name)
		fmt.Printf("    Description: %s\n", r.Description)
		fmt.Printf("    Category: %s\n", r.Category)
		if len(r.Tags) > 0 {
			fmt.Printf("    Tags: %s\n", strings.Join(r.Tags, ", "))
		}
		fmt.Println()
	}
}

// listByCategory lists functions grouped by category.
func listByCategory(reg *registry.Registry) {
	categories := reg.GetCategories()

	fmt.Println("Functions by Category:")
	fmt.Println()

	for _, category := range categories {
		functions := reg.GetByCategory(category)
		fmt.Printf("  %s:\n", category)
		for _, fn := range functions {
			fmt.Printf("    - %s\n", fn)
		}
		fmt.Println()
	}
}

// listByTag lists all available tags.
func listByTag(reg *registry.Registry) {
	tags := reg.GetTags()

	fmt.Println("Available Tags:")
	fmt.Println()

	for _, tag := range tags {
		functions := reg.GetByTag(tag)
		fmt.Printf("  %s (%d function", tag, len(functions))
		if len(functions) != 1 {
			fmt.Print("s")
		}
		fmt.Println(")")

		// Show functions with this tag
		sort.Strings(functions)
		for _, fn := range functions {
			fmt.Printf("    - %s\n", fn)
		}
		fmt.Println()
	}
}
