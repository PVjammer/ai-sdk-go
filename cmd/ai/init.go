package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/pvjammer/ai-sdk-go/pkg/config"
	"github.com/spf13/cobra"
)

func createInitCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "init",
		Short: "Generate default config file",
		Long: `Generate a default configuration file at ~/.config/ai/config.toml.

The config file includes LLM provider settings, agent settings, and
per-function settings with defaults derived from each function's
declared parameters.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			path, err := config.ConfigPath()
			if err != nil {
				return fmt.Errorf("cannot determine config path: %w", err)
			}

			// Check if file exists
			if _, err := os.Stat(path); err == nil {
				fmt.Printf("Config file already exists: %s\n", path)
				fmt.Print("Overwrite? [y/N] ")

				reader := bufio.NewReader(os.Stdin)
				answer, _ := reader.ReadString('\n')
				answer = strings.TrimSpace(strings.ToLower(answer))

				if answer != "y" && answer != "yes" {
					fmt.Println("Aborted.")
					return nil
				}
			}

			// Generate default config from registry
			appCfg := config.DefaultAppConfig(registryFunctionDefaults())

			// Write to disk
			if err := config.SaveAppConfig(path, appCfg); err != nil {
				return fmt.Errorf("failed to write config: %w", err)
			}

			fmt.Printf("Config written to: %s\n", path)
			return nil
		},
	}
}
