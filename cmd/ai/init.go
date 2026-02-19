package main

import (
	"fmt"
	"os"

	"github.com/pvjammer/ai-sdk-go/pkg/config"
	"github.com/spf13/cobra"
)

func createInitCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "init",
		Short: "Create or update config file",
		Long: `Create or update the configuration file at ~/.config/ai/config.toml.

If no config file exists a fresh one is created with defaults for every
registered function.

If a config file already exists it is updated non-destructively: new
function sections are added with their defaults and all existing values
are left unchanged. Safe to run at any time after adding new functions
or plugins.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			path, err := config.ConfigPath()
			if err != nil {
				return fmt.Errorf("cannot determine config path: %w", err)
			}

			defaults := config.DefaultAppConfig(registryFunctionDefaults())

			// If a config already exists, merge — never overwrite user values.
			var final *config.AppConfig
			if _, err := os.Stat(path); err == nil {
				existing, loadErr := config.LoadAppConfig(path)
				if loadErr != nil {
					return fmt.Errorf("failed to read existing config: %w", loadErr)
				}
				final = config.MergeDefaults(existing, defaults)
				fmt.Printf("Updating config: %s\n", path)
			} else {
				final = defaults
				fmt.Printf("Creating config: %s\n", path)
			}

			if err := config.SaveAppConfig(path, final); err != nil {
				return fmt.Errorf("failed to write config: %w", err)
			}

			fmt.Println("Done.")
			return nil
		},
	}
}
