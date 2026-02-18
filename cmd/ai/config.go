package main

import (
	"fmt"
	"os"
	"sort"
	"strconv"

	"github.com/charmbracelet/huh"
	"github.com/pvjammer/ai-sdk-go/pkg/config"
	"github.com/pvjammer/ai-sdk-go/pkg/registry"
	"github.com/spf13/cobra"
)

func createConfigCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "config",
		Short: "View or edit configuration",
		Long:  `Open an interactive TUI editor to browse and modify AI CLI settings.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			return runConfigTUI()
		},
	}

	cmd.AddCommand(createConfigShowCommand())
	cmd.AddCommand(createConfigPathCommand())

	return cmd
}

func createConfigShowCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "show",
		Short: "Print current configuration",
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := loadOrDefaultAppConfig()
			if err != nil {
				return err
			}

			path, _ := config.ConfigPath()
			if _, err := os.Stat(path); err == nil {
				fmt.Printf("# Config file: %s\n\n", path)
			} else {
				fmt.Printf("# No config file found, showing defaults\n\n")
			}

			fmt.Print(config.MarshalAppConfig(cfg))
			return nil
		},
	}
}

func createConfigPathCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "path",
		Short: "Print config file path",
		RunE: func(cmd *cobra.Command, args []string) error {
			path, err := config.ConfigPath()
			if err != nil {
				return err
			}
			fmt.Println(path)
			return nil
		},
	}
}

func loadOrDefaultAppConfig() (*config.AppConfig, error) {
	path, err := config.ConfigPath()
	if err != nil {
		return nil, err
	}

	if _, err := os.Stat(path); err == nil {
		return config.LoadAppConfig(path)
	}

	return config.DefaultAppConfig(registryFunctionDefaults()), nil
}

func runConfigTUI() error {
	cfg, err := loadOrDefaultAppConfig()
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	reg := registry.Global()

	// Build form groups
	var groups []*huh.Group

	// --- LLM Settings ---
	groups = append(groups, huh.NewGroup(
		huh.NewInput().
			Title("LLM Provider").
			Description("LLM provider name (e.g., ollama)").
			Value(&cfg.LLM.Provider),
		huh.NewInput().
			Title("LLM Endpoint").
			Description("Provider API endpoint URL").
			Value(&cfg.LLM.Endpoint),
		huh.NewInput().
			Title("LLM Model").
			Description("Default model to use").
			Value(&cfg.LLM.Model),
	).Title("LLM Settings"))

	// --- Agent Settings ---
	// huh inputs work with strings, so we convert
	maxIterStr := strconv.Itoa(cfg.Agent.MaxIterations)
	tempStr := strconv.FormatFloat(cfg.Agent.Temperature, 'f', -1, 64)
	maxTokStr := strconv.Itoa(cfg.Agent.MaxTokens)

	groups = append(groups, huh.NewGroup(
		huh.NewInput().
			Title("Max Iterations").
			Description("Maximum agent loop iterations").
			Value(&maxIterStr),
		huh.NewInput().
			Title("Temperature").
			Description("LLM sampling temperature (0.0-1.0)").
			Value(&tempStr),
		huh.NewInput().
			Title("Max Tokens").
			Description("Maximum tokens per LLM call").
			Value(&maxTokStr),
	).Title("Agent Settings"))

	// --- Per-function pages ---
	funcNames := make([]string, 0, len(cfg.Functions))
	for name := range cfg.Functions {
		funcNames = append(funcNames, name)
	}
	sort.Strings(funcNames)

	// Stable pointers for huh bindings
	type funcBindings struct {
		enabledCLI       bool
		enabledTool      bool
		requiresApproval bool
		paramStrings     map[string]*string
		paramBools       map[string]*bool
	}

	bindings := make(map[string]*funcBindings)

	for _, name := range funcNames {
		fc := cfg.Functions[name]
		fb := &funcBindings{
			enabledCLI:       fc.EnabledCLI,
			enabledTool:      fc.EnabledTool,
			requiresApproval: fc.RequiresApproval,
			paramStrings:     make(map[string]*string),
			paramBools:       make(map[string]*bool),
		}

		var fields []huh.Field

		fields = append(fields,
			huh.NewConfirm().
				Title("Enabled for CLI").
				Description("Show this function as a CLI subcommand").
				Value(&fb.enabledCLI),
			huh.NewConfirm().
				Title("Enabled as Tool").
				Description("Make available to the AI agent").
				Value(&fb.enabledTool),
			huh.NewConfirm().
				Title("Requires Approval").
				Description("Prompt before agent executes this tool").
				Value(&fb.requiresApproval),
		)

		// Add function-specific params from registry declarations
		registration := reg.GetRegistration(name)
		if registration != nil {
			for _, p := range registration.ConfigParams {
				currentVal := fc.Params[p.Key]

				if p.Type == "bool" {
					bv := false
					if b, ok := currentVal.(bool); ok {
						bv = b
					}
					fb.paramBools[p.Key] = &bv

					fields = append(fields,
						huh.NewConfirm().
							Title(p.Key).
							Description(p.Description).
							Value(fb.paramBools[p.Key]),
					)
				} else if len(p.Options) > 0 {
					// Select field for enums
					sv := ""
					if currentVal != nil {
						sv = fmt.Sprintf("%v", currentVal)
					}
					fb.paramStrings[p.Key] = &sv

					opts := make([]huh.Option[string], len(p.Options))
					for i, o := range p.Options {
						opts[i] = huh.NewOption(o, o)
					}

					fields = append(fields,
						huh.NewSelect[string]().
							Title(p.Key).
							Description(p.Description).
							Options(opts...).
							Value(fb.paramStrings[p.Key]),
					)
				} else {
					// Text input for strings/ints/floats
					sv := ""
					if currentVal != nil {
						sv = fmt.Sprintf("%v", currentVal)
					}
					fb.paramStrings[p.Key] = &sv

					fields = append(fields,
						huh.NewInput().
							Title(p.Key).
							Description(p.Description).
							Value(fb.paramStrings[p.Key]),
					)
				}
			}
		}

		bindings[name] = fb
		groups = append(groups, huh.NewGroup(fields...).Title(fmt.Sprintf("Function: %s", name)))
	}

	// Run TUI form
	form := huh.NewForm(groups...)
	if err := form.Run(); err != nil {
		return fmt.Errorf("config editor cancelled: %w", err)
	}

	// Write values back from bindings to config
	if v, err := strconv.Atoi(maxIterStr); err == nil {
		cfg.Agent.MaxIterations = v
	}
	if v, err := strconv.ParseFloat(tempStr, 64); err == nil {
		cfg.Agent.Temperature = v
	}
	if v, err := strconv.Atoi(maxTokStr); err == nil {
		cfg.Agent.MaxTokens = v
	}

	for _, name := range funcNames {
		fb := bindings[name]
		fc := cfg.Functions[name]
		fc.EnabledCLI = fb.enabledCLI
		fc.EnabledTool = fb.enabledTool
		fc.RequiresApproval = fb.requiresApproval

		// Write back param strings/bools
		registration := reg.GetRegistration(name)
		if registration != nil {
			for _, p := range registration.ConfigParams {
				if p.Type == "bool" {
					if bPtr, ok := fb.paramBools[p.Key]; ok {
						fc.Params[p.Key] = *bPtr
					}
				} else if sPtr, ok := fb.paramStrings[p.Key]; ok {
					switch p.Type {
					case "int":
						if v, err := strconv.Atoi(*sPtr); err == nil {
							fc.Params[p.Key] = v
						}
					case "float":
						if v, err := strconv.ParseFloat(*sPtr, 64); err == nil {
							fc.Params[p.Key] = v
						}
					default:
						fc.Params[p.Key] = *sPtr
					}
				}
			}
		}

		cfg.Functions[name] = fc
	}

	// Save to disk
	path, err := config.ConfigPath()
	if err != nil {
		return err
	}

	if err := config.SaveAppConfig(path, cfg); err != nil {
		return fmt.Errorf("failed to save config: %w", err)
	}

	fmt.Printf("Config saved to: %s\n", path)
	return nil
}
