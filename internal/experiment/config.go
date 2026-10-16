// Package experiment implements cmd/experiment-runner: reproducible, scripted A/B runs of
// examples/tui-agent across a matrix of configs (model, branching, context size, ...), each
// starting from the same checkpoint every time — a fresh git worktree for git-sourced cases,
// never a leftover one — with a mechanical (not LLM-judged) report at the end.
package experiment

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// Config is one experiment suite, loaded from a YAML file (see experiments/configs/*.yaml for
// the two suites this package was built to reproduce).
type Config struct {
	Name        string         `yaml:"name"`
	Description string         `yaml:"description"`

	// Defaults are forwarded to tui-agent as CLI flags, one per key: a key "context_window"
	// with value 98304 becomes "--context-window 98304". snake_case keys convert to
	// kebab-case flags. This is deliberately generic — new tui-agent flags work from a YAML
	// config immediately, with no change to this package, as long as the key name (with
	// underscores swapped for dashes) matches the flag name.
	Defaults map[string]interface{} `yaml:"defaults"`

	// Source describes what working directory each case's tui-agent process runs in.
	Source Source `yaml:"source"`

	// Turns run in order, identically, against every case — that's what makes cases
	// comparable. ForbiddenTools lets a turn declare "calling any of these here would violate
	// the turn's own instruction" (e.g. a recall probe that says "without looking it up
	// again") — checked mechanically from the structured tool-call log, not guessed.
	Turns []Turn `yaml:"turns"`

	// Cases are the actual runs: each one's map is merged over Defaults (case wins on
	// conflicting keys) and forwarded the same way.
	Cases []Case `yaml:"cases"`
}

// Source is where a case's tui-agent process does its work.
//
//	type: scratch   — an empty directory per case (tui-agent's own --coding-tools init,
//	                   or no coding tools at all for a pure lookup/assistant suite).
//	type: git        — repo/ref checked out into a FRESH git worktree per case, every run.
//	                   ref defaults to "HEAD", resolved to an exact commit SHA once at the
//	                   start of a run and pinned in the report for provenance; every case in
//	                   that run, and every later rerun of the same config (unless ref is
//	                   changed), starts from that same exact commit — never a leftover
//	                   worktree, never whatever HEAD happens to be by the time it runs.
type Source struct {
	Type string `yaml:"type"`
	Repo string `yaml:"repo"`
	Ref  string `yaml:"ref"`
}

type Turn struct {
	Prompt         string   `yaml:"prompt"`
	ForbiddenTools []string `yaml:"forbidden_tools,omitempty"`
}

type Case struct {
	Name string `yaml:"name"`
	// Replicates runs this case this many times, as separate sub-cases named "<name>-r1",
	// "<name>-r2", ... (each with identical merged flags) — see ExpandCases. Default 1.
	Replicates int                    `yaml:"replicates,omitempty"`
	Overrides  map[string]interface{} `yaml:",inline"`
}

// ExpandCases turns any Case with Replicates > 1 into that many identically-configured
// sub-cases with distinct names, so a suite can ask for "n copies of this" without pasting the
// same block n times in the YAML. Call this once, right after Load, before anything else in
// this package sees the case list.
func ExpandCases(cases []Case) []Case {
	out := make([]Case, 0, len(cases))
	for _, c := range cases {
		n := c.Replicates
		if n <= 1 {
			out = append(out, c)
			continue
		}
		for i := 1; i <= n; i++ {
			rep := c
			rep.Name = fmt.Sprintf("%s-r%d", c.Name, i)
			rep.Replicates = 0
			out = append(out, rep)
		}
	}
	return out
}

func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	if cfg.Name == "" {
		return nil, fmt.Errorf("config has no name")
	}
	if len(cfg.Cases) == 0 {
		return nil, fmt.Errorf("config has no cases")
	}
	cfg.Cases = ExpandCases(cfg.Cases)
	if len(cfg.Turns) == 0 {
		return nil, fmt.Errorf("config has no turns")
	}
	switch cfg.Source.Type {
	case "scratch":
	case "git":
		if cfg.Source.Repo == "" {
			return nil, fmt.Errorf("source.type=git requires source.repo")
		}
		if cfg.Source.Ref == "" {
			cfg.Source.Ref = "HEAD"
		}
	default:
		return nil, fmt.Errorf("source.type must be \"scratch\" or \"git\", got %q", cfg.Source.Type)
	}
	seen := make(map[string]bool, len(cfg.Cases))
	for _, c := range cfg.Cases {
		if c.Name == "" {
			return nil, fmt.Errorf("every case needs a name")
		}
		if seen[c.Name] {
			return nil, fmt.Errorf("duplicate case name %q", c.Name)
		}
		seen[c.Name] = true
	}
	return &cfg, nil
}

// merged returns c's overrides merged over the config's defaults, case winning on conflicts.
func (cfg *Config) merged(c Case) map[string]interface{} {
	out := make(map[string]interface{}, len(cfg.Defaults)+len(c.Overrides))
	for k, v := range cfg.Defaults {
		out[k] = v
	}
	for k, v := range c.Overrides {
		out[k] = v
	}
	return out
}
