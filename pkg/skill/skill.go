// Package skill provides skill discovery, parsing, and management for the ai CLI.
//
// Skills are directories containing a SKILL.md file with YAML frontmatter and a
// markdown body. This package implements the agentskills.io standard format for
// maximum interoperability with Claude Code, Cursor, Gemini CLI, and others.
//
// Three skill types are supported:
//   - SkillTypePrompt: markdown instructions only; body injected into LLM context
//   - SkillTypeFunction: backed by a registered AIFunction (frontmatter: function: name)
//   - SkillTypeScript: prompt + bundled scripts in a scripts/ subdirectory
package skill

// SkillType describes what backs a skill.
type SkillType string

const (
	// SkillTypePrompt is a markdown-only skill with no backing Go code.
	SkillTypePrompt SkillType = "prompt"
	// SkillTypeFunction is backed by a named AIFunction in the registry.
	SkillTypeFunction SkillType = "function"
	// SkillTypeScript is a prompt skill that bundles executable scripts.
	SkillTypeScript SkillType = "script"
)

// SkillScope indicates where the skill was discovered.
// Higher values take precedence on name collision.
type SkillScope int

const (
	SkillScopeBuiltin         SkillScope = 1 // embedded in binary
	SkillScopeUserSDK         SkillScope = 2 // ~/.config/ai/skills/
	SkillScopeUserPortable    SkillScope = 3 // ~/.agents/skills/
	SkillScopeProjectPortable SkillScope = 4 // .agents/skills/
	SkillScopeProjectSDK      SkillScope = 5 // .claude/skills/
)

// SkillFrontmatter holds the YAML frontmatter fields parsed from SKILL.md.
// All fields follow the agentskills.io specification.
type SkillFrontmatter struct {
	// Standard agentskills.io fields
	Name                   string            `yaml:"name"`
	Description            string            `yaml:"description"`
	ArgumentHint           string            `yaml:"argument-hint"`
	AllowedTools           string            `yaml:"allowed-tools"`
	DisableModelInvocation bool              `yaml:"disable-model-invocation"`
	UserInvocable          *bool             `yaml:"user-invocable"` // nil = true (default)
	Model                  string            `yaml:"model"`
	License                string            `yaml:"license"`
	Compatibility          string            `yaml:"compatibility"`
	Metadata               map[string]string `yaml:"metadata"`

	// ai-sdk-go extension: backs this skill with a registered AIFunction
	Function string `yaml:"function"`
}

// SkillRecord is a parsed, discovered skill ready for use.
type SkillRecord struct {
	Frontmatter SkillFrontmatter
	Type        SkillType
	SkillDir    string // absolute path to the skill directory
	SkillMDPath string // absolute path to SKILL.md
	Scope       SkillScope
}

// IsUserInvocable reports whether the skill should appear in the CLI/REPL menu.
// Returns true when the user-invocable frontmatter field is absent (the default).
func (r *SkillRecord) IsUserInvocable() bool {
	if r.Frontmatter.UserInvocable == nil {
		return true
	}
	return *r.Frontmatter.UserInvocable
}

// IsModelInvocable reports whether the skill should appear in the LLM catalog.
// Returns true unless disable-model-invocation is explicitly set.
func (r *SkillRecord) IsModelInvocable() bool {
	return !r.Frontmatter.DisableModelInvocation
}
