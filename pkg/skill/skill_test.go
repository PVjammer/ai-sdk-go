package skill_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/pvjammer/ai-sdk-go/pkg/skill"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func writeSkillMD(t *testing.T, dir, name, content string) string {
	t.Helper()
	skillDir := filepath.Join(dir, name)
	require.NoError(t, os.MkdirAll(skillDir, 0o755))
	path := filepath.Join(skillDir, "SKILL.md")
	require.NoError(t, os.WriteFile(path, []byte(content), 0o644))
	return skillDir
}

const promptSkill = `---
name: debug-session
description: Start a structured debugging session.
---

Follow this debugging process: $ARGUMENTS
`

const functionSkill = `---
name: pr-summary
description: Summarize a pull request.
argument-hint: "[pr-number]"
function: summarize
---

Summarize PR $1 for a technical audience.
`

const noBodySkill = `---
name: quick-tip
description: A tip with no body.
---
`

func TestLoader_Scan(t *testing.T) {
	tmp := t.TempDir()
	writeSkillMD(t, tmp, "debug-session", promptSkill)
	writeSkillMD(t, tmp, "pr-summary", functionSkill)

	loader := skill.NewSkillLoader()
	require.NoError(t, loader.Scan([]string{tmp}))

	assert.Equal(t, 2, loader.Count())

	r, ok := loader.Get("debug-session")
	require.True(t, ok)
	assert.Equal(t, skill.SkillTypePrompt, r.Type)
	assert.Equal(t, "Start a structured debugging session.", r.Frontmatter.Description)

	r2, ok := loader.Get("pr-summary")
	require.True(t, ok)
	assert.Equal(t, skill.SkillTypeFunction, r2.Type)
	assert.Equal(t, "summarize", r2.Frontmatter.Function)
	assert.Equal(t, "[pr-number]", r2.Frontmatter.ArgumentHint)
}

func TestLoader_Catalog(t *testing.T) {
	tmp := t.TempDir()
	writeSkillMD(t, tmp, "zebra-skill", "---\nname: zebra-skill\ndescription: Zebra.\n---\nBody.\n")
	writeSkillMD(t, tmp, "alpha-skill", "---\nname: alpha-skill\ndescription: Alpha.\n---\nBody.\n")

	loader := skill.NewSkillLoader()
	require.NoError(t, loader.Scan([]string{tmp}))

	catalog := loader.Catalog()
	require.Len(t, catalog, 2)
	// Catalog is sorted alphabetically
	assert.Equal(t, "alpha-skill", catalog[0].Frontmatter.Name)
	assert.Equal(t, "zebra-skill", catalog[1].Frontmatter.Name)
}

func TestLoader_CatalogXML(t *testing.T) {
	tmp := t.TempDir()
	writeSkillMD(t, tmp, "debug-session", promptSkill)

	loader := skill.NewSkillLoader()
	require.NoError(t, loader.Scan([]string{tmp}))

	xml := loader.CatalogXML()
	assert.Contains(t, xml, "<available_skills>")
	assert.Contains(t, xml, "<name>debug-session</name>")
	assert.Contains(t, xml, "Start a structured debugging session.")
	assert.Contains(t, xml, "activate_skill")
}

func TestLoader_DisableModelInvocation(t *testing.T) {
	content := `---
name: private-skill
description: Hidden from model.
disable-model-invocation: true
---
Body.
`
	tmp := t.TempDir()
	writeSkillMD(t, tmp, "private-skill", content)

	loader := skill.NewSkillLoader()
	require.NoError(t, loader.Scan([]string{tmp}))

	// Should still be in catalog (discoverable)
	_, ok := loader.Get("private-skill")
	assert.True(t, ok)

	// Should not appear in catalog XML
	xml := loader.CatalogXML()
	assert.NotContains(t, xml, "private-skill")
}

func TestLoader_UserInvocable(t *testing.T) {
	content := `---
name: llm-only
description: LLM-only skill.
user-invocable: false
---
Body.
`
	tmp := t.TempDir()
	writeSkillMD(t, tmp, "llm-only", content)

	loader := skill.NewSkillLoader()
	require.NoError(t, loader.Scan([]string{tmp}))

	r, ok := loader.Get("llm-only")
	require.True(t, ok)
	assert.False(t, r.IsUserInvocable())
	assert.True(t, r.IsModelInvocable()) // should still appear in catalog XML
}

func TestReadBody(t *testing.T) {
	tmp := t.TempDir()
	skillDir := writeSkillMD(t, tmp, "debug-session", promptSkill)
	mdPath := filepath.Join(skillDir, "SKILL.md")

	body, err := skill.ReadBody(mdPath)
	require.NoError(t, err)
	assert.Contains(t, body, "Follow this debugging process")
	assert.NotContains(t, body, "---")
	assert.NotContains(t, body, "name: debug-session")
}

func TestReadBody_NoBody(t *testing.T) {
	tmp := t.TempDir()
	skillDir := writeSkillMD(t, tmp, "quick-tip", noBodySkill)
	mdPath := filepath.Join(skillDir, "SKILL.md")

	body, err := skill.ReadBody(mdPath)
	require.NoError(t, err)
	assert.Equal(t, "", body)
}

func TestSubstitute(t *testing.T) {
	body := "Process $ARGUMENTS. First arg: $0. Second: $ARGUMENTS[1]. Dir: ${SKILL_DIR}. Session: ${SESSION_ID}."

	result := skill.Substitute(body, []string{"foo", "bar"}, "/skills/test", "session-abc")

	assert.Equal(t, "Process foo bar. First arg: foo. Second: bar. Dir: /skills/test. Session: session-abc.", result)
}

func TestSubstitute_OutOfBounds(t *testing.T) {
	body := "First: $0. Missing: $5. Also missing: $ARGUMENTS[3]."
	result := skill.Substitute(body, []string{"only-one"}, "/dir", "")
	assert.Equal(t, "First: only-one. Missing: . Also missing: .", result)
}

func TestSubstitute_NoArgs(t *testing.T) {
	body := "No args here: $ARGUMENTS."
	result := skill.Substitute(body, nil, "/dir", "")
	assert.Equal(t, "No args here: .", result)
}

func TestLoader_ScanMissingDir(t *testing.T) {
	loader := skill.NewSkillLoader()
	// Should not error on missing directories
	err := loader.Scan([]string{"/nonexistent/path/that/does/not/exist"})
	assert.NoError(t, err)
	assert.Equal(t, 0, loader.Count())
}

func TestLoader_MissingName(t *testing.T) {
	content := `---
description: No name field.
---
Body.
`
	tmp := t.TempDir()
	writeSkillMD(t, tmp, "unnamed", content)

	loader := skill.NewSkillLoader()
	require.NoError(t, loader.Scan([]string{tmp}))

	// Should be skipped due to missing name
	assert.Equal(t, 0, loader.Count())
}
