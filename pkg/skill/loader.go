package skill

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"gopkg.in/yaml.v3"
)

// scanPath pairs a directory with its scope priority.
type scanPath struct {
	dir   string
	scope SkillScope
}

// SkillLoader discovers and manages skills from the filesystem.
// It is safe for concurrent reads after Scan() completes.
type SkillLoader struct {
	mu      sync.RWMutex
	catalog []SkillRecord           // ordered by name
	index   map[string]*SkillRecord // name → record (highest-priority wins)
}

// NewSkillLoader creates an empty SkillLoader ready for Scan().
func NewSkillLoader() *SkillLoader {
	return &SkillLoader{
		index: make(map[string]*SkillRecord),
	}
}

// defaultScanPaths returns the five standard scan paths in priority order.
// Higher-priority paths override lower-priority paths on name collision.
func defaultScanPaths() []scanPath {
	home, _ := os.UserHomeDir()
	cwd, _ := os.Getwd()

	paths := []scanPath{}

	// Priority 2: ~/.config/ai/skills/
	if home != "" {
		cfgDir := os.Getenv("XDG_CONFIG_HOME")
		if cfgDir == "" {
			cfgDir = filepath.Join(home, ".config")
		}
		paths = append(paths, scanPath{
			dir:   filepath.Join(cfgDir, "ai", "skills"),
			scope: SkillScopeUserSDK,
		})

		// Priority 3: ~/.agents/skills/
		paths = append(paths, scanPath{
			dir:   filepath.Join(home, ".agents", "skills"),
			scope: SkillScopeUserPortable,
		})
	}

	// Priority 4: .agents/skills/ (project-portable)
	if cwd != "" {
		paths = append(paths, scanPath{
			dir:   filepath.Join(cwd, ".agents", "skills"),
			scope: SkillScopeProjectPortable,
		})

		// Priority 5: .claude/skills/ (project-SDK, highest)
		paths = append(paths, scanPath{
			dir:   filepath.Join(cwd, ".claude", "skills"),
			scope: SkillScopeProjectSDK,
		})
	}

	return paths
}

// Scan walks the standard skill directories plus any extraPaths, parsing
// SKILL.md files and building the in-memory catalog.
//
// On name collision the higher-priority (higher SkillScope value) entry wins.
// Warnings are printed to stderr for collisions.
func (l *SkillLoader) Scan(extraPaths []string) error {
	paths := defaultScanPaths()

	// Extra paths from config use project-SDK scope (highest user-configurable priority)
	for _, p := range extraPaths {
		if p == "" {
			continue
		}
		expanded := expandHome(p)
		paths = append(paths, scanPath{dir: expanded, scope: SkillScopeProjectSDK})
	}

	newIndex := make(map[string]*SkillRecord)
	var newCatalog []SkillRecord

	for _, sp := range paths {
		entries, err := os.ReadDir(sp.dir)
		if err != nil {
			// Directory not existing is normal; skip silently
			continue
		}

		for _, entry := range entries {
			if !entry.IsDir() {
				continue
			}

			skillDir := filepath.Join(sp.dir, entry.Name())
			skillMD := filepath.Join(skillDir, "SKILL.md")

			if _, err := os.Stat(skillMD); err != nil {
				continue
			}

			fm, err := parseFrontmatter(skillMD)
			if err != nil {
				fmt.Fprintf(os.Stderr, "warning: skill %s: %v\n", skillMD, err)
				continue
			}

			if fm.Name == "" {
				fmt.Fprintf(os.Stderr, "warning: skill %s: missing 'name' in frontmatter\n", skillMD)
				continue
			}

			// Determine skill type
			skillType := SkillTypePrompt
			if fm.Function != "" {
				skillType = SkillTypeFunction
			} else if hasScripts(skillDir) {
				skillType = SkillTypeScript
			}

			record := SkillRecord{
				Frontmatter: *fm,
				Type:        skillType,
				SkillDir:    skillDir,
				SkillMDPath: skillMD,
				Scope:       sp.scope,
			}

			// Collision check: keep the higher-priority entry
			if existing, exists := newIndex[fm.Name]; exists {
				if sp.scope > existing.Scope {
					fmt.Fprintf(os.Stderr, "warning: skill %q: project-scope overrides user-scope\n", fm.Name)
					newIndex[fm.Name] = &record
				} else {
					fmt.Fprintf(os.Stderr, "warning: skill %q: ignoring lower-priority duplicate at %s\n", fm.Name, skillMD)
				}
			} else {
				newIndex[fm.Name] = &record
			}
		}
	}

	// Build sorted catalog slice
	names := make([]string, 0, len(newIndex))
	for name := range newIndex {
		names = append(names, name)
	}
	sortStrings(names)

	for _, name := range names {
		newCatalog = append(newCatalog, *newIndex[name])
	}

	l.mu.Lock()
	l.catalog = newCatalog
	l.index = newIndex
	l.mu.Unlock()

	return nil
}

// Catalog returns all discovered skills, sorted by name.
func (l *SkillLoader) Catalog() []SkillRecord {
	l.mu.RLock()
	defer l.mu.RUnlock()
	result := make([]SkillRecord, len(l.catalog))
	copy(result, l.catalog)
	return result
}

// Get retrieves a skill by name. Returns the record and true if found.
func (l *SkillLoader) Get(name string) (*SkillRecord, bool) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	r, ok := l.index[name]
	if !ok {
		return nil, false
	}
	// Return a copy so callers can't mutate the index
	copy := *r
	return &copy, true
}

// ReadBody reads and returns the full markdown body of a skill (everything after
// the closing frontmatter delimiter). Returns an empty string if no body exists.
func ReadBody(skillMDPath string) (string, error) {
	data, err := os.ReadFile(skillMDPath)
	if err != nil {
		return "", fmt.Errorf("reading %s: %w", skillMDPath, err)
	}

	body := extractBody(data)
	return body, nil
}

// CatalogXML returns the <available_skills> XML block for injection into the
// LLM system prompt. Skills with disable-model-invocation are excluded.
func (l *SkillLoader) CatalogXML() string {
	l.mu.RLock()
	defer l.mu.RUnlock()

	if len(l.catalog) == 0 {
		return ""
	}

	var visible []SkillRecord
	for _, r := range l.catalog {
		if r.IsModelInvocable() {
			visible = append(visible, r)
		}
	}

	if len(visible) == 0 {
		return ""
	}

	var b strings.Builder
	b.WriteString("<available_skills>\n")
	for _, r := range visible {
		b.WriteString("  <skill>\n")
		b.WriteString(fmt.Sprintf("    <name>%s</name>\n", xmlEscape(r.Frontmatter.Name)))
		b.WriteString(fmt.Sprintf("    <description>%s</description>\n", xmlEscape(r.Frontmatter.Description)))
		b.WriteString(fmt.Sprintf("    <location>%s</location>\n", xmlEscape(r.SkillMDPath)))
		b.WriteString("  </skill>\n")
	}
	b.WriteString("</available_skills>\n")
	b.WriteString("To activate a skill, call the activate_skill tool with the skill name. Only load skills relevant to the current task.")

	return b.String()
}

// Count returns the number of discovered skills.
func (l *SkillLoader) Count() int {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return len(l.catalog)
}

// parseFrontmatter opens a SKILL.md file and unmarshals its YAML frontmatter.
func parseFrontmatter(path string) (*SkillFrontmatter, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading: %w", err)
	}

	fmBytes, err := extractFrontmatter(data)
	if err != nil {
		return nil, err
	}

	var fm SkillFrontmatter
	if err := yaml.Unmarshal(fmBytes, &fm); err != nil {
		return nil, fmt.Errorf("parsing frontmatter: %w", err)
	}

	return &fm, nil
}

// extractFrontmatter returns the raw YAML bytes between the first and second
// "---" delimiters in the file. Returns an error if no valid frontmatter found.
func extractFrontmatter(data []byte) ([]byte, error) {
	scanner := bufio.NewScanner(bytes.NewReader(data))

	var inFrontmatter bool
	var fmLines []string
	dashCount := 0

	for scanner.Scan() {
		line := scanner.Text()
		trimmed := strings.TrimSpace(line)

		if trimmed == "---" {
			dashCount++
			if dashCount == 1 {
				inFrontmatter = true
				continue
			}
			if dashCount == 2 {
				// End of frontmatter
				break
			}
		}

		if inFrontmatter {
			fmLines = append(fmLines, line)
		}
	}

	if dashCount < 2 {
		return nil, fmt.Errorf("no YAML frontmatter found (need opening and closing '---')")
	}

	return []byte(strings.Join(fmLines, "\n")), nil
}

// extractBody returns everything after the second "---" delimiter.
func extractBody(data []byte) string {
	scanner := bufio.NewScanner(bytes.NewReader(data))
	dashCount := 0
	var bodyLines []string
	pastFrontmatter := false

	for scanner.Scan() {
		line := scanner.Text()
		trimmed := strings.TrimSpace(line)

		if !pastFrontmatter && trimmed == "---" {
			dashCount++
			if dashCount == 2 {
				pastFrontmatter = true
			}
			continue
		}

		if pastFrontmatter {
			bodyLines = append(bodyLines, line)
		}
	}

	return strings.TrimSpace(strings.Join(bodyLines, "\n"))
}

// hasScripts reports whether a skill directory contains a non-empty scripts/ subdirectory.
func hasScripts(skillDir string) bool {
	entries, err := os.ReadDir(filepath.Join(skillDir, "scripts"))
	return err == nil && len(entries) > 0
}

// expandHome replaces a leading ~ with the user's home directory.
func expandHome(path string) string {
	if !strings.HasPrefix(path, "~") {
		return path
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return path
	}
	return filepath.Join(home, path[1:])
}

// xmlEscape escapes characters that would break XML text content.
func xmlEscape(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	s = strings.ReplaceAll(s, ">", "&gt;")
	return s
}

// sortStrings sorts a string slice in place (avoids importing sort for a single call).
func sortStrings(ss []string) {
	for i := 1; i < len(ss); i++ {
		for j := i; j > 0 && ss[j] < ss[j-1]; j-- {
			ss[j], ss[j-1] = ss[j-1], ss[j]
		}
	}
}
