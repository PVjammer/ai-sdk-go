// Package skills provides AIFunction implementations for skill discovery and activation.
package skills

import (
	"context"
	"fmt"
	"reflect"
	"strings"

	"github.com/pvjammer/ai-sdk-go/internal/schema"
	"github.com/pvjammer/ai-sdk-go/pkg/function"
	"github.com/pvjammer/ai-sdk-go/pkg/skill"
)

// ActivateSkillFunction loads a skill by name and returns its body with variable substitution applied.
// This is the primary mechanism by which the LLM activates a skill.
type ActivateSkillFunction struct {
	function.Base
	loader *skill.SkillLoader
}

// ActivateInput is the input schema for activate_skill.
type ActivateInput struct {
	Name      string   `json:"name" validate:"required" description:"Skill name from the available_skills catalog"`
	Arguments []string `json:"arguments" description:"Arguments passed to the skill for $ARGUMENTS substitution"`
}

// ActivateOutput is the structured result returned by activate_skill.
type ActivateOutput struct {
	// Content is the skill body with all variable substitutions applied.
	// The caller (agent) should inject this into the conversation.
	Content string `json:"content"`

	// Function is non-empty for Type B (function-backed) skills.
	// It names the AIFunction registered in the tool registry that the model should call.
	Function string `json:"function,omitempty"`

	// SkillDir is the absolute path to the skill's directory.
	// Available to the model for referencing bundled scripts via ${SKILL_DIR}.
	SkillDir string `json:"skill_dir"`

	// SkillType indicates the skill type: "prompt", "function", or "script".
	SkillType string `json:"skill_type"`
}

func newActivateSkill(loader *skill.SkillLoader, cfg function.Config) function.AIFunction {
	return &ActivateSkillFunction{
		Base:   function.NewBase(cfg),
		loader: loader,
	}
}

// Execute implements the AIFunction interface.
func (f *ActivateSkillFunction) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	if f.loader == nil {
		return nil, fmt.Errorf("activate_skill: skill loader not initialized")
	}

	in, err := f.parseInput(input)
	if err != nil {
		return nil, err
	}

	record, ok := f.loader.Get(in.Name)
	if !ok {
		// Helpful error: list available skills
		catalog := f.loader.Catalog()
		names := make([]string, len(catalog))
		for i, r := range catalog {
			names[i] = r.Frontmatter.Name
		}
		return nil, fmt.Errorf("skill %q not found; available: %s", in.Name, strings.Join(names, ", "))
	}

	// Respect user-invocable: false — LLM should not activate user-hidden skills
	// (user-invocable: false means only the user can trigger it, not the model)
	// However: disable-model-invocation is a stronger signal; user-invocable is about UI.
	// We block model access only for disable-model-invocation here.
	if record.Frontmatter.DisableModelInvocation {
		return nil, fmt.Errorf("skill %q cannot be activated by the model (disable-model-invocation: true)", in.Name)
	}

	// Read skill body
	body, err := skill.ReadBody(record.SkillMDPath)
	if err != nil {
		return nil, fmt.Errorf("reading skill body: %w", err)
	}

	// Retrieve session ID from context (may be empty if not set)
	sessionID := skill.SessionIDFromCtx(ctx)

	// Apply variable substitution
	body = skill.Substitute(body, in.Arguments, record.SkillDir, sessionID)

	return &ActivateOutput{
		Content:   body,
		Function:  record.Frontmatter.Function,
		SkillDir:  record.SkillDir,
		SkillType: string(record.Type),
	}, nil
}

// parseInput handles both typed *ActivateInput and map[string]interface{} (from LLM tool calls).
func (f *ActivateSkillFunction) parseInput(input interface{}) (*ActivateInput, error) {
	switch v := input.(type) {
	case *ActivateInput:
		return v, nil

	case map[string]interface{}:
		in := &ActivateInput{}

		if name, ok := v["name"].(string); ok {
			in.Name = name
		}
		if in.Name == "" {
			return nil, fmt.Errorf("activate_skill: 'name' is required")
		}

		// Arguments may arrive as []interface{} from JSON tool calls
		switch args := v["arguments"].(type) {
		case []string:
			in.Arguments = args
		case []interface{}:
			for _, a := range args {
				if s, ok := a.(string); ok {
					in.Arguments = append(in.Arguments, s)
				}
			}
		}

		return in, nil

	default:
		return nil, fmt.Errorf("activate_skill: unsupported input type %T", input)
	}
}

// InputSchema implements the AIFunction interface.
func (f *ActivateSkillFunction) InputSchema() *function.Schema {
	return schema.MustGenerateSchema(reflect.TypeOf(ActivateInput{}))
}

// Metadata implements the AIFunction interface.
func (f *ActivateSkillFunction) Metadata() *function.Metadata {
	return function.NewMetadata("ActivateSkillFunction").
		WithName("activate_skill").
		WithToolName("activate_skill").
		WithToolDescription("Load a skill by name and make its instructions available. Call this when a skill in the available_skills catalog is relevant to the current task.").
		WithCLIName("activate_skill").
		WithCategory("skills").
		WithTags("skills", "meta")
}
