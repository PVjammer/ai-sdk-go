package skills

import (
	"github.com/pvjammer/ai-sdk-go/pkg/function"
	"github.com/pvjammer/ai-sdk-go/pkg/registry"
	"github.com/pvjammer/ai-sdk-go/pkg/skill"
)

// globalLoader is the SkillLoader instance used by activate_skill and list_skills.
// It is nil until SetLoader is called. The factory closures read it at Execute()
// time (not at init() time), so SetLoader must be called before the first tool
// invocation — which main() guarantees by setting it before rootCmd.Execute().
var globalLoader *skill.SkillLoader

// SetLoader sets the SkillLoader used by the activate_skill and list_skills tools.
// Call this from main() after SkillLoader.Scan() completes.
func SetLoader(l *skill.SkillLoader) {
	globalLoader = l
}

func init() {
	registry.Global().Register("activate_skill", func(cfg function.Config) function.AIFunction {
		return newActivateSkill(globalLoader, cfg)
	})

	registry.Global().Register("list_skills", func(cfg function.Config) function.AIFunction {
		return newListSkills(globalLoader, cfg)
	})
}
