package skill

import (
	"regexp"
	"strconv"
	"strings"
)

var (
	reArgumentsN = regexp.MustCompile(`\$ARGUMENTS\[(\d+)\]`)
	reArgN       = regexp.MustCompile(`\$(\d+)`)
)

// Substitute applies variable substitution to a skill body before it reaches the LLM.
//
// Variables replaced:
//
//	${SKILL_DIR}       → absolute path to the skill's directory
//	${SESSION_ID}      → current agent session ID
//	$ARGUMENTS         → all args joined with a space
//	$ARGUMENTS[N]      → the Nth argument (0-indexed), empty string if out of bounds
//	$N                 → shorthand for $ARGUMENTS[N]
//
// Substitution is applied in the order listed above.
// Shell injection (!`cmd`) is not executed in this implementation.
func Substitute(body string, args []string, skillDir, sessionID string) string {
	// 1. Named variables
	body = strings.ReplaceAll(body, "${SKILL_DIR}", skillDir)
	body = strings.ReplaceAll(body, "${SESSION_ID}", sessionID)

	// 2. $ARGUMENTS[N] — must come BEFORE plain $ARGUMENTS to avoid corrupting
	//    the indexed form (e.g. "$ARGUMENTS[1]" must not be half-replaced first).
	body = reArgumentsN.ReplaceAllStringFunc(body, func(match string) string {
		sub := reArgumentsN.FindStringSubmatch(match)
		if len(sub) < 2 {
			return match
		}
		n, err := strconv.Atoi(sub[1])
		if err != nil || n < 0 || n >= len(args) {
			return ""
		}
		return args[n]
	})

	// 3. $ARGUMENTS → all args joined (after indexed form is handled)
	allArgs := strings.Join(args, " ")
	body = strings.ReplaceAll(body, "$ARGUMENTS", allArgs)

	// 4. $N shorthand — only replace bare $<digit> patterns that aren't part of
	// a larger identifier (e.g. avoid breaking $ARGUMENTS already replaced above)
	body = reArgN.ReplaceAllStringFunc(body, func(match string) string {
		sub := reArgN.FindStringSubmatch(match)
		if len(sub) < 2 {
			return match
		}
		n, err := strconv.Atoi(sub[1])
		if err != nil || n < 0 || n >= len(args) {
			return ""
		}
		return args[n]
	})

	return body
}
