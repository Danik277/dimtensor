# dimtensor Skills

Specialized knowledge for working on dimtensor.

## Available Skills

| Skill | Description |
|-------|-------------|
| `units-design` | Design new unit modules (dimensions, scales, patterns) |
| `deploy` | Deploy to PyPI |
| `code-review` | Review code for correctness and patterns |

## How Skills Work

Skills are **automatically detected** by Claude based on your request. When you say something that matches a skill's description, Claude will load the skill's instructions.

Examples:
- "Design astronomy units" → loads `units-design` skill
- "Deploy v1.2.0 to PyPI" → loads `deploy` skill
- "Review the new xarray module" → loads `code-review` skill

## Skills vs Agents vs Plans

| Mechanism | Purpose | When to Use |
|-----------|---------|-------------|
| **Skills** | Knowledge/guidance | "How do I do X?" |
| **Plans** | Document decisions | Before creating new files |
| **Agents** | Separate execution | Complex isolated tasks |
| **CONTINUITY.md** | Track state | Always (main workflow) |

## Creating New Skills

1. Create folder: `.claude/skills/skill-name/`
2. Create `SKILL.md` with frontmatter:
   ```yaml
   ---
   name: skill-name
   description: When to use this skill (max 1024 chars)
   allowed-tools: Read, Write, Edit  # optional
   ---
   ```
3. Add instructions in markdown
