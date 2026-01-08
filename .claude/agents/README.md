# dimtensor Specialized Agents

Agents tailored for the dimtensor project. Use these when you need specialized expertise.

## Available Agents

| Agent | When to Use |
|-------|-------------|
| `code-reviewer` | Review code for correctness, patterns, edge cases |
| `test-writer` | Write unit tests for new functionality |
| `units-designer` | Design new unit modules (dimensions, scales, conversions) |
| `explorer` | Explore codebase to understand patterns and find code |
| `planner` | Create detailed implementation plans |

## Usage

Spawn an agent using the Task tool:

```
Task tool with subagent_type="general-purpose" and prompt referencing the agent file
```

Or ask the worker to "use the code-reviewer agent to review this file."

## Agent Files

Each agent file contains:
- `name`: Agent identifier
- `description`: When to use this agent
- `model`: Which model to use (sonnet for most, opus for complex)
- Instructions for the agent's behavior
