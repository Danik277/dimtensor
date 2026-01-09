# dimtensor Agent System

## Architecture: Orchestrator + Sub-Agents

The main agent is an **ORCHESTRATOR** that spawns specialized sub-agents for parallel work.

```
┌─────────────────┐
│  ORCHESTRATOR   │  Reads queue, spawns agents, updates CONTINUITY.md
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────┐
    ▼         ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│planner │ │implmtr │ │tester  │ │deployer│
└────────┘ └────────┘ └────────┘ └────────┘
```

## Available Agents

| Agent | File | Purpose |
|-------|------|---------|
| `orchestrator` | orchestrator.md | Main controller, spawns sub-agents |
| `planner` | planner.md | Creates plans for 🗺️ tasks |
| `implementer` | implementer.md | Implements features |
| `test-writer` | test-writer.md | Writes tests |
| `code-reviewer` | code-reviewer.md | Reviews code quality |
| `deployer` | deployer.md | Deploys to PyPI |

## How It Works

1. **Orchestrator reads CONTINUITY.md** to find pending tasks
2. **Identifies parallelizable work** (e.g., plan next feature while implementing current)
3. **Spawns sub-agents using Task tool** - multiple in ONE message for parallelism
4. **Waits for results** and updates CONTINUITY.md
5. **Repeats** until all tasks are DONE

## Spawning Sub-Agents

Use the Task tool with `subagent_type="general-purpose"`:

```
Task tool call:
- description: "Plan model hub feature"
- prompt: "You are a PLANNER for dimtensor. Read .claude/agents/planner.md for instructions. Task: Create plan for task #102 (model hub architecture). Return the plan file path when done."
- subagent_type: "general-purpose"
```

## Parallel Execution

Spawn multiple agents in ONE message for parallel work:

```
Message with 3 Task tool calls:
1. planner → task #105 (equation database plan)
2. implementer → task #103 (model hub - plan exists)
3. test-writer → tests for visualization module
```

## Rules

1. **Orchestrator updates CONTINUITY.md** - sub-agents don't
2. **Sub-agents return results** - orchestrator records them
3. **Spawn parallel when possible** - maximize throughput
4. **Keep going until queue empty** - don't stop after versions
