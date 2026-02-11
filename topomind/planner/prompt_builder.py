from typing import List
from ..tools.schema import Tool


class PlannerPromptBuilder:
    """
    Responsible for constructing the planner prompt.

    IMPORTANT:
    Planner is ONLY responsible for tool selection.
    It must NOT include execution-layer contracts (e.g. DSL rules).
    """

    def build(
        self,
        user_input: str,
        signals,
        tools: List[Tool],
    ) -> str:

        # Deterministic ordering
        tools = sorted(tools, key=lambda t: t.name)

        tool_blocks = []

        for t in tools:
            block = []
            block.append(f"- {t.name}")
            block.append(f"  Description: {t.description}")
            block.append(f"  Inputs: {t.input_schema}")

            if getattr(t, "strict", False):
                block.append("  STRICT: Tool requires valid argument structure.")

            tool_blocks.append("\n".join(block))

        tool_desc = "\n\n".join(tool_blocks)

        constraint_block = """
CRITICAL CONSTRAINTS:

1. You MUST choose EXACTLY one tool from the list below.
2. You MUST NOT invent new tool names.
3. You MUST NOT rename tools.
4. You MUST return STRICT JSON.
5. If you output a tool name not in the list, the system will fail.
"""

        return f"""
You are the planning engine of an AI agent.

Your job is to select the SINGLE most appropriate tool
to handle the user request.

You DO NOT generate answers.
You DO NOT produce natural language responses.
You ONLY choose one tool and provide arguments.

Return STRICT JSON:
{{ "tool": "...", "args": {{...}}, "reasoning": "...", "confidence": 0.0-1.0 }}

{constraint_block}

User request:
"{user_input}"

Stable context:
{signals}

Available tools:
{tool_desc}
""".strip()
