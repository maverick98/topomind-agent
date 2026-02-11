from typing import List
from ..tools.schema import Tool


class PlannerPromptBuilder:
    """
    Constructs planner prompt.
    Planner is generic.
    No domain logic allowed.
    """

    def build(
        self,
        user_input: str,
        signals,
        tools: List[Tool],
    ) -> str:

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

        return f"""
You are the planning engine of an AI agent.

Your job is to compose one or more tool calls
to satisfy the user request.

You DO NOT generate answers.
You DO NOT execute tools.
You ONLY select tools and provide arguments.

Return STRICT JSON in this format:

{{
  "steps": [
    {{
      "tool": "tool_name",
      "args": {{ }}
    }}
  ],
  "confidence": 0.0-1.0
}}

Rules:
- You may return one or multiple steps.
- Tools must be selected ONLY from the available list.
- Do NOT invent tool names.
- Arguments must match the tool input schema.
- No markdown.
- No explanation outside JSON.

User request:
"{user_input}"

Stable context:
{signals}

Available tools:
{tool_desc}
""".strip()
