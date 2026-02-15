import json
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

        signals = signals or {}
        tools = sorted(tools, key=lambda t: t.name)

        tool_blocks = []

        for t in tools:
            block = []
            block.append(f"- {t.name}")
            block.append(f"  Description: {t.description}")
            block.append(
                f"  Inputs (JSON schema): {json.dumps(t.input_schema, indent=2)}"
            )

            if getattr(t, "strict", False):
                block.append("  STRICT: Tool requires valid argument structure.")

            tool_blocks.append("\n".join(block))

        tool_desc = "\n\n".join(tool_blocks)

        return f"""
You are the planning engine of an AI agent.

Your job is to select EXACTLY ONE tool call
to satisfy the user request.

You DO NOT generate answers.
You DO NOT execute tools.
You ONLY select the correct tool and provide arguments.

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
- You MUST return EXACTLY ONE step.
- Tools must be selected ONLY from the available list.
- Do NOT invent tool names.
- Arguments MUST strictly match the input schema.
- No markdown.
- No explanation outside JSON.

User request:
"{user_input}"

Stable context (JSON):
{json.dumps(signals, indent=2)}

Available tools:
{tool_desc}
""".strip()
