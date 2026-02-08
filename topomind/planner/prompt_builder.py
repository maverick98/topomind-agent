from typing import List
from ..tools.schema import Tool


class PlannerPromptBuilder:
    """
    Builds a minimal routing prompt.
    Planner is ONLY responsible for tool selection.
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
            tool_blocks.append("\n".join(block))

        tool_desc = "\n\n".join(tool_blocks)

        return f"""
You are a routing engine.

Your job is to select the SINGLE most appropriate tool.

You DO NOT generate code.
You DO NOT generate DSL.
You DO NOT explain execution contracts.

Return STRICT JSON:
{{ 
  "tool": "...",
  "args": {{...}},
  "reasoning": "...",
  "confidence": 0.0-1.0
}}

IMPORTANT:
- Output MUST be valid JSON.
- Use only double quotes.
- Do NOT use triple quotes.
- args must contain raw user input only.
- If stable context contains "previous_tool",
  you MUST choose a different tool.

User request:
"{user_input}"

Stable context:
{signals}

Available tools:
{tool_desc}
""".strip()
