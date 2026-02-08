from typing import List
from ..tools.schema import Tool


class PlannerPromptBuilder:
    """
    Responsible for constructing the full planner prompt.
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

            if getattr(t, "prompt", None):
                block.append("  Execution Contract:")
                block.append("  " + t.prompt.strip().replace("\n", "\n  "))

            if getattr(t, "strict", False):
                block.append("  STRICT: This tool requires exact contract adherence.")

            tool_blocks.append("\n".join(block))

        tool_desc = "\n\n".join(tool_blocks)

        strict_enabled = any(getattr(t, "strict", False) for t in tools)

        strict_block = ""
        if strict_enabled:
            strict_block = (
                "\nSTRICT MODE ENABLED\n"
                "Tool contracts must be followed exactly.\n"
                "Invalid arguments may cause execution failure.\n"
            )

        constraint_block = """
CRITICAL CONSTRAINTS:

1. You MUST choose EXACTLY one tool from the list below.
2. You MUST NOT invent new tool names.
3. You MUST NOT rename tools.
4. If the request could be handled in multiple ways,
   you STILL must choose only from the listed tools.
5. If you output a tool name not in the list,
   the system will fail.
"""

        return f"""
You are the planning engine of an AI agent.

Your job is to select the SINGLE most appropriate tool to handle the user request.

You DO NOT generate answers.
You DO NOT produce natural language responses.
You ONLY choose one tool and provide arguments.

If a tool has an "Execution Contract",
you must respect that contract when forming arguments.

If a tool is marked STRICT,
violating its execution contract may cause runtime failure.

IMPORTANT:
If stable context contains "previous_tool" or "previous_error",
it means the previous tool choice failed.
You MUST choose a DIFFERENT tool if possible.

Return STRICT JSON:
{{ "tool": "...", "args": {{...}}, "reasoning": "...", "confidence": 0.0-1.0 }}

{constraint_block}

User request:
"{user_input}"

Stable context:
{signals}

Available tools:
{tool_desc}

{strict_block}
""".strip()
