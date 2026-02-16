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

            schema_str = json.dumps(
                {
                    k: f"<{v}>" if isinstance(v, str) else v
                    for k, v in t.input_schema.items()
                },
                indent=2,
            )

            block.append(f"  Inputs (JSON schema): {schema_str}")

            if getattr(t, "strict", False):
                block.append("  STRICT: Tool requires valid argument structure.")

            tool_blocks.append("\n".join(block))

        tool_desc = "\n\n".join(tool_blocks)

        return f"""
You are the planning engine of an AI agent.

Your job is to select the necessary tool calls
to satisfy the user request.

You DO NOT generate answers.
You DO NOT execute tools.
You ONLY select the correct tools and provide arguments.

Return STRICT JSON in this format:

{{
  "steps": [
    {{
      "tool": "tool_name",
      "args": {{}}
    }}
  ],
  "confidence": 0.0-1.0
}}

Rules:
- You MUST return valid JSON parsable by Python json.loads().
- JSON MUST contain only the keys shown in the format above.
- NO comments inside JSON.
- NO trailing commas.
- NO markdown.
- NO explanations.
- Tools must be selected ONLY from the available list.
- Do NOT invent tool names.
- Arguments MUST strictly match the input schema.
- ONLY include steps whose required arguments are fully known at planning time.
- DO NOT create steps that depend on outputs of previous tools.
- If a tool produces intermediate output needed by another tool,
  include ONLY the first tool. The executor will handle chaining.

User request:
"{user_input}"

Stable context (JSON):
{json.dumps(signals, indent=2)}

Available tools:
{tool_desc}
""".strip()
