import json
import uuid
import logging
import time
from typing import List, Optional

from ..interface import ReasoningEngine
from ..plan_model import Plan, PlanStep
from ..prompt_builder import PlannerPromptBuilder
from ...models.tool_call import ToolCall
from ...tools.schema import Tool
from ...agent.llm.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Safe JSON Extraction
# ------------------------------------------------------------

def extract_first_json(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None

    stack = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            stack += 1
        elif text[i] == "}":
            stack -= 1
            if stack == 0:
                return text[start:i + 1]

    return None


