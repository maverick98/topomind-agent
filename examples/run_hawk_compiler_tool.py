from topomind.connectors.manager import ConnectorManager
from topomind.connectors.ollama import OllamaConnector
from topomind.tools.registry import ToolRegistry
from topomind.tools.schema import Tool
from topomind.tools.executor import ToolExecutor
from topomind.agent.core import Agent
from topomind.planner.adapters.ollama import OllamaPlanner
from topomind.models.tool_result import ToolResult
from topomind.connectors.base import ExecutionConnector


import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ============================================================
# Hawk Print Connector (NO execution — just print DSL)
# ============================================================

from topomind.connectors.base import ExecutionConnector


class HawkPrintConnector(ExecutionConnector):
    """
    Does NOT execute Hawk.
    Only prints generated DSL.
    """

    def execute(self, tool, arguments):
        code = arguments.get("code", "")

        print("\n================ GENERATED HAWK CODE ================\n")
        print(code)
        print("\n=====================================================\n")

        return ToolResult(
            tool_name=tool.name,
            tool_version=tool.version,
            status="success",
            output={"code": code},
            error=None,
            latency_ms=0,
            stability_signal=1.0,
        )


# ============================================================
# YOUR ORIGINAL PROMPT — UNCHANGED
# ============================================================

HAWK_COMPILER_PROMPT = """You are a Hawk DSL code generation engine.

You DO NOT answer in natural language.
You DO NOT explain.
You DO NOT call external tools.
You DO NOT emit JSON.
You DO NOT emit markdown.

You ONLY emit valid Hawk DSL code.

Your output must be a complete Hawk program.

------------------------------------------------------------
ROLE
------------------------------------------------------------

You are a deterministic compiler frontend.
You convert user intent into executable Hawk DSL.

The backend exposes a SINGLE tool:

executeHawk(code: string)

The LLM does NOT call tools directly.
The platform will pass your entire output to executeHawk.

Therefore:
- Output MUST be pure Hawk code
- Output MUST be syntactically valid
- Output MUST be semantically valid
- No commentary allowed

------------------------------------------------------------
HAWK LANGUAGE GRAMMAR (EBNF)
------------------------------------------------------------

program         ::= function_def+

function_def    ::= "function" identifier "(" ")" block

block           ::= "{" statement* "}"

statement       ::= var_decl
                  | assignment
                  | exec_stmt
                  | return_stmt
                  | if_stmt
                  | for_stmt
                  | expr_stmt

var_decl        ::= "var" identifier "=" expression
assignment      ::= identifier "=" expression
exec_stmt       ::= "exec" identifier "(" argument_list? ")"
return_stmt     ::= "return" expression

if_stmt         ::= "if" "(" condition ")" block
for_stmt        ::= "for" "(" assignment ";" condition ";" assignment ")" block

expr_stmt       ::= expression

condition       ::= expression comparator expression

comparator      ::= "==" | "!=" | ">" | "<" | ">=" | "<="

expression      ::= literal
                  | identifier
                  | identifier "(" argument_list? ")"
                  | expression operator expression

operator        ::= "+" | "-" | "*" | "/"

argument_list   ::= expression ("," expression)*

literal         ::= number | string

identifier      ::= letter (letter | digit | "_")*
number          ::= digit+
string          ::= "\"" (any_char)* "\""

------------------------------------------------------------
SEMANTIC RULES
------------------------------------------------------------

1. Entry point MUST be:
   function main()

2. All programs MUST start execution from main()

3. No undefined variables allowed.

4. No undefined function calls allowed.

5. All variables must be declared before use.

6. Only allowed top-level functions:
   - main
   - user-defined helpers

7. No recursion unless explicitly required.

8. No external APIs unless defined in grammar.

9. No dynamic code construction.

10. Deterministic output only.

------------------------------------------------------------
PLANNING RULES
------------------------------------------------------------

Before emitting code:
- Understand user intent.
- Translate intent into procedural logic.
- Break into deterministic steps.
- Prefer simple constructs.
- Avoid unnecessary loops.
- Avoid redundant variables.

If task is impossible in Hawk grammar:
Emit minimal valid Hawk program:

function main() {
}

------------------------------------------------------------
OUTPUT FORMAT
------------------------------------------------------------

Output ONLY valid Hawk code.
No markdown.
No explanation.
No comments.
No extra text.
"""

# ============================================================
# Infrastructure
# ============================================================

connectors = ConnectorManager()

# Planner LLM connector
connectors.register("llm", OllamaConnector(model="mistral"))


# Hawk print connector (execution phase)
connectors.register("hawk_print", HawkPrintConnector())

registry = ToolRegistry()

registry.register(
    Tool(
        name="executeHawk",
        description="Generate Hawk DSL program",
        input_schema={"code": "string"},
        output_schema={"code": "string"},
        connector_name="hawk_print",
        prompt=HAWK_COMPILER_PROMPT,
        strict=True,
    )
)

executor = ToolExecutor(connectors=connectors, registry=registry)

#planner = OllamaPlanner(model="mistral")
planner = OllamaPlanner(model="phi3:mini")


agent = Agent(planner=planner, executor=executor)

# ============================================================
# Run Hawk DSL Generation Tests
# ============================================================

print("\n=== Hawk DSL Generation Tests ===\n")

queries = [
       "Calculate sum of 1 to 5",
]

for q in queries:
    print(f"\n>>> {q}")
    result = agent.handle_query(q)
    print("ToolResult:", result)

print("\n=== End Hawk Tests ===\n")
