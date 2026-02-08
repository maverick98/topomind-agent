class OllamaPlanner(ReasoningEngine):

    def __init__(self, model: str = "mistral"):
        self.model = model
        self.url = "http://localhost:11434/api/chat"
        self.prompt_builder = PlannerPromptBuilder()

    # -------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------

    def generate_plan(self, user_input: str, signals, tools: List[Tool]) -> Plan:
        tools = sorted(tools, key=lambda t: t.name)

        prompt = self._build_prompt(user_input, signals, tools)
        response_text = self._call_llm(prompt)

        try:
            return self._parse_llm_response(response_text, user_input, tools)
        except Exception as e:
            logger.error(f"[PLANNER ERROR] {e}")
            logger.error(f"Failed planner output: {response_text}")
            return self._fallback_plan()

    # -------------------------------------------------------------
    # Internal Responsibilities
    # -------------------------------------------------------------

    def _build_prompt(self, user_input: str, signals, tools: List[Tool]) -> str:
        prompt = self.prompt_builder.build(
            user_input=user_input,
            signals=signals,
            tools=tools,
        )
        logger.debug("----- PLANNER PROMPT -----")
        logger.debug(prompt)
        return prompt

    def _call_llm(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        response = requests.post(
            self.url,
            json=payload,
            proxies={"http": None, "https": None},
        )

        text = response.json().get("message", {}).get("content", "").strip()
        logger.debug(f"Planner raw LLM output: {text}")
        return text

    def _parse_llm_response(
        self,
        text: str,
        user_input: str,
        tools: List[Tool],
    ) -> Plan:

        result = json.loads(text)

        tool_name = result.get("tool", "echo")
        args = result.get("args", {})

        logger.info(f"[PLANNER] Tool chosen: {tool_name}")
        logger.info(f"[PLANNER] Confidence: {result.get('confidence')}")
        logger.info(f"[PLANNER] Reasoning: {result.get('reasoning')}")

        if not args:
            tool_obj = next((t for t in tools if t.name == tool_name), None)
            if tool_obj and tool_obj.input_schema:
                first_param = list(tool_obj.input_schema.keys())[0]
                args = {first_param: user_input}

        step = PlanStep(
            action=ToolCall(
                id=str(uuid.uuid4()),
                tool_name=tool_name,
                arguments=args,
            ),
            reasoning=result.get("reasoning", "LLM decision"),
            confidence=float(result.get("confidence", 0.7)),
        )

        return Plan(steps=[step], goal="LLM-driven planning")

    def _fallback_plan(self) -> Plan:
        step = PlanStep(
            action=ToolCall(
                id=str(uuid.uuid4()),
                tool_name="echo",
                arguments={"text": "Planner failed"},
            ),
            reasoning="Fallback due to LLM parse failure.",
            confidence=0.2,
        )
        return Plan(steps=[step], goal="Fallback")
