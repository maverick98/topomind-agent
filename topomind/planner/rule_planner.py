from .interface import ReasoningEngine


class RuleBasedPlanner(ReasoningEngine):
    def generate_plan(self, user_input: str, signals, tools):
        stable = signals.get("stable_entities", [])

        if stable:
            return {
                "tool": "echo",
                "args": {"text": f"Still talking about: {stable[0]}"}
            }

        if "hello" in user_input.lower():
            return {"tool": "echo", "args": {"text": "Hello from TopoMind Planner!"}}

        return {"tool": "echo", "args": {"text": f"You said: {user_input}"}}
