import ast
import operator
import math


class SafeExpressionEvaluator(ast.NodeVisitor):
    """
    Safe arithmetic expression evaluator using AST Visitor pattern.
    Only allows whitelisted operators and math functions.
    """

    ALLOWED_BINOPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
    }

    ALLOWED_UNARYOPS = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    ALLOWED_FUNCTIONS = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "exp": math.exp,
    }

    MAX_DEPTH = 25

    def evaluate(self, expression: str):
        tree = ast.parse(expression, mode="eval")
        self._check_depth(tree)
        return self.visit(tree.body)

    # ---------------- Depth Guard ----------------

    def _check_depth(self, node, depth=0):
        if depth > self.MAX_DEPTH:
            raise ValueError("Expression too complex")
        for child in ast.iter_child_nodes(node):
            self._check_depth(child, depth + 1)

    # ---------------- Visitors ----------------

    def visit_BinOp(self, node):
        op_type = type(node.op)
        if op_type not in self.ALLOWED_BINOPS:
            raise ValueError("Operator not allowed")

        left = self.visit(node.left)
        right = self.visit(node.right)
        return self.ALLOWED_BINOPS[op_type](left, right)

    def visit_UnaryOp(self, node):
        op_type = type(node.op)
        if op_type not in self.ALLOWED_UNARYOPS:
            raise ValueError("Unary operator not allowed")

        operand = self.visit(node.operand)
        return self.ALLOWED_UNARYOPS[op_type](operand)

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Attribute):
            raise ValueError("Only math.<function>() allowed")

        if not isinstance(node.func.value, ast.Name):
            raise ValueError("Invalid function call")

        if node.func.value.id != "math":
            raise ValueError("Only math module allowed")

        func_name = node.func.attr

        if func_name not in self.ALLOWED_FUNCTIONS:
            raise ValueError(f"Function '{func_name}' not allowed")

        args = [self.visit(arg) for arg in node.args]
        return self.ALLOWED_FUNCTIONS[func_name](*args)

    def visit_Name(self, node):
        if node.id == "math":
            return math
        raise ValueError("Variables not allowed")

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants allowed")

    def visit_Num(self, node):  # legacy Python support
        return node.n

    def generic_visit(self, node):
        raise ValueError(f"Unsupported expression: {type(node).__name__}")
