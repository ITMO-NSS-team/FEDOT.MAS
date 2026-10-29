"""Bounded decimal arithmetic with runtime receipts; no Python eval or I/O."""

import ast
import json
from decimal import Decimal, InvalidOperation, localcontext, ROUND_HALF_EVEN

from google.adk.tools import ToolContext

CALCULATIONS_KEY = "__fedotmas_calculations"


def evaluate(expression: str, inputs: dict, decimal_places: int | None = None) -> str:
    if len(expression) > 1000 or len(inputs) > 32:
        raise ValueError("Calculation is too large")
    if decimal_places is not None and (
        type(decimal_places) is not int or not -12 <= decimal_places <= 12
    ):
        raise ValueError("decimal_places must be an integer from -12 to 12")

    def number(value):
        if (
            isinstance(value, bool)
            or not isinstance(value, (str, int, float))
            or len(str(value)) > 128
        ):
            raise ValueError("Inputs must be finite decimal numbers")
        result = Decimal(str(value))
        if not result.is_finite() or abs(result.adjusted()) > 200:
            raise ValueError("Input is outside the numeric range")
        return result

    variables = {key: number(value) for key, value in inputs.items()}
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError("Expression is too complex")

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return number(ast.get_source_segment(expression, node))
        if isinstance(node, ast.Name) and node.id in variables:
            return variables[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if (
                isinstance(node.op, ast.Pow)
                and right == right.to_integral_value()
                and abs(right) <= 32
            ):
                return left ** int(right)
        raise ValueError(
            "Use numbers, named inputs, parentheses and + - * / integer powers only"
        )

    with localcontext() as ctx:
        ctx.prec = 50
        ctx.rounding = ROUND_HALF_EVEN
        ctx.Emax = 400
        ctx.Emin = -400
        result = visit(tree.body)
        if decimal_places is not None:
            result = result.quantize(Decimal(1).scaleb(-decimal_places))
        if not result.is_finite() or abs(result.adjusted()) > 200:
            raise ValueError("Result is outside the numeric range")
        return format(result, "f")


def calculate(
    expression: str,
    inputs_json: str,
    tool_context: ToolContext,
    decimal_places: int | None = None,
) -> dict:
    """Compute arithmetic from explicit inputs. Returns a decimal string.

    Pass inputs_json as a JSON object mapping variable names to numeric values
    from evidence. Preserve units and apply requested rounding via decimal_places.
    Supports +, -, *, / and bounded integer powers; no code or external access.
    """
    try:
        if len(inputs_json) > 10000:
            raise ValueError("inputs_json is too large")
        inputs = json.loads(inputs_json)
        if not isinstance(inputs, dict):
            raise ValueError("inputs_json must be an object")
        result = evaluate(expression, inputs, decimal_places)
    except (ValueError, TypeError, SyntaxError, ArithmeticError, RecursionError) as exc:
        return {"status": "error", "error": str(exc)[:300]}
    receipt = {
        "expression": expression,
        "inputs": inputs,
        "decimal_places": decimal_places,
        "result": result,
    }
    root = dict(tool_context.state.get(CALCULATIONS_KEY) or {})
    root[tool_context.agent_name] = [
        *(root.get(tool_context.agent_name) or []),
        receipt,
    ][-16:]
    tool_context.state[CALCULATIONS_KEY] = root
    return {"status": "completed", **receipt}


def result_matches(value, result: str) -> bool:
    try:
        return not isinstance(value, bool) and Decimal(str(value)) == Decimal(result)
    except (InvalidOperation, ValueError):
        return False
