"""Safe arithmetic: AST whitelist + Decimal. No eval/exec, no names, no attributes.

Grammar: numbers, + - * / // % ** (integer exponent, |exp| <= 12), unary +/-,
parentheses, and the functions round(x[, n]), abs(x), min(...), max(...)."""
from __future__ import annotations

import ast
import operator
from decimal import ROUND_HALF_UP, Decimal, DecimalException, localcontext

from ..exceptions import CalculatorError

MAX_LEN, MAX_NODES, MAX_LITERAL, MAX_MAGNITUDE = 200, 120, 40, 100   # |literal| < 10**101
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}


def _num(node: ast.expr, src: str) -> Decimal:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        text = (ast.get_source_segment(src, node) or "").strip()
        if not text or len(text) > MAX_LITERAL:
            raise CalculatorError("bad number")
        d = Decimal(text)
        if not d.is_finite() or abs(d.adjusted()) > MAX_MAGNITUDE:
            raise CalculatorError("number out of range")
        return d
    raise CalculatorError("unsupported syntax")


def _ev(node: ast.expr, src: str) -> Decimal:
    if isinstance(node, ast.Constant):
        return _num(node, src)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _ev(node.operand, src)
        return -v if isinstance(node.op, ast.USub) else +v
    if isinstance(node, ast.BinOp):
        a, b = _ev(node.left, src), _ev(node.right, src)
        if isinstance(node.op, ast.Pow):
            if b != b.to_integral_value() or abs(b) > 12:
                raise CalculatorError("exponent must be an integer within +-12")
            return a ** int(b)
        op = _BIN.get(type(node.op))
        if op is None:
            raise CalculatorError("unsupported operator")
        return op(a, b)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        name, args = node.func.id, [_ev(a, src) for a in node.args]
        if name == "abs" and len(args) == 1:
            return abs(args[0])
        if name in ("min", "max") and 1 <= len(args) <= 8:
            return (min if name == "min" else max)(args)
        if name == "round" and len(args) in (1, 2):
            n = int(args[1]) if len(args) == 2 and args[1] == args[1].to_integral_value() else 0
            if not 0 <= n <= 12:
                raise CalculatorError("round digits must be within 0..12")
            return args[0].quantize(Decimal(1).scaleb(-n), rounding=ROUND_HALF_UP)
        raise CalculatorError("unsupported function")
    raise CalculatorError("unsupported syntax")


def evaluate(expression) -> str:
    if not isinstance(expression, str) or not expression.strip() or len(expression) > MAX_LEN:
        raise CalculatorError("expression must be a non-empty string up to 200 chars")
    src = expression.strip()
    try:
        tree = ast.parse(src, mode="eval")
    except (SyntaxError, ValueError):
        raise CalculatorError("syntax error") from None
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise CalculatorError("expression too complex")
    try:
        with localcontext() as ctx:
            ctx.prec, ctx.Emax, ctx.Emin = 50, 1000, -1000
            v = _ev(tree.body, src)
            if v == v.to_integral_value():
                return str(int(v))
            return format(v.normalize(), "f")
    except CalculatorError:
        raise
    except (DecimalException, ZeroDivisionError, OverflowError, ValueError):
        raise CalculatorError("arithmetic error") from None
