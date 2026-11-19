import pytest

from traceforge.environment.calculator import evaluate
from traceforge.exceptions import CalculatorError


@pytest.mark.parametrize("expr,want", [
    ("1+2*3", "7"), ("(1+2)*3", "9"), ("10/4", "2.5"), ("10//4", "2"), ("10%4", "2"),
    ("2**10", "1024"), ("-5+2", "-3"), ("+3", "3"), ("0.1+0.2", "0.3"), ("1.10*3", "3.3"),
    ("1/3", "0.33333333333333333333333333333333333333333333333333"),
    ("round(2.675, 2)", "2.68"), ("round(-2.5)", "-3"), ("round(1234.5678, 0)", "1235"),
    ("abs(-7.25)", "7.25"), ("min(3, 1, 2)", "1"), ("max(3, 1.5)", "3"),
    ("1234.56 * 1.0825", "1336.4112"), ("100.00 / 3", "33.333333333333333333333333333333333333333333333333"),
    ("2 ** -2", "0.25"), ("1e3 + 1", "1001"), ("123456789012345678901234567890 + 1", "123456789012345678901234567891"),
    ("(1.08123456 * 100000) - 108123.456", "0"),
])
def test_arithmetic_is_exact_decimal(expr, want):
    assert evaluate(expr) == want


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')", "().__class__", "open('x')", "eval('1')", "exec('1')",
    "[1,2][0]", "{'a':1}", "lambda: 1", "x + 1", "print(1)", "'a' * 3", "True + 1", "1 if 1 else 2",
    "1 < 2", "not 1", "1 and 2", "(1).real", "1; 2", "import os", "a.b", "round(1, x)", "abs(1, 2)",
    "min()", "round(1, 2, 3)", "round(1, 99)", "9 ** 99", "2 ** 0.5", "1 / 0", "1 // 0", "0 / 0", "1 % 0",
    "", "   ", "1 +", "((", "1e99999999999", "10 ** 12 ** 2", "abs(lambda: 1)", "f'{1}'", "...", "1j",
    "~1", "1 << 2", "1 @ 2", "max(*[1,2])", "round(x=1)", "abs(1).real",
])
def test_unsafe_or_invalid_expressions_are_rejected(expr):
    with pytest.raises(CalculatorError):
        evaluate(expr)


def test_non_string_and_oversized_input_rejected():
    for bad in (None, 5, [], "1+" * 200):
        with pytest.raises(CalculatorError):
            evaluate(bad)


def test_complexity_limit():
    with pytest.raises(CalculatorError):
        evaluate("+".join(["1"] * 80))


def test_error_messages_are_generic_and_do_not_echo_input():
    try:
        evaluate("secret_name + 1")
    except CalculatorError as e:
        assert "secret_name" not in str(e)


def test_no_floating_point_artifacts_for_money_arithmetic():
    assert evaluate("108.12 * 100") == "10812"
    assert evaluate("round(0.615 * 1000, 0)") == "615"
    assert evaluate("round(1.005 * 100, 0) / 100") == "1.01"
