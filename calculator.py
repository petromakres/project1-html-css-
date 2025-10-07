#!/usr/bin/env python3
"""
CLI Calculator with a safe expression evaluator.

Features:
- Evaluate arithmetic expressions safely (no eval)
- Supports +, -, *, /, //, %, **, parentheses, unary +/-, integers and floats
- REPL mode (interactive) or one-off evaluation via --expr

Usage examples:
  python3 calculator.py --expr "2 + 3 * 4"
  python3 calculator.py
    calc> (10 - 3) * 2
    14
    calc> quit

This tool intentionally rejects names, function calls, attributes, and any non-arithmetic constructs.
"""
from __future__ import annotations

import argparse
import ast
import math
import sys
from typing import Union, Any

Number = Union[int, float]


class EvaluationError(Exception):
    pass


class SafeEvaluator(ast.NodeVisitor):
    """Evaluate arithmetic expressions using Python's AST safely.

    Only a restricted set of nodes and operators are allowed.
    """

    MAX_ABS_VALUE = 10 ** 100  # Guard against absurdly large magnitudes
    MAX_POW_EXPONENT = 1000     # Guard against massive exponentiation

    def visit(self, node: ast.AST) -> Number:  # type: ignore[override]
        value = super().visit(node)
        self._ensure_reasonable(value)
        return value

    # Top-level Expression
    def visit_Expression(self, node: ast.Expression) -> Number:
        return self.visit(node.body)

    # Numeric literals
    def visit_Constant(self, node: ast.Constant) -> Number:
        if isinstance(node.value, (int, float)):
            return node.value
        raise EvaluationError("Only numeric literals are allowed")

    # Parentheses are implicit in the AST via grouped nodes; no special handler needed

    # Unary operations: +x, -x
    def visit_UnaryOp(self, node: ast.UnaryOp) -> Number:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise EvaluationError("Unsupported unary operator")

    # Binary operations
    def visit_BinOp(self, node: ast.BinOp) -> Number:
        left = self.visit(node.left)
        right = self.visit(node.right)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            # Guard against abusive exponentiation
            if isinstance(right, (int, float)) and abs(right) > self.MAX_POW_EXPONENT:
                raise EvaluationError("Exponent too large")
            return left ** right

        raise EvaluationError("Unsupported binary operator")

    # Disallow everything else
    def visit_Name(self, node: ast.Name) -> Number:  # type: ignore[override]
        raise EvaluationError("Names are not allowed")

    def visit_Call(self, node: ast.Call) -> Number:  # type: ignore[override]
        raise EvaluationError("Function calls are not allowed")

    def visit_Attribute(self, node: ast.Attribute) -> Number:  # type: ignore[override]
        raise EvaluationError("Attributes are not allowed")

    def visit_Subscript(self, node: ast.Subscript) -> Number:  # type: ignore[override]
        raise EvaluationError("Subscriptions are not allowed")

    def generic_visit(self, node: ast.AST) -> Number:  # type: ignore[override]
        # Anything not explicitly allowed is rejected
        raise EvaluationError(f"Unsupported syntax: {node.__class__.__name__}")

    @classmethod
    def _ensure_reasonable(cls, value: Any) -> None:
        if isinstance(value, (int, float)):
            # Prevent absurdly large numbers
            if isinstance(value, int):
                # Quick magnitude check via bit length ~ log2(|value|)
                if value.bit_length() > 4096:  # ~ 2^4096 ~ 10^1233
                    raise EvaluationError("Integer magnitude too large")
            else:
                if math.isinf(value) or math.isnan(value):
                    raise EvaluationError("Non-finite result")
                if abs(value) > cls.MAX_ABS_VALUE:
                    raise EvaluationError("Float magnitude too large")
        else:
            raise EvaluationError("Non-numeric result")


def parse_expression(expression: str) -> ast.Expression:
    try:
        parsed = ast.parse(expression, mode="eval")
        if not isinstance(parsed, ast.Expression):
            raise EvaluationError("Invalid expression")
        return parsed
    except SyntaxError as exc:
        raise EvaluationError(f"Syntax error: {exc.msg}") from exc


def evaluate_expression(expression: str) -> Number:
    tree = parse_expression(expression)
    evaluator = SafeEvaluator()
    return evaluator.visit(tree)


def format_result(value: Number) -> str:
    # Render ints without trailing .0; floats concise but precise
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        # Use a reasonable precision; strip trailing zeros
        return ("{:.12g}".format(value))
    return str(value)


def run_repl() -> int:
    print("Type expressions to evaluate. Type 'quit' or 'exit' to leave.")
    try:
        import readline  # noqa: F401  (optional, improves UX if available)
    except Exception:
        pass

    while True:
        try:
            line = input("calc> ").strip()
        except EOFError:
            print()  # newline on Ctrl-D
            return 0

        if not line:
            continue
        if line.lower() in {"quit", "exit"}:
            return 0

        try:
            result = evaluate_expression(line)
            print(format_result(result))
        except EvaluationError as exc:
            print(f"Error: {exc}", file=sys.stderr)
        except ZeroDivisionError:
            print("Error: Division by zero", file=sys.stderr)
        except Exception as exc:  # Unexpected, but keep the tool robust for CLI use
            print(f"Error: {exc.__class__.__name__}: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="calculator",
        description="CLI calculator with a safe expression evaluator",
    )
    parser.add_argument(
        "-e",
        "--expr",
        dest="expr",
        type=str,
        help="Evaluate a single expression and exit",
    )

    args = parser.parse_args(argv)

    if args.expr is not None:
        try:
            value = evaluate_expression(args.expr)
            print(format_result(value))
            return 0
        except EvaluationError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        except ZeroDivisionError:
            print("Error: Division by zero", file=sys.stderr)
            return 2

    return run_repl()


if __name__ == "__main__":
    raise SystemExit(main())
