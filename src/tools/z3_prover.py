import ast
from types import SimpleNamespace

import z3


_ALLOWED_NAMES = {"p", "q", "x", "y"}
_ALLOWED_Z3_CALLS = {"And", "Or", "Not", "Implies", "Xor", "If"}
_ALLOWED_NODES = {
    ast.Expression, ast.Call, ast.Name, ast.Load, ast.Attribute,
    ast.Compare, ast.Eq, ast.NotEq, ast.BoolOp, ast.And, ast.Or,
    ast.UnaryOp, ast.Not, ast.Constant,
}


def _validate_logic_expression(logic_str: str) -> ast.Expression:
    if not isinstance(logic_str, str) or not logic_str.strip():
        raise ValueError("logic expression must be a non-empty string")
    if len(logic_str) > 4_096:
        raise ValueError("logic expression exceeds the 4096-character limit")

    tree = ast.parse(logic_str, mode="eval")
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            raise ValueError(f"unsupported expression element: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES | {"z3"}:
            raise ValueError(f"unknown name: {node.id}")
        if isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name) or node.value.id != "z3":
                raise ValueError("attribute access is restricted to approved z3 functions")
            if node.attr not in _ALLOWED_Z3_CALLS:
                raise ValueError(f"z3 function is not allowed: {node.attr}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Attribute):
                raise ValueError("only approved z3 function calls are allowed")
            if node.keywords:
                raise ValueError("keyword arguments are not allowed")
    return tree

def prove_theorem(logic_str: str) -> str:
    """
    Evaluates a Z3 logical constraint string.
    Example logic_str: "z3.Not(z3.And(p, q)) == z3.Or(z3.Not(p), z3.Not(q))"
    """
    try:
        p, q, x, y = z3.Bools('p q x y')
        tree = _validate_logic_expression(logic_str)
        namespace = {
            "z3": SimpleNamespace(**{name: getattr(z3, name) for name in _ALLOWED_Z3_CALLS}),
            "p": p,
            "q": q,
            "x": x,
            "y": y,
        }
        constraint = eval(compile(tree, "<z3-expression>", "eval"), {"__builtins__": {}}, namespace)
        if not z3.is_bool(constraint):
            raise ValueError("expression must produce a Z3 Boolean")
        s = z3.Solver()
        # To prove a theorem, we test if its negation is unsatisfiable
        s.add(z3.Not(constraint))
        result = s.check()
        
        if result == z3.unsat:
            return "PROVED (Unsatisfiable Negation)"
        elif result == z3.sat:
            return f"DISPROVED. Counter-example: {s.model()}"
        else:
            return "UNKNOWN"
    except Exception as e:
        return f"Error proving theorem: {e}"
