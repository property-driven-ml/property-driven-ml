"""
Safe expression evaluation utilities.

This module provides safe alternatives to eval() for parsing user-controlled strings
containing mathematical expressions and function calls.
"""

import ast
from typing import Any, Dict, Tuple, Union


def _is_literal(node: ast.AST) -> bool:
    """Check if an AST node represents a literal value."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float, str, type(None)))
    if isinstance(node, ast.Tuple):
        return all(_is_literal(elt) for elt in node.elts)
    if isinstance(node, ast.Name) and node.id == 'inf':
        return True
    return False


def _literal_value(node: ast.AST) -> Any:
    """Extract the literal value from an AST node."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(_literal_value(elt) for elt in node.elts)
    if isinstance(node, ast.Name) and node.id == 'inf':
        return float('inf')
    raise ValueError('Only literal values are allowed in arguments')


def safe_call(expr: str, allowed: Dict[str, Any]) -> Any:
    """
    Safely evaluate a function call expression.
    
    Args:
        expr: String expression containing a function call
        allowed: Dictionary mapping function names to callable objects
        
    Returns:
        Result of the function call
        
    Raises:
        ValueError: If the expression is malformed or contains disallowed operations
    """
    tree = ast.parse(expr, mode='eval')
    if not isinstance(tree, ast.Expression) or not isinstance(tree.body, ast.Call):
        raise ValueError('Only simple function calls are allowed')
    
    call = tree.body
    if not isinstance(call.func, ast.Name):
        raise ValueError('Only direct calls to allowed functions are permitted')
    
    func_name = call.func.id
    if func_name not in allowed:
        raise ValueError(f"Function '{func_name}' is not allowed")
    
    if not all(_is_literal(a) for a in call.args):
        raise ValueError('Only literal positional arguments are allowed')
    
    if not all(isinstance(kw, ast.keyword) and kw.arg and _is_literal(kw.value) for kw in call.keywords):
        raise ValueError('Only literal keyword arguments are allowed')
    
    args = [_literal_value(a) for a in call.args]
    kwargs = {}
    for kw in call.keywords:
        if kw.arg is None:  # **kwargs arguments not allowed
            raise ValueError('**kwargs arguments are not allowed')
        kwargs[kw.arg] = _literal_value(kw.value)
    
    return allowed[func_name](*args, **kwargs)


def _eval_arith(node: ast.AST, context: Dict[str, float]) -> float:
    """
    Safely evaluate an arithmetic expression.
    
    Args:
        node: AST node representing the expression
        context: Dictionary of variable names to values
        
    Returns:
        Evaluated numeric result
        
    Raises:
        ValueError: If the expression contains unsupported operations
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    
    if isinstance(node, ast.Name):
        if node.id == 'inf':
            return float('inf')
        if node.id in context:
            return float(context[node.id])
        raise ValueError(f"Unknown variable '{node.id}' in bounds")
    
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval_arith(node.operand, context)
        return v if isinstance(node.op, ast.UAdd) else -v
    
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        l = _eval_arith(node.left, context)
        r = _eval_arith(node.right, context)
        if isinstance(node.op, ast.Add):
            return l + r
        if isinstance(node.op, ast.Sub):
            return l - r
        if isinstance(node.op, ast.Mult):
            return l * r
        return l / r
    
    raise ValueError(f'Unsupported expression: {ast.dump(node)}')


def safe_bounds(expr: str, context: Dict[str, float]) -> Tuple[float, float]:
    """
    Safely evaluate a bounds expression like "(-1, 1)".
    
    Args:
        expr: String expression containing bounds tuple
        context: Dictionary of variable names to values
        
    Returns:
        Tuple of (lower_bound, upper_bound)
        
    Raises:
        ValueError: If the expression is malformed
    """
    tree = ast.parse(expr, mode='eval')
    if not isinstance(tree, ast.Expression) or not isinstance(tree.body, ast.Tuple) or len(tree.body.elts) != 2:
        raise ValueError('Bounds must be a 2-tuple')
    
    lo = _eval_arith(tree.body.elts[0], context)
    hi = _eval_arith(tree.body.elts[1], context)
    return (lo, hi)
