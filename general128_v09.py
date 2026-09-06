from __future__ import annotations

"""General-128 V0.9: selective cognitive augmentation.

The 0.5B language cortex is left untouched on general language/knowledge tasks.
General-128 is invoked only for tasks where an explicit structure can be
extracted and verified (currently grade-school arithmetic reasoning).

This is a system-level research prototype, not a smaller standalone LLM.
"""

import ast
import re
from collections import Counter
from typing import Optional

import general128_v07 as v07
from general128_v07 import General128V07, Primitive, SemanticEvent
from general128_v08 import LanguageCortex, V08Result, _extract_number, _num_equal


# Extend the behavior library before recruiting operator families.  Family
# identity is still learned from demonstrations rather than selected by label.
if not any(p.name == "div" for p in v07.PRIMITIVES):
    v07.PRIMITIVES.append(Primitive("div", lambda a, b: a / b if b != 0 else float("inf")))
if not any(p.name == "pow" for p in v07.PRIMITIVES):
    v07.PRIMITIVES.append(Primitive("pow", lambda a, b: a ** b if abs(b) <= 12 else float("inf")))


class CausalArithmetic128:
    ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)

    def __init__(self, core: General128V07):
        self.core = core
        self.ops = 0
        demos = {
            "g_add": [(2, 3, 5), (7, -2, 5), (-4, -3, -7)],
            "g_sub": [(8, 3, 5), (2, 7, -5), (-4, -3, -1)],
            "g_mul": [(2, 3, 6), (7, -2, -14), (-4, -3, 12)],
            "g_div": [(8, 2, 4), (9, 3, 3), (-6, 2, -3)],
            "g_pow": [(2, 3, 8), (3, 2, 9), (5, 1, 5)],
        }
        self.learned = {s: self.core.learn_operator(s, ex) for s, ex in demos.items()}

    @staticmethod
    def _clean(expr: str) -> str:
        x = expr.strip().strip('`').replace('^', '**')
        x = x.split('FINAL:')[0].strip()
        x = re.sub(r"\s+$", "", x)
        return x[:300]

    def evaluate(self, expr: str) -> float:
        tree = ast.parse(self._clean(expr), mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, self.ALLOWED):
                raise ValueError(type(node).__name__)
        return float(self._eval(tree.body))

    def _apply(self, symbol: str, a: float, b: float) -> float:
        if symbol == "g_div" and abs(b) < 1e-15:
            raise ZeroDivisionError
        r = self.core.numeric(symbol, a, b)
        self.ops += 1
        if not r.verified:
            raise ValueError("operator verification failed")
        self.core.encode_event(SemanticEvent(
            relation=f"causal:{symbol}", quantity=float(r.answer),
            goal="math_program", confidence=r.confidence,
            payload={"a": a, "b": b, "family_trace": r.trace},
        ))
        return float(r.answer)

    def _eval(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp):
            a = self._eval(node.operand)
            if isinstance(node.op, ast.USub): return self._apply("g_sub", 0.0, a)
            if isinstance(node.op, ast.UAdd): return a
        if isinstance(node, ast.BinOp):
            a, b = self._eval(node.left), self._eval(node.right)
            if isinstance(node.op, ast.Add): return self._apply("g_add", a, b)
            if isinstance(node.op, ast.Sub): return self._apply("g_sub", a, b)
            if isinstance(node.op, ast.Mult): return self._apply("g_mul", a, b)
            if isinstance(node.op, ast.Div): return self._apply("g_div", a, b)
            if isinstance(node.op, ast.Pow): return self._apply("g_pow", a, b)
        raise ValueError("unsupported expression")


class General128V09:
    def __init__(self, cortex_endpoint: str):
        self.cortex = LanguageCortex(cortex_endpoint)
        self.core = General128V07()
        self.math = CausalArithmetic128(self.core)

    @staticmethod
    def _expr(text: str) -> Optional[str]:
        m = re.search(r"EXPR\s*:\s*([^\n]+)", text, re.I)
        if not m:
            return None
        x = m.group(1).strip().strip('`').strip()
        return None if x.upper().startswith("INVALID") else x

    @staticmethod
    def _fmt(x: float) -> str:
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return ("%.10f" % x).rstrip('0').rstrip('.')

    def answer_math(self, question: str) -> V08Result:
        calls0 = self.cortex.calls
        trace = []
        direct = self.cortex.call(
            "Solve the problem carefully. End with exactly FINAL: <number>.",
            question, 220,
        )
        direct_n = _extract_number(direct)
        trace.append(f"direct={direct_n}")

        prompts = [
            "Convert the entire word problem into one arithmetic expression using ONLY numeric constants, parentheses, +, -, *, /, **. Include every required step and no prose. End exactly EXPR: <expression>.",
            "Independently derive a single arithmetic expression for the requested final quantity. Use only numbers, parentheses, +, -, *, /, **. Check units and rates before answering. End exactly EXPR: <expression>.",
        ]
        core_candidates = []
        for k, sys in enumerate(prompts, 1):
            txt = self.cortex.call(sys, question, 180)
            expr = self._expr(txt)
            trace.append(f"expr{k}={expr}")
            if expr:
                try:
                    val = self.math.evaluate(expr)
                    n = self._fmt(val)
                    core_candidates.append(n)
                    trace.append(f"core{k}={n}")
                except Exception as e:
                    trace.append(f"core{k}_error={type(e).__name__}")

        if len(core_candidates) >= 2 and _num_equal(core_candidates[0], core_candidates[1]):
            final = core_candidates[0]
            conf = 0.96
            trace.append("core_consensus")
        elif core_candidates and direct_n is not None and any(_num_equal(direct_n, x) for x in core_candidates):
            final = direct_n
            conf = 0.93
            trace.append("direct_core_consensus")
        elif not core_candidates:
            final = direct_n
            conf = 0.62
            trace.append("direct_fallback")
        else:
            choices = []
            if direct_n is not None: choices.append(direct_n)
            choices += core_candidates
            # de-duplicate numerically as strings for the small candidate set
            uniq=[]
            for x in choices:
                if not any(_num_equal(x,y) for y in uniq): uniq.append(x)
            candidate_text = ", ".join(f"C{i+1}={x}" for i,x in enumerate(uniq))
            verify = self.cortex.call(
                "Re-solve the original math problem from scratch. Candidate answers are only hints and may all be wrong. End exactly FINAL: <number>.",
                question + "\n\nCandidates: " + candidate_text, 240,
            )
            v = _extract_number(verify)
            trace.append(f"verifier={v}")
            final = v if v is not None else (core_candidates[0] if core_candidates else direct_n)
            conf = 0.82

        self.core.encode_event(SemanticEvent(
            relation="final_math_decision", goal="gsm8k", confidence=conf,
            payload={"direct": direct_n, "core": core_candidates, "final": final},
        ))
        return V08Result(final, conf, min(4, self.cortex.calls-calls0), self.cortex.calls-calls0, trace)

    def snapshot(self):
        s = self.core.snapshot()
        s.update({"version":"0.9","cortex_calls":self.cortex.calls,"causal_math_ops":self.math.ops})
        return s
