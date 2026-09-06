from __future__ import annotations

"""General-128 V0.8 language-cortex prototype.

Architecture:
raw text -> small language cortex -> semantic/reasoning bridge -> General-128 core
-> recurrent verifier -> raw text / benchmark answer.

This is a research prototype.  The language cortex is an external local
OpenAI-compatible endpoint (used with SmolLM2-360M-Instruct Q4_K_M in CI).
The 128-unit core keeps persistent state and executes safe arithmetic programs
through explicit operator events.  It does not use benchmark labels.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import ast
import json
import re
import urllib.request

from general128_v07 import General128V07, SemanticEvent

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class LanguageCortex:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.calls = 0

    def call(self, system: str, user: str, max_tokens: int = 96) -> str:
        body = json.dumps({
            "model": "local-model",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint, data=body,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read().decode("utf-8"))
        self.calls += 1
        return data["choices"][0]["message"]["content"]


def parse_letter(text: str, nchoices: int) -> Optional[int]:
    t = text.strip().upper()
    hits = re.findall(r"(?:FINAL|ANSWER)?\s*[:=]?\s*\b([A-Z])\b", t)
    for h in reversed(hits):
        j = ord(h) - 65
        if 0 <= j < nchoices:
            return j
    return None


def _extract_number(text: str) -> Optional[str]:
    t = text.replace(",", "")
    pats = [
        r"FINAL(?:\s+ANSWER)?\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
        r"ANSWER\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
        r"####\s*([-+]?\d+(?:\.\d+)?)",
    ]
    for p in pats:
        m = re.findall(p, t, flags=re.I)
        if m:
            return m[-1]
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", t)
    return nums[-1] if nums else None


def _num_equal(a: Optional[str], b: Optional[str], tol: float = 1e-7) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))
    except Exception:
        return False


class SafeArithmetic128:
    """Safe AST arithmetic executor routed through General-128 semantic events."""

    ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)

    def __init__(self, core: General128V07):
        self.core = core
        self.ops = 0

    def evaluate(self, expression: str) -> float:
        tree = ast.parse(expression.strip(), mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, self.ALLOWED):
                raise ValueError(f"unsupported syntax: {type(node).__name__}")
        return float(self._eval(tree.body))

    def _event(self, opname: str, a: float, b: Optional[float], out: float) -> None:
        payload = {"a": a, "out": out}
        if b is not None:
            payload["b"] = b
        self.core.encode_event(SemanticEvent(
            relation=f"arith:{opname}", quantity=out,
            goal="verified_arithmetic", payload=payload,
        ))
        self.ops += 1

    def _eval(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp):
            a = self._eval(node.operand)
            if isinstance(node.op, ast.USub):
                out = -a
                self._event("neg", a, None, out)
                return out
            if isinstance(node.op, ast.UAdd):
                return a
        if isinstance(node, ast.BinOp):
            a, b = self._eval(node.left), self._eval(node.right)
            if isinstance(node.op, ast.Add): out, name = a + b, "add"
            elif isinstance(node.op, ast.Sub): out, name = a - b, "sub"
            elif isinstance(node.op, ast.Mult): out, name = a * b, "mul"
            elif isinstance(node.op, ast.Div):
                if b == 0: raise ZeroDivisionError
                out, name = a / b, "div"
            elif isinstance(node.op, ast.Pow):
                if abs(b) > 12: raise ValueError("power too large")
                out, name = a ** b, "pow"
            else:
                raise ValueError("operator not allowed")
            if abs(out) > 1e18:
                raise ValueError("result too large")
            self._event(name, a, b, out)
            return out
        raise ValueError("unsupported expression")


@dataclass
class V08Result:
    answer: Any
    confidence: float
    cycles: int
    cortex_calls: int
    trace: List[str]


class General128V08:
    def __init__(self, cortex_endpoint: str):
        self.core = General128V07()
        self.cortex = LanguageCortex(cortex_endpoint)
        self.arithmetic = SafeArithmetic128(self.core)

    @staticmethod
    def _mc_prompt(question: str, choices: Sequence[str]) -> str:
        opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
        return f"{question}\n\n{opts}"

    def answer_mc(self, question: str, choices: Sequence[str]) -> V08Result:
        start_calls = self.cortex.calls
        prompt = self._mc_prompt(question, choices)
        out1 = self.cortex.call(
            "Solve the multiple-choice question. Think internally and output only FINAL: <letter>.",
            prompt, 32,
        )
        p1 = parse_letter(out1, len(choices))
        self.core.encode_event(SemanticEvent(
            relation="candidate_answer", goal="multiple_choice",
            confidence=0.72, payload={"candidate": p1, "pass": 1},
        ))

        out2 = self.cortex.call(
            "Independently reconsider every option. Avoid anchoring on a previous guess. Output only FINAL: <letter>.",
            prompt, 48,
        )
        p2 = parse_letter(out2, len(choices))
        self.core.encode_event(SemanticEvent(
            relation="candidate_answer", goal="multiple_choice",
            confidence=0.78, payload={"candidate": p2, "pass": 2},
        ))

        trace = [f"pass1={p1}", f"pass2={p2}"]
        if p1 is not None and p1 == p2:
            return V08Result(p1, 0.92, 2, self.cortex.calls-start_calls, trace+["consensus"])

        candidates = sorted({p for p in (p1, p2) if p is not None})
        hint = " / ".join(LETTERS[p] for p in candidates) if candidates else "none"
        out3 = self.cortex.call(
            "Act as a final verifier. Re-solve the question; candidate passes may be wrong. Output only FINAL: <letter>.",
            prompt + f"\n\nEarlier candidate letters: {hint}", 64,
        )
        p3 = parse_letter(out3, len(choices))
        self.core.encode_event(SemanticEvent(
            relation="verified_answer", goal="multiple_choice",
            confidence=0.82, payload={"candidate": p3, "pass": 3},
        ))
        trace.append(f"verifier={p3}")
        final = p3 if p3 is not None else (p2 if p2 is not None else p1)
        return V08Result(final, 0.82 if p3 is not None else 0.55, 3,
                         self.cortex.calls-start_calls, trace)

    @staticmethod
    def _extract_expr(text: str) -> Optional[str]:
        # Prefer explicit EXPR: marker; accept a JSON {"expression":"..."} fallback.
        m = re.search(r"EXPR\s*:\s*([^\n]+)", text, flags=re.I)
        if m:
            s = m.group(1).strip().strip('`').strip()
            if len(s) <= 220:
                return s
        m = re.search(r'"expression"\s*:\s*"([^"]+)"', text)
        return m.group(1).strip() if m else None

    def answer_math(self, question: str) -> V08Result:
        start_calls = self.cortex.calls
        direct = self.cortex.call(
            "Solve the grade-school math problem carefully. End with exactly FINAL: <number>.",
            question, 220,
        )
        direct_num = _extract_number(direct)

        expr_out = self.cortex.call(
            "Translate the word problem into ONE arithmetic expression using only numeric constants, parentheses, +, -, *, /, and **. Do not do the arithmetic yourself. End with EXPR: <expression>. If impossible, write EXPR: INVALID.",
            question, 160,
        )
        expr = self._extract_expr(expr_out)
        core_num: Optional[str] = None
        trace = [f"direct={direct_num}", f"expr={expr}"]
        if expr and expr.upper() != "INVALID":
            try:
                val = self.arithmetic.evaluate(expr)
                if abs(val-round(val)) < 1e-9:
                    core_num = str(int(round(val)))
                else:
                    core_num = ("%.10f" % val).rstrip("0").rstrip(".")
                trace.append(f"core={core_num}")
            except Exception as e:
                trace.append(f"core_error={type(e).__name__}")

        if _num_equal(direct_num, core_num):
            return V08Result(core_num, 0.95, 2, self.cortex.calls-start_calls, trace+["numeric_consensus"])
        if core_num is None:
            return V08Result(direct_num, 0.65, 2, self.cortex.calls-start_calls, trace+["direct_fallback"])

        verify = self.cortex.call(
            "You are a strict math verifier. Solve the original problem yourself, then choose which candidate numerical answer is correct. End with exactly FINAL: <number>.",
            question + f"\n\nCandidate A: {direct_num}\nCandidate B: {core_num}", 220,
        )
        vnum = _extract_number(verify)
        trace.append(f"verifier={vnum}")
        if _num_equal(vnum, core_num):
            final = core_num
        elif _num_equal(vnum, direct_num):
            final = direct_num
        else:
            final = vnum if vnum is not None else core_num
        self.core.encode_event(SemanticEvent(
            relation="verified_answer", goal="gsm8k", confidence=0.84,
            payload={"direct": direct_num, "core": core_num, "final": final},
        ))
        return V08Result(final, 0.84, 3, self.cortex.calls-start_calls, trace)

    def snapshot(self) -> Dict[str, Any]:
        s = self.core.snapshot()
        s.update({
            "version": "0.8",
            "raw_text_interface": True,
            "language_cortex_calls": self.cortex.calls,
            "arithmetic_core_ops": self.arithmetic.ops,
        })
        return s
