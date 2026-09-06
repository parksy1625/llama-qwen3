from __future__ import annotations

"""General-128 V0.9R1: selective Qwen cortex + causal arithmetic tool loop.

The language cortex plans in natural language, but arithmetic is delegated to an
explicit CALC protocol. Every accepted CALC expression is executed by the
General-128 learned operator-family path (`core.numeric`) rather than Python's
arithmetic operators directly.

Protocol per turn:
    CALC: <numeric expression>
    FINAL: <number>

This is a system-level research prototype, not a standalone 128-unit LLM.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from general128_v07 import General128V07, SemanticEvent
from general128_v08 import LanguageCortex, V08Result, _extract_number
from general128_v09 import CausalArithmetic128


@dataclass
class ToolTurn:
    request: str
    result: Optional[str]
    raw: str


class General128V09R1:
    def __init__(self, cortex_endpoint: str, max_tool_steps: int = 6):
        self.cortex = LanguageCortex(cortex_endpoint)
        self.core = General128V07()
        self.math = CausalArithmetic128(self.core)
        self.max_tool_steps = max_tool_steps
        self.tool_calls = 0

    @staticmethod
    def _fmt(x: float) -> str:
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return ("%.10f" % x).rstrip("0").rstrip(".")

    @staticmethod
    def _calc(text: str) -> Optional[str]:
        # Keep the protocol deliberately narrow: one CALC line, no prose after it.
        m = re.search(r"(?:^|\n)\s*CALC\s*:\s*([^\n]+)", text, re.I)
        if not m:
            return None
        expr = m.group(1).strip().strip("`").strip()
        if not expr or len(expr) > 220:
            return None
        return expr

    @staticmethod
    def _final(text: str) -> Optional[str]:
        # Require an explicit final marker in the tool loop.
        m = re.findall(r"(?:^|\n)\s*FINAL\s*:\s*([-+]?\d+(?:\.\d+)?)\s*(?:$|\n)", text, re.I)
        return m[-1] if m else None

    def answer_math(self, question: str) -> V08Result:
        calls0 = self.cortex.calls
        ops0 = self.math.ops
        transcript: List[str] = ["PROBLEM:\n" + question]
        trace: List[str] = []
        turns: List[ToolTurn] = []

        system = (
            "Solve the grade-school arithmetic word problem using the calculator tool. "
            "Do NOT perform arithmetic mentally. On each turn output exactly one line: "
            "either CALC: <numeric expression> using only numbers, parentheses, +, -, *, /, **, "
            "or FINAL: <number>. Use CALC for every arithmetic operation needed. "
            "After receiving RESULT, continue from that numeric result."
        )

        final: Optional[str] = None
        for step in range(self.max_tool_steps):
            user = "\n\n".join(transcript) + "\n\nNEXT ACTION:"
            raw = self.cortex.call(system, user, 80)
            final = self._final(raw)
            if final is not None:
                trace.append(f"step{step+1}:final={final}")
                turns.append(ToolTurn("FINAL", final, raw))
                break

            expr = self._calc(raw)
            if expr is None:
                trace.append(f"step{step+1}:protocol_error")
                turns.append(ToolTurn("INVALID", None, raw))
                # One repair turn with an explicit protocol reminder.
                transcript.append("FORMAT ERROR: output only CALC: <expression> or FINAL: <number>.")
                continue

            try:
                val = self.math.evaluate(expr)
                result = self._fmt(val)
                self.tool_calls += 1
                trace.append(f"step{step+1}:calc={expr}=>{result}")
                turns.append(ToolTurn(expr, result, raw))
                transcript.append("CALC: " + expr)
                transcript.append("RESULT: " + result)
                self.core.encode_event(SemanticEvent(
                    relation="calculator_result", quantity=float(val), goal="gsm8k_tool_loop",
                    confidence=1.0, payload={"expression": expr, "result": result, "step": step + 1},
                ))
            except Exception as e:
                trace.append(f"step{step+1}:calc_error={type(e).__name__}")
                turns.append(ToolTurn(expr, None, raw))
                transcript.append("CALC ERROR: expression rejected. Use only valid numeric arithmetic.")

        if final is None:
            # Finalization gets one short turn but no free-form arithmetic permission.
            raw = self.cortex.call(
                "Return the final answer from the calculator results already shown. Output exactly FINAL: <number>. Do not calculate anything new.",
                "\n\n".join(transcript), 32,
            )
            final = self._final(raw) or _extract_number(raw)
            trace.append(f"forced_final={final}")

        causal_ops = self.math.ops - ops0
        confidence = 0.94 if final is not None and causal_ops > 0 else 0.58
        self.core.encode_event(SemanticEvent(
            relation="final_math_decision", goal="gsm8k", confidence=confidence,
            payload={"final": final, "causal_ops": causal_ops, "tool_calls": self.tool_calls},
        ))
        return V08Result(
            final, confidence, min(self.max_tool_steps + 1, self.cortex.calls - calls0),
            self.cortex.calls - calls0, trace,
        )

    def snapshot(self):
        s = self.core.snapshot()
        s.update({
            "version": "0.9R1",
            "cortex_calls": self.cortex.calls,
            "calculator_tool_calls": self.tool_calls,
            "causal_math_ops": self.math.ops,
        })
        return s
