from __future__ import annotations

"""General-128 V0.9R2: conservative selective augmentation.

R1 proved that the 128-core can sit causally in the arithmetic path, but a weak
0.5B planner can generate a wrong calculation program. R2 therefore preserves
Qwen's direct answer by default. A core-generated candidate may replace it only
when two independent audits both reproduce/support the core candidate.

No benchmark labels are used by the gate.
"""

from typing import Optional

from general128_v08 import V08Result, _extract_number, _num_equal
from general128_v09r1 import General128V09R1
from general128_v07 import SemanticEvent


class General128V09R2(General128V09R1):
    def answer_math(self, question: str, baseline_hint: Optional[str] = None) -> V08Result:
        calls0 = self.cortex.calls

        if baseline_hint is None:
            direct = self.cortex.call(
                "Solve the grade-school math problem carefully. End exactly FINAL: <number>.",
                question, 180,
            )
            baseline = _extract_number(direct)
        else:
            baseline = str(baseline_hint) if baseline_hint is not None else None

        ops0 = self.math.ops
        tool_result = super().answer_math(question)
        tool = str(tool_result.answer) if tool_result.answer is not None else None
        causal_ops = self.math.ops - ops0
        trace = [f"baseline={baseline}"] + list(tool_result.trace)

        # Safe cases first: tool failed or agrees with the direct model.
        if tool is None or causal_ops <= 0:
            final = baseline
            conf = 0.78 if baseline is not None else 0.30
            gate = "baseline_no_causal_tool"
        elif baseline is not None and _num_equal(baseline, tool):
            final = baseline
            conf = 0.97
            gate = "baseline_tool_agree"
        else:
            # Two differently worded audits. Both must independently land on the
            # core candidate before R2 is allowed to override the direct answer.
            candidate_block = f"Direct candidate: {baseline}\nCalculator candidate: {tool}"
            audit1_raw = self.cortex.call(
                "Independently solve the original word problem. Candidate answers may be wrong. End exactly FINAL: <number>.",
                question + "\n\n" + candidate_block,
                180,
            )
            audit2_raw = self.cortex.call(
                "Check quantities, units, rates, and what the question asks. Re-solve independently and end exactly FINAL: <number>.",
                question + "\n\n" + candidate_block,
                180,
            )
            a1 = _extract_number(audit1_raw)
            a2 = _extract_number(audit2_raw)
            trace += [f"audit1={a1}", f"audit2={a2}"]
            if _num_equal(a1, tool) and _num_equal(a2, tool):
                final = tool
                conf = 0.94
                gate = "tool_double_audit"
            else:
                final = baseline if baseline is not None else tool
                conf = 0.82 if baseline is not None else 0.62
                gate = "baseline_conservative"

        trace.append("gate=" + gate)
        self.core.encode_event(SemanticEvent(
            relation="r2_gate", goal="gsm8k", confidence=conf,
            payload={"baseline": baseline, "tool": tool, "causal_ops": causal_ops,
                     "final": final, "gate": gate},
        ))
        return V08Result(
            final, conf,
            min(12, self.cortex.calls - calls0),
            self.cortex.calls - calls0,
            trace,
        )

    def snapshot(self):
        s = super().snapshot()
        s["version"] = "0.9R2"
        return s
