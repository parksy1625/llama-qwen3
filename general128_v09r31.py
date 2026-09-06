from __future__ import annotations

"""V0.9R3.1: R3 validator with a concrete few-shot plan interface for 0.5B cortex."""
from typing import Optional, Tuple
from general128_v09r3 import General128V09R3


class General128V09R31(General128V09R3):
    def _ask_plan(self, question: str, variant: int) -> Tuple[Optional[dict], str]:
        examples = '''
Example 1 problem: A box has 3 rows of 4 apples and then 2 extra apples. How many apples?
Example 1 output:
PLAN: {"steps":[{"op":"mul","a":3,"b":4,"out":"r0"},{"op":"add","a":"r0","b":2,"out":"r1"}],"answer":"r1"}

Example 2 problem: Mina has 20 candies and gives away 7. How many remain?
Example 2 output:
PLAN: {"steps":[{"op":"sub","a":20,"b":7,"out":"r0"}],"answer":"r0"}
'''
        if variant == 1:
            sys = (
                "Convert the problem to arithmetic JSON, not a prose solution. "
                "Copy the exact PLAN format shown in the examples. Allowed ops: add, sub, mul, div. "
                "Operands must be either numeric literals copied from the problem or quoted prior result names like \"r0\". "
                "Use r0, r1, ... in order. Maximum 6 steps. Do not invent numbers.\n" + examples
            )
        else:
            sys = (
                "Independently make the smallest arithmetic dependency graph. Return exactly one PLAN JSON line. "
                "Allowed ops: add, sub, mul, div. Every literal number must occur in the problem; intermediate values must be \"rN\" references. "
                "Use r0, r1, ... sequentially and maximum 6 steps.\n" + examples
            )
        raw = self.cortex.call(sys, question + "\nOutput only PLAN: {...}", 220)
        return self._extract_plan(raw), raw

    def snapshot(self):
        s = super().snapshot(); s["version"] = "0.9R3.1"; return s
