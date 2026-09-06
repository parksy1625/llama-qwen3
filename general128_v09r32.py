from __future__ import annotations

"""General-128 V0.9R3.2: symbolic candidate search + semantic verifier.

The 0.5B cortex no longer has to emit a complete arithmetic program.  The
system enumerates short arithmetic programs from quantities grounded in the
question, asks the cortex only to select a candidate ID, and executes selected
programs through General-128 learned operator families.  Two independently
prompted selections must agree numerically; a final conservative audit is
required before a result may replace the direct Qwen answer.

No benchmark labels or reference answers are used by the search, verifier, or
routing gate.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math
import re

from general128_v07 import General128V07, SemanticEvent
from general128_v08 import LanguageCortex, V08Result, _extract_number, _num_equal
from general128_v09 import CausalArithmetic128


@dataclass(frozen=True)
class Atom:
    key: str
    text: str
    value: float
    source: str


@dataclass(frozen=True)
class Expr:
    op: Optional[str]
    left: object
    right: object
    text: str
    used: Tuple[str, ...]
    depth: int


class General128V09R32:
    OPS = ("add", "sub", "mul", "div")
    SYMBOL = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
    CORE_SYMBOL = {"add": "g_add", "sub": "g_sub", "mul": "g_mul", "div": "g_div"}

    def __init__(self, cortex_endpoint: str, shortlist: int = 56):
        self.cortex = LanguageCortex(cortex_endpoint)
        self.core = General128V07()
        self.math = CausalArithmetic128(self.core)
        self.shortlist = shortlist
        self.searches = 0
        self.consensus = 0
        self.overrides = 0
        self.fallbacks = 0

    @staticmethod
    def _fmt(x: float) -> str:
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return ("%.10f" % x).rstrip("0").rstrip(".")

    @staticmethod
    def _atoms(question: str) -> List[Atom]:
        q = question.replace(",", "")
        atoms: List[Atom] = []
        for i, m in enumerate(re.finditer(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", q)):
            try:
                v = float(m.group(0))
            except Exception:
                continue
            atoms.append(Atom(f"n{i}", m.group(0), v, "text"))
        low = q.lower()
        semantic = [
            (r"\b(twice|double)\b", 2.0, "twice"),
            (r"\btriple\b", 3.0, "triple"),
            (r"\bhalf\b", 0.5, "half"),
            (r"\bquarter\b", 0.25, "quarter"),
            (r"\bdozen\b", 12.0, "dozen"),
            (r"\bpercent\b|%", 100.0, "percent-base"),
        ]
        j = 0
        for pat, val, name in semantic:
            if re.search(pat, low):
                # Avoid duplicating an identical explicit number solely for search size.
                if not any(abs(a.value - val) < 1e-12 and a.source == "text" for a in atoms):
                    atoms.append(Atom(f"s{j}", name, val, "semantic")); j += 1
        # Long GSM problems can contain many incidental quantities; cap while preserving order.
        return atoms[:10]

    @classmethod
    def _leaf(cls, atom: Atom) -> Expr:
        return Expr(None, atom, None, atom.text, (atom.key,), 0)

    @classmethod
    def _join(cls, op: str, a: Expr, b: Expr) -> Expr:
        sym = cls.SYMBOL[op]
        return Expr(op, a, b, f"({a.text} {sym} {b.text})", tuple(sorted(set(a.used + b.used))), max(a.depth, b.depth) + 1)

    @classmethod
    def _enumerate(cls, atoms: Sequence[Atom]) -> List[Expr]:
        leaves = [cls._leaf(a) for a in atoms]
        out: List[Expr] = []
        seen = set()

        def add(e: Expr):
            if e.text not in seen:
                seen.add(e.text); out.append(e)

        # One-operation expressions.
        for i, a in enumerate(leaves):
            for j, b in enumerate(leaves):
                if i == j:
                    continue
                if i < j:
                    add(cls._join("add", a, b)); add(cls._join("mul", a, b))
                add(cls._join("sub", a, b))
                if abs(atoms[j].value) > 1e-15:
                    add(cls._join("div", a, b))

        level1 = list(out)
        # Two-operation expressions, using a fresh grounded quantity at the second step.
        for e in level1:
            for leaf in leaves:
                if leaf.used[0] in e.used:
                    continue
                add(cls._join("add", e, leaf)); add(cls._join("mul", e, leaf))
                add(cls._join("sub", e, leaf)); add(cls._join("sub", leaf, e))
                add(cls._join("div", e, leaf)); add(cls._join("div", leaf, e))
        return out

    @staticmethod
    def _cues(question: str) -> Dict[str, float]:
        q = question.lower()
        s = {"add": 0.0, "sub": 0.0, "mul": 0.0, "div": 0.0}
        for pat, op, w in [
            (r"\b(total|altogether|together|combined|in all|sum)\b", "add", 2.0),
            (r"\b(left|remain|remaining|gave away|spent|lost|fewer|difference)\b", "sub", 2.2),
            (r"\b(each|every|per day|per hour|times|rows of|groups of|twice|double)\b", "mul", 2.2),
            (r"\b(split|equally|each person|average|per person|divided)\b", "div", 2.2),
            (r"\bmore\b", "add", 0.8),
            (r"\bless\b", "sub", 0.8),
        ]:
            if re.search(pat, q): s[op] += w
        return s

    @classmethod
    def _ops_in(cls, e: Expr) -> List[str]:
        if e.op is None: return []
        return cls._ops_in(e.left) + cls._ops_in(e.right) + [e.op]

    @classmethod
    def _rank(cls, e: Expr, question: str, natoms: int) -> float:
        cues = cls._cues(question); ops = cls._ops_in(e)
        score = sum(cues[o] for o in ops)
        # Prefer compact plans and use of multiple grounded quantities, but retain diversity.
        score += 0.65 * len(set(e.used)) - 0.55 * max(0, len(ops) - 1)
        if len(set(e.used)) == min(3, natoms): score += 0.25
        if e.op and cues[e.op] > 0: score += 0.35
        return score

    def _shortlist(self, question: str) -> List[Expr]:
        atoms = self._atoms(question)
        all_expr = self._enumerate(atoms)
        ranked = sorted(all_expr, key=lambda e: (-self._rank(e, question, len(atoms)), len(self._ops_in(e)), e.text))
        # Round-robin across final operators so lexical heuristics cannot collapse the menu.
        buckets: Dict[str, List[Expr]] = {o: [] for o in self.OPS}
        for e in ranked:
            if e.op in buckets: buckets[e.op].append(e)
        chosen: List[Expr] = []
        idx = {o: 0 for o in self.OPS}
        while len(chosen) < self.shortlist:
            progressed = False
            for op in sorted(self.OPS, key=lambda o: -self._cues(question)[o]):
                if idx[op] < len(buckets[op]):
                    chosen.append(buckets[op][idx[op]]); idx[op] += 1; progressed = True
                    if len(chosen) >= self.shortlist: break
            if not progressed: break
        self.searches += 1
        return chosen

    def _menu(self, question: str, xs: Sequence[Expr]) -> str:
        lines = [f"C{i}: {e.text}" for i, e in enumerate(xs)]
        return question + "\n\nCandidate calculations (do not calculate them; choose the calculation matching the story):\n" + "\n".join(lines)

    @staticmethod
    def _pick(text: str, n: int) -> Optional[int]:
        hits = re.findall(r"(?:PICK|ANSWER|CANDIDATE)\s*[:=]?\s*C?(\d+)", text, re.I)
        if not hits:
            hits = re.findall(r"\bC(\d+)\b", text, re.I)
        if not hits: return None
        try: j = int(hits[-1])
        except: return None
        return j if 0 <= j < n else None

    def _select(self, question: str, xs: Sequence[Expr], variant: int) -> Optional[int]:
        menu = self._menu(question, xs)
        if variant == 1:
            sys = "Match the word problem to exactly one candidate calculation. Do not solve the arithmetic. Check which quantities and operations the story requires. Output only PICK: <candidate number>."
        else:
            sys = "Independently inspect the story structure, units, and requested quantity, then select the candidate expression that represents it. Do not compute values. Output only PICK: <candidate number>."
        raw = self.cortex.call(sys, menu, 24)
        return self._pick(raw, len(xs))

    def _eval(self, e: Expr) -> float:
        if e.op is None:
            return float(e.left.value)
        a = self._eval(e.left); b = self._eval(e.right)
        if e.op == "div" and abs(b) < 1e-15: raise ZeroDivisionError
        r = self.core.numeric(self.CORE_SYMBOL[e.op], a, b)
        if not r.verified or not isinstance(r.answer, (int, float)):
            raise ValueError("unverified core result")
        self.math.ops += 1
        val = float(r.answer)
        if not math.isfinite(val) or abs(val) > 1e15: raise ValueError("range")
        self.core.encode_event(SemanticEvent(
            relation=f"r32:{e.op}", quantity=val, goal="candidate_execution",
            confidence=r.confidence, payload={"expression": e.text, "a": a, "b": b},
        ))
        return val

    @staticmethod
    def _audit_choice(text: str) -> Optional[str]:
        m = re.findall(r"USE\s*:\s*(BASE|CORE)", text, re.I)
        return m[-1].upper() if m else None

    def answer_math(self, question: str, baseline_hint: Optional[str] = None) -> V08Result:
        calls0 = self.cortex.calls; ops0 = self.math.ops
        if baseline_hint is None:
            raw = self.cortex.call("Solve carefully and end exactly FINAL: <number>.", question, 180)
            baseline = _extract_number(raw)
        else:
            baseline = str(baseline_hint) if baseline_hint is not None else None

        xs = self._shortlist(question)
        trace = [f"baseline={baseline}", f"candidates={len(xs)}"]
        if not xs:
            self.fallbacks += 1
            return V08Result(baseline, 0.75, self.cortex.calls-calls0, self.cortex.calls-calls0, trace+["gate=no_candidates"])

        p1 = self._select(question, xs, 1); p2 = self._select(question, xs, 2)
        trace += [f"pick1={p1}", f"pick2={p2}"]
        if p1 is None or p2 is None:
            self.fallbacks += 1
            return V08Result(baseline, 0.78, self.cortex.calls-calls0, self.cortex.calls-calls0, trace+["gate=pick_failure"])

        try:
            v1 = self._fmt(self._eval(xs[p1])); v2 = self._fmt(self._eval(xs[p2]))
        except Exception as e:
            self.fallbacks += 1
            return V08Result(baseline, 0.78, self.cortex.calls-calls0, self.cortex.calls-calls0, trace+[f"gate=core_error:{type(e).__name__}"])
        trace += [f"core1={v1}:{xs[p1].text}", f"core2={v2}:{xs[p2].text}"]

        if not _num_equal(v1, v2):
            self.fallbacks += 1
            return V08Result(baseline, 0.80, self.cortex.calls-calls0, self.cortex.calls-calls0, trace+["gate=no_numeric_consensus"])

        self.consensus += 1; core = v1
        if baseline is not None and _num_equal(baseline, core):
            return V08Result(baseline, 0.97, self.cortex.calls-calls0, self.cortex.calls-calls0, trace+["gate=baseline_core_agree"])

        audit_raw = self.cortex.call(
            "Compare two proposed numerical answers to the original word problem. Re-check the story and units. Output only USE: BASE or USE: CORE. Prefer BASE unless CORE is clearly supported by the problem structure.",
            question + f"\n\nBASE answer: {baseline}\nCORE answer: {core}\nCORE calculation A: {xs[p1].text}\nCORE calculation B: {xs[p2].text}",
            32,
        )
        audit = self._audit_choice(audit_raw); trace.append(f"audit={audit}")
        if audit == "CORE":
            final = core; self.overrides += 1; conf = 0.93; gate = "verified_override"
        else:
            final = baseline; self.fallbacks += 1; conf = 0.83; gate = "conservative_base"
        trace.append("gate=" + gate)
        self.core.encode_event(SemanticEvent(
            relation="r32_gate", goal="gsm8k", confidence=conf,
            payload={"baseline": baseline, "core": core, "audit": audit, "final": final},
        ))
        return V08Result(final, conf, self.cortex.calls-calls0, self.cortex.calls-calls0, trace)

    def snapshot(self):
        s = self.core.snapshot(); s.update({
            "version": "0.9R3.2", "cortex_calls": self.cortex.calls,
            "causal_math_ops": self.math.ops, "searches": self.searches,
            "numeric_consensus": self.consensus, "overrides": self.overrides,
            "fallbacks": self.fallbacks,
        }); return s
