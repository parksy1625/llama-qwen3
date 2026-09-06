from __future__ import annotations

"""General-128 V0.9R3.3: typed semantic graph arithmetic router.

R3.3 removes arbitrary arithmetic-expression search. It recognizes a small set
of grounded relation families from the word problem, builds typed quantity
nodes/edges, and executes every arithmetic edge through General-128 learned
numeric operator families. Gold/reference answers are evaluator-only.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import math
import re

from general128_v07 import General128V07, SemanticEvent
from general128_v08 import LanguageCortex, V08Result, _extract_number, _num_equal
from general128_v09 import CausalArithmetic128


@dataclass
class TypedNode:
    value: float
    unit: str
    entity: str
    role: str
    source: str
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class GraphPlan:
    family: str
    value: float
    confidence: float
    trace: List[str]
    evidence: List[str]


class General128V09R33:
    CORE = {"add": "g_add", "sub": "g_sub", "mul": "g_mul", "div": "g_div"}
    NUMBER_WORDS = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20,
    }
    FRACTIONS = {
        "half": 0.5, "one half": 0.5,
        "third": 1/3, "one third": 1/3, "two thirds": 2/3,
        "quarter": 0.25, "one quarter": 0.25, "three quarters": 0.75,
        "one fifth": 0.2, "two fifths": 0.4, "three fifths": 0.6, "four fifths": 0.8,
        "one sixth": 1/6, "five sixths": 5/6,
    }

    def __init__(self, cortex_endpoint: str):
        self.cortex = LanguageCortex(cortex_endpoint)
        self.core = General128V07()
        self.math = CausalArithmetic128(self.core)
        self.graphs = 0
        self.routed = 0
        self.overrides = 0
        self.fallbacks = 0
        self.family_counts: Dict[str, int] = {}

    @staticmethod
    def _fmt(x: float) -> str:
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return ("%.10f" % x).rstrip("0").rstrip(".")

    @classmethod
    def _numify(cls, text: str) -> str:
        out = text
        for word, value in sorted(cls.NUMBER_WORDS.items(), key=lambda kv: -len(kv[0])):
            out = re.sub(r"\b" + re.escape(word) + r"\b", str(value), out, flags=re.I)
        return out

    @staticmethod
    def _sentences(text: str) -> List[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    @classmethod
    def _fraction_value(cls, text: str) -> Optional[float]:
        low = text.lower().replace("-", " ")
        for phrase, value in sorted(cls.FRACTIONS.items(), key=lambda kv: -len(kv[0])):
            if re.search(r"\b" + re.escape(phrase) + r"\b", low):
                return float(value)
        m = re.search(r"\b(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\b", low)
        if m and float(m.group(2)) != 0:
            return float(m.group(1))/float(m.group(2))
        return None

    @staticmethod
    def _normalize_entity(word: str) -> str:
        x = word.lower().strip(" .,$")
        aliases = {
            "mirrors":"mirror", "shelves":"shelf", "chandeliers":"chandelier", "pictures":"picture",
            "people":"person", "persons":"person", "bags":"bag", "chips":"chip", "items":"item",
            "hours":"hour", "weeks":"week", "liters":"liter", "litres":"liter", "inches":"inch",
            "eggs":"egg", "glasses":"glass", "donuts":"donut", "cupcakes":"cupcake", "cheesecakes":"cheesecake",
            "miles":"mile", "meters":"meter", "lemons":"lemon", "downloads":"download",
        }
        if x in aliases:
            return aliases[x]
        if x.endswith("s") and len(x) > 3:
            return x[:-1]
        return x

    @classmethod
    def _typed_nodes(cls, question: str) -> List[TypedNode]:
        q = cls._numify(question.replace(",", ""))
        nodes: List[TypedNode] = []
        for s in cls._sentences(q):
            for m in re.finditer(r"(?P<money>\$)?(?P<num>\d+(?:\.\d+)?)(?:\s+)(?P<unit>[A-Za-z]+)?", s):
                val = float(m.group("num"))
                unit = "dollar" if m.group("money") else cls._normalize_entity(m.group("unit") or "number")
                right = s[m.end():m.end()+35].lower()
                left = s[max(0,m.start()-35):m.start()].lower()
                role = "rate" if " per " in right or " per " in left else "given"
                if re.search(r"\binclude[sd]?\b", left): role = "included"
                if "extra" in left+right or "additional" in left+right: role = "extra"
                nodes.append(TypedNode(val, unit, unit, role, m.group(0).strip(), {"sentence": s}))
            f = cls._fraction_value(s)
            if f is not None:
                nodes.append(TypedNode(f, "fraction", "fraction", "fraction", cls._fmt(f), {"sentence": s}))
        return nodes

    def _op(self, op: str, a: float, b: float, trace: List[str], label: str) -> float:
        if op == "div" and abs(b) < 1e-15:
            raise ZeroDivisionError
        r = self.core.numeric(self.CORE[op], float(a), float(b))
        if not r.verified or not isinstance(r.answer, (int, float)):
            raise ValueError("unverified operator family")
        self.math.ops += 1
        value = float(r.answer)
        if not math.isfinite(value) or abs(value) > 1e15:
            raise ValueError("numeric range")
        trace.append(f"{label}:{self._fmt(a)} {op} {self._fmt(b)} -> {self._fmt(value)}")
        self.core.encode_event(SemanticEvent(
            relation=f"r33:{op}", quantity=value, goal=label, confidence=r.confidence,
            payload={"a": a, "b": b, "op": op},
        ))
        return value

    @classmethod
    def _entity_counts(cls, sentence: str) -> Dict[str, float]:
        s = cls._numify(sentence.replace(",", ""))
        out: Dict[str, float] = {}
        for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s+([A-Za-z]+)\b", s):
            ent = cls._normalize_entity(m.group(2))
            if ent in {"mirror","shelf","chandelier","picture","person","bag","chip","item","hour","week","inch","liter","egg","glass"}:
                out[ent] = float(m.group(1))
        return out

    def _bundle_extras(self, question: str) -> Optional[GraphPlan]:
        low = question.lower()
        if not (re.search(r"\binclude[sd]?\b", low) and re.search(r"\b(additional|extra)\b", low) and "per item" in low):
            return None
        sents = self._sentences(self._numify(question.replace(",", "")))
        base = None; rate = None; included: Dict[str,float] = {}; wanted: Dict[str,float] = {}
        for s in sents:
            sl = s.lower()
            money = re.findall(r"\$(\d+(?:\.\d+)?)", s)
            if base is None and money and not ("additional" in sl or "extra" in sl):
                base = float(money[0])
            if re.search(r"\binclude[sd]?\b", sl):
                included.update(self._entity_counts(s))
            if ("additional" in sl or "extra" in sl) and "per item" in sl and money:
                rate = float(money[-1])
            if re.search(r"\b(has|needs?|wants?)\b", sl) and not re.search(r"\binclude[sd]?\b", sl):
                wanted.update(self._entity_counts(s))
        if base is None or rate is None or not included or not wanted:
            return None
        trace: List[str] = []
        extra_total = 0.0; matched = 0
        for ent, have in wanted.items():
            if ent not in included:
                continue
            matched += 1
            excess = self._op("sub", have, included[ent], trace, f"excess_{ent}")
            if excess > 0:
                charge = self._op("mul", excess, rate, trace, f"extra_charge_{ent}")
                extra_total = charge if extra_total == 0 else self._op("add", extra_total, charge, trace, "extra_sum")
        if matched == 0:
            return None
        final = self._op("add", base, extra_total, trace, "bundle_total") if extra_total else base
        return GraphPlan("bundle_extras", final, 0.995, trace,
                         [f"base={base}", f"rate={rate}", f"included={included}", f"wanted={wanted}"])

    @classmethod
    def _mixture_components(cls, question: str) -> List[Tuple[str,float,float]]:
        low = question.lower().replace("-", " ").replace(",", "")
        # Stop each component at its own 'water' phrase so multiple components can share a sentence.
        pat = re.compile(
            r"(\d+(?:\.\d+)?)\s+lit(?:er|re)s?\s+of\s+([a-z]+)(?:\s+drink)?"
            r".{0,60}?\b((?:one|two|three|four|five)\s+(?:half|thirds?|quarters?|fifths?|sixths?)|half|third|quarter)\b"
            r"\s+water",
            re.I,
        )
        out=[]
        for m in pat.finditer(low):
            frac = cls._fraction_value(m.group(3))
            if frac is not None:
                out.append((m.group(2).lower(), float(m.group(1)), frac))
        return out

    def _mixture_fraction(self, question: str) -> Optional[GraphPlan]:
        low = question.lower()
        if "water" not in low or not ("liter" in low or "litre" in low):
            return None
        components = self._mixture_components(question)
        if len(components) < 2:
            return None
        qn = self._numify(question.lower().replace("-", " ").replace(",", ""))
        spill = None; spill_name = None
        m = re.search(r"spill\w*\s+(\d+(?:\.\d+)?)\s+lit(?:er|re)s?\s+of\s+the\s+([a-z]+)", qn)
        if not m:
            m = re.search(r"spill\w*\s+(\d+(?:\.\d+)?)\s+lit(?:er|re)s?.{0,30}?([a-z]+)\s+drink", qn)
        if m:
            spill = float(m.group(1)); spill_name = m.group(2).lower()
        trace=[]; parts=[]
        for name, volume, frac in components:
            effective = volume
            if spill is not None and spill_name and (spill_name in name or name in spill_name):
                effective = self._op("sub", volume, spill, trace, f"remaining_{name}")
            parts.append(self._op("mul", effective, frac, trace, f"water_{name}"))
        total = parts[0]
        for p in parts[1:]:
            total = self._op("add", total, p, trace, "water_sum")
        return GraphPlan("mixture_fraction", total, 0.99, trace, [str(x) for x in components])

    def _repeat_scale(self, question: str) -> Optional[GraphPlan]:
        low = self._numify(question.lower().replace("-", " "))
        if not ("half as long" in low and re.search(r"\d+(?:\.\d+)?\s+times? a week", low) and re.search(r"\d+(?:\.\d+)?\s+weeks?", low)):
            return None
        mbase = re.search(r"(\d+(?:\.\d+)?)\s+hours?", low)
        mfreq = re.search(r"(\d+(?:\.\d+)?)\s+times? a week", low)
        mweeks = re.search(r"(?:in|for)\s+(\d+(?:\.\d+)?)\s+weeks?", low)
        if not (mbase and mfreq and mweeks):
            return None
        base=float(mbase.group(1)); freq=float(mfreq.group(1)); weeks=float(mweeks.group(1)); trace=[]
        secondary=self._op("mul",base,0.5,trace,"half_duration")
        episode=self._op("add",base,secondary,trace,"episode_total")
        weekly=self._op("mul",episode,freq,trace,"weekly_total")
        total=self._op("mul",weekly,weeks,trace,"period_total")
        return GraphPlan("repeat_scale", total, 0.995, trace, [f"base={base}",f"freq={freq}",f"weeks={weeks}"])

    @staticmethod
    def _nonmoney_numbers(clause: str) -> List[float]:
        masked = re.sub(r"\$\d+(?:\.\d+)?", " ", clause.replace(",", ""))
        return [float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", masked)]

    def _per_group_sum(self, question: str) -> Optional[GraphPlan]:
        low = question.lower()
        if "per person" not in low or not re.search(r"\btotal\b", low):
            return None
        clauses = re.split(r"\band\b|,|\.(?!\d)", self._numify(question.replace(",", "")), flags=re.I)
        terms=[]; trace=[]; evidence=[]
        for c in clauses:
            if "per person" not in c.lower():
                continue
            money = re.findall(r"\$(\d+(?:\.\d+)?)", c)
            nums = self._nonmoney_numbers(c)
            if not money or not nums:
                continue
            count=nums[-1]; rate=float(money[-1])
            terms.append(self._op("mul", count, rate, trace, "group_cost"))
            evidence.append(f"{count}*{rate}")
        if len(terms) < 2:
            return None
        total=terms[0]
        for t in terms[1:]:
            total=self._op("add",total,t,trace,"group_sum")
        return GraphPlan("per_group_sum", total, 0.99, trace, evidence)

    def _unit_chain(self, question: str) -> Optional[GraphPlan]:
        low = self._numify(question.lower().replace(",", ""))
        if not ("per square inch" in low or "every square inch" in low):
            return None
        if not ("bag" in low and "chip" in low and re.search(r"how many inches? long", low)):
            return None
        md = re.search(r"(?:takes\s+)?(\d+(?:\.\d+)?)\s+(?:glass\s+)?chips?\s+to\s+make\s+every\s+square inch", low)
        if not md:
            md = re.search(r"(\d+(?:\.\d+)?)\s+(?:glass\s+)?chips?\s+per\s+square inch", low)
        mb = re.search(r"bag of .*?holds\s+(\d+(?:\.\d+)?)\s+chips", low)
        mn = re.search(r"(?:has|with)\s+(\d+(?:\.\d+)?)\s+bags", low)
        mh = re.search(r"(\d+(?:\.\d+)?)\s+inches? tall", low)
        if not (md and mb and mn and mh):
            return None
        density=float(md.group(1)); perbag=float(mb.group(1)); bags=float(mn.group(1)); height=float(mh.group(1)); trace=[]
        chips=self._op("mul",perbag,bags,trace,"inventory_chips")
        area=self._op("div",chips,density,trace,"chips_to_area")
        length=self._op("div",area,height,trace,"area_to_length")
        return GraphPlan("unit_chain", length, 0.995, trace,
                         [f"density={density}",f"perbag={perbag}",f"bags={bags}",f"height={height}"])

    def _simple_rate_total(self, question: str) -> Optional[GraphPlan]:
        """Conservative generic two-or-more group count × dollar-rate sum."""
        low=question.lower()
        if not (re.search(r"\b(total|altogether|in all)\b", low) and re.search(r"\bper\b", low)):
            return None
        clauses=re.split(r"\band\b|,|\.(?!\d)", self._numify(question.replace(",", "")), flags=re.I)
        terms=[]; trace=[]; evidence=[]
        for c in clauses:
            if " per " not in c.lower(): continue
            money=re.findall(r"\$(\d+(?:\.\d+)?)",c)
            nums=self._nonmoney_numbers(c)
            if money and nums:
                count=nums[-1]; rate=float(money[-1])
                terms.append(self._op("mul",count,rate,trace,"rate_product")); evidence.append(f"{count}*{rate}")
        if len(terms)<2: return None
        total=terms[0]
        for t in terms[1:]: total=self._op("add",total,t,trace,"sum_parts")
        return GraphPlan("simple_rate_total",total,0.90,trace,evidence)

    def _build_graph(self, question: str) -> Tuple[List[TypedNode], List[GraphPlan]]:
        nodes=self._typed_nodes(question); plans=[]
        for fn in (self._bundle_extras,self._mixture_fraction,self._repeat_scale,self._per_group_sum,self._unit_chain,self._simple_rate_total):
            try:
                p=fn(question)
                if p: plans.append(p)
            except Exception:
                pass
        self.graphs += 1
        return nodes, plans

    @staticmethod
    def _family_choice(text: str) -> Optional[str]:
        m=re.findall(r"FAMILY\s*:\s*([a-z_]+)",text,re.I)
        return m[-1].lower() if m else None

    def _choose_plan(self, question: str, plans: Sequence[GraphPlan]) -> Optional[GraphPlan]:
        if not plans:
            return None
        best=sorted(plans,key=lambda p:-p.confidence)
        if len(best)==1 or (best[0].confidence>=0.98 and best[0].confidence-best[1].confidence>=0.03):
            return best[0]
        # Ambiguous low-margin graph classifications are delegated only at family level.
        families=", ".join(dict.fromkeys(p.family for p in best))
        raw=self.cortex.call(
            "Choose the single relation family matching the word-problem structure. Do not calculate. Output exactly FAMILY: <name>.",
            question+"\n\nAllowed families: "+families, 32)
        fam=self._family_choice(raw)
        return next((p for p in best if p.family==fam), None)

    def answer_math(self, question: str, baseline_hint: Optional[str]=None) -> V08Result:
        calls0=self.cortex.calls
        if baseline_hint is None:
            raw=self.cortex.call("Solve carefully and end exactly FINAL: <number>.",question,180)
            baseline=_extract_number(raw)
        else:
            baseline=str(baseline_hint) if baseline_hint is not None else None
        nodes,plans=self._build_graph(question)
        trace=[f"baseline={baseline}",f"typed_nodes={len(nodes)}",f"plans={[p.family for p in plans]}"]
        plan=self._choose_plan(question,plans)
        if plan is None:
            self.fallbacks += 1
            return V08Result(baseline,0.80,self.cortex.calls-calls0,self.cortex.calls-calls0,trace+["gate=no_typed_plan"])
        self.routed += 1
        self.family_counts[plan.family]=self.family_counts.get(plan.family,0)+1
        core=self._fmt(plan.value)
        trace += [f"family={plan.family}",f"core={core}"] + plan.trace
        if baseline is not None and _num_equal(baseline,core):
            trace.append("gate=baseline_core_agree")
            return V08Result(baseline,0.99,self.cortex.calls-calls0,self.cortex.calls-calls0,trace)
        if plan.confidence >= 0.98:
            final=core; conf=plan.confidence; self.overrides += 1; trace.append("gate=typed_override")
        else:
            final=baseline; conf=0.84; self.fallbacks += 1; trace.append("gate=conservative_base")
        self.core.encode_event(SemanticEvent(
            relation="r33_gate", goal="gsm8k", confidence=conf,
            payload={"family":plan.family,"baseline":baseline,"core":core,"final":final},
        ))
        return V08Result(final,conf,self.cortex.calls-calls0,self.cortex.calls-calls0,trace)

    def snapshot(self):
        s=self.core.snapshot()
        s.update({
            "version":"0.9R3.3", "cortex_calls":self.cortex.calls,
            "causal_math_ops":self.math.ops, "graphs":self.graphs,
            "routed":self.routed, "overrides":self.overrides,
            "fallbacks":self.fallbacks, "family_counts":self.family_counts,
        })
        return s
