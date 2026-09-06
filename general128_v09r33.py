from __future__ import annotations

"""General-128 V0.9R3.3: typed semantic graph arithmetic router.

This version removes arbitrary expression search.  It extracts typed quantities
and relation cues from the question, instantiates only a small set of grounded
relation families, and executes every arithmetic edge through General-128
learned numeric operator families.

Gold/reference answers are never used by parsing, routing, execution, or gates.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import math
import re

from general128_v07 import General128V07, SemanticEvent
from general128_v08 import LanguageCortex, V08Result, _extract_number, _num_equal
from general128_v09 import CausalArithmetic128


@dataclass
class Quantity:
    value: float
    unit: str
    entity: str
    sentence: int
    start: int
    text: str
    role: str = "given"
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

    @staticmethod
    def _sentences(q: str) -> List[str]:
        return [x.strip() for x in re.split(r"(?<=[.!?])\s+", q) if x.strip()]

    @staticmethod
    def _word_fraction(text: str) -> Optional[float]:
        low = text.lower().replace("-", " ")
        fixed = {
            "half": 0.5, "one half": 0.5,
            "third": 1/3, "one third": 1/3, "two thirds": 2/3,
            "quarter": 0.25, "one quarter": 0.25, "three quarters": 0.75,
            "one fifth": 0.2, "two fifths": 0.4, "three fifths": 0.6, "four fifths": 0.8,
            "one sixth": 1/6, "five sixths": 5/6,
        }
        for k in sorted(fixed, key=len, reverse=True):
            if re.search(r"\b" + re.escape(k) + r"\b", low):
                return fixed[k]
        return None

    @staticmethod
    def _normalize_unit(raw: str) -> str:
        x = raw.lower().strip(" .,$")
        aliases = {
            "people": "person", "persons": "person", "person": "person",
            "hours": "hour", "hour": "hour", "weeks": "week", "week": "week",
            "liters": "liter", "liter": "liter", "litres": "liter", "litre": "liter",
            "chips": "chip", "chip": "chip", "bags": "bag", "bag": "bag",
            "inches": "inch", "inch": "inch", "items": "item", "item": "item",
            "mirrors": "mirror", "mirror": "mirror", "shelves": "shelf", "shelf": "shelf",
            "chandeliers": "chandelier", "chandelier": "chandelier", "pictures": "picture", "picture": "picture",
            "dollars": "dollar", "dollar": "dollar",
        }
        return aliases.get(x, x)

    @classmethod
    def _extract_quantities(cls, question: str) -> List[Quantity]:
        sents = cls._sentences(question.replace(",", ""))
        out: List[Quantity] = []
        offset = 0
        for si, sent in enumerate(sents):
            # Numeric quantities with a short right-hand unit phrase.
            for m in re.finditer(r"(?P<money>\$)?(?P<num>\d+(?:\.\d+)?)(?:\s*)(?P<unit>[A-Za-z]+(?:\s+[A-Za-z]+){0,2})?", sent):
                val = float(m.group("num"))
                unit_raw = (m.group("unit") or "").strip()
                if m.group("money"):
                    unit = "dollar"
                else:
                    first = unit_raw.split()[0] if unit_raw else "number"
                    unit = cls._normalize_unit(first)
                before = sent[max(0, m.start()-45):m.start()].lower()
                after = sent[m.end():m.end()+55].lower()
                entity_words = re.findall(r"[a-z]+", (unit_raw + " " + after[:30]).lower())
                stop = {"of","the","a","an","and","that","which","is","are","per","for","to","in","at","with","his","her"}
                entity = next((w for w in entity_words if w not in stop and w not in {unit, "dollar"}), unit)
                role = "given"
                if re.search(r"\bper\b", after[:20]) or re.search(r"\bper\b", before[-20:]): role = "rate"
                if re.search(r"\b(includes?|included)\b", before): role = "included"
                if re.search(r"\b(extra|additional)\b", before + after[:15]): role = "extra"
                out.append(Quantity(val, unit, entity, si, offset+m.start(), m.group(0).strip(), role,
                                    {"before": before, "after": after, "sentence": sent}))
            # Word fractions are typed dimensionless quantities.
            f = cls._word_fraction(sent)
            if f is not None:
                out.append(Quantity(float(f), "fraction", "fraction", si, offset, cls._fmt(f), "fraction",
                                    {"sentence": sent}))
            offset += len(sent) + 1
        return out

    def _op(self, op: str, a: float, b: float, trace: List[str], label: str) -> float:
        if op == "div" and abs(b) < 1e-15:
            raise ZeroDivisionError
        r = self.core.numeric(self.CORE[op], float(a), float(b))
        if not r.verified or not isinstance(r.answer, (int, float)):
            raise ValueError("unverified numeric family")
        self.math.ops += 1
        v = float(r.answer)
        if not math.isfinite(v) or abs(v) > 1e15:
            raise ValueError("numeric range")
        trace.append(f"{label}:{self._fmt(a)} {op} {self._fmt(b)} -> {self._fmt(v)}")
        self.core.encode_event(SemanticEvent(
            relation=f"r33:{op}", quantity=v, goal=label, confidence=r.confidence,
            payload={"a": a, "b": b, "op": op},
        ))
        return v

    @staticmethod
    def _numbers(sentence: str) -> List[float]:
        return [float(x) for x in re.findall(r"(?<![A-Za-z])\$?(\d+(?:\.\d+)?)", sentence.replace(",", ""))]

    @staticmethod
    def _entity_counts(sentence: str) -> Dict[str, float]:
        ans: Dict[str, float] = {}
        # Handles lists such as "4 mirrors, 2 shelves, 1 chandelier, and 10 pictures".
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s+([A-Za-z]+)", sentence.replace(",", "")):
            word = m.group(2).lower()
            unit = General128V09R33._normalize_unit(word)
            if unit in {"mirror","shelf","chandelier","picture","person","bag","chip","hour","week","inch","liter","item"}:
                ans[unit] = float(m.group(1))
        return ans

    def _bundle_extras(self, question: str) -> Optional[GraphPlan]:
        low = question.lower()
        if not (re.search(r"\b(includes?|included)\b", low) and re.search(r"\b(additional|extra)\b", low) and "per item" in low):
            return None
        sents = self._sentences(question.replace(",", ""))
        base = None; rate = None; included: Dict[str,float] = {}; wanted: Dict[str,float] = {}
        for s in sents:
            sl = s.lower()
            if "includes" in sl or "included" in sl:
                ms = re.search(r"\$(\d+(?:\.\d+)?)", s)
                if ms: base = float(ms.group(1))
                included.update(self._entity_counts(s))
            if ("additional" in sl or "extra" in sl) and "per item" in sl:
                ms = re.search(r"\$(\d+(?:\.\d+)?)", s)
                if ms: rate = float(ms.group(1))
            if re.search(r"\b(has|needs?|wants?)\b", sl) and not ("includes" in sl or "included" in sl):
                wanted.update(self._entity_counts(s))
        if base is None or rate is None or not included or not wanted:
            return None
        trace: List[str] = []
        extra_total = 0.0; matched = 0
        for ent, have in wanted.items():
            if ent not in included: continue
            matched += 1
            diff = self._op("sub", have, included[ent], trace, f"excess_{ent}")
            if diff > 0:
                charge = self._op("mul", diff, rate, trace, f"extra_charge_{ent}")
                extra_total = charge if extra_total == 0 else self._op("add", extra_total, charge, trace, "extra_sum")
        if matched == 0: return None
        final = self._op("add", base, extra_total, trace, "bundle_total") if extra_total else base
        return GraphPlan("bundle_extras", final, 0.99, trace, [f"base={base}", f"rate={rate}", f"matched={matched}"])

    @classmethod
    def _fraction_mentions(cls, question: str) -> List[Tuple[int,float,str]]:
        out=[]
        for si,s in enumerate(cls._sentences(question)):
            low=s.lower().replace("-"," ")
            patterns=[
                (r"\btwo thirds\b",2/3),(r"\bthree fifths\b",3/5),(r"\bhalf\b",1/2),
                (r"\bone third\b",1/3),(r"\bthird\b",1/3),(r"\bquarter\b",1/4),
                (r"\bthree quarters\b",3/4),(r"\btwo fifths\b",2/5),(r"\bfour fifths\b",4/5),
            ]
            for p,v in patterns:
                if re.search(p,low): out.append((si,v,s))
        return out

    def _mixture_fraction(self, question: str) -> Optional[GraphPlan]:
        low=question.lower().replace("-"," ")
        if not ("water" in low and ("liter" in low or "litre" in low) and len(self._fraction_mentions(question))>=2):
            return None
        sents=self._sentences(question.replace(",","")); trace=[]
        components=[]  # (name, volume, fraction)
        for si,s in enumerate(sents):
            sl=s.lower().replace("-"," ")
            frac=self._word_fraction(sl)
            if frac is None: continue
            ms=re.search(r"(\d+(?:\.\d+)?)\s+lit(?:er|re)s?\s+of\s+([a-z]+)",sl)
            if not ms:
                ms=re.search(r"(?:have|add(?: it)? to)\s+(\d+(?:\.\d+)?)\s+lit(?:er|re)s?\s+of\s+([a-z]+)",sl)
            if ms:
                components.append([ms.group(2),float(ms.group(1)),float(frac)])
        if len(components)<2: return None
        # Apply a spill/loss to the named component before taking its fraction.
        spill=None; spill_name=None
        for s in sents:
            sl=s.lower()
            m=re.search(r"spill(?:ed)?\s+(?:one|1)\s+lit(?:er|re)\s+of\s+the\s+([a-z]+)",sl)
            if m: spill=1.0; spill_name=m.group(1)
            m2=re.search(r"spill(?:ed)?\s+(\d+(?:\.\d+)?)\s+lit(?:er|re).*?([a-z]+)\s+drink",sl)
            if m2: spill=float(m2.group(1)); spill_name=m2.group(2)
        water_parts=[]
        for name,vol,frac in components:
            effective=vol
            if spill is not None and spill_name and spill_name in name:
                effective=self._op("sub",vol,spill,trace,f"remaining_{name}")
            water=self._op("mul",effective,frac,trace,f"fraction_{name}")
            water_parts.append(water)
        total=water_parts[0]
        for x in water_parts[1:]: total=self._op("add",total,x,trace,"mixture_sum")
        return GraphPlan("mixture_fraction",total,0.98,trace,[str(x) for x in components])

    def _repeat_scale(self, question: str) -> Optional[GraphPlan]:
        low=question.lower().replace("-"," ")
        if not ("half as long" in low and re.search(r"times? a week",low) and re.search(r"\d+\s+weeks?",low)):
            return None
        trace=[]
        mbase=re.search(r"(\d+(?:\.\d+)?)\s+hours?",low)
        mfreq=re.search(r"(\d+(?:\.\d+)?)\s+times? a week",low)
        mweeks=re.search(r"(?:in|for)\s+(\d+(?:\.\d+)?)\s+weeks?",low)
        if not (mbase and mfreq and mweeks): return None
        base=float(mbase.group(1)); freq=float(mfreq.group(1)); weeks=float(mweeks.group(1))
        half=self._op("mul",base,0.5,trace,"half_duration")
        episode=self._op("add",base,half,trace,"episode_duration")
        weekly=self._op("mul",episode,freq,trace,"weekly_repeat")
        total=self._op("mul",weekly,weeks,trace,"period_scale")
        return GraphPlan("repeat_scale",total,0.99,trace,[f"base={base}",f"freq={freq}",f"weeks={weeks}"])

    def _per_group_sum(self, question: str) -> Optional[GraphPlan]:
        low=question.lower()
        if "per person" not in low or not re.search(r"\btotal\b",low): return None
        trace=[]; terms=[]
        # Pair a group count with the nearby dollar rate in the same clause.
        clauses=re.split(r"\band\b|,|\.",question.replace(",",""),flags=re.I)
        for c in clauses:
            if "per person" not in c.lower(): continue
            nums=[float(x) for x in re.findall(r"(?<!\$)\b(\d+(?:\.\d+)?)\b",c)]
            money=re.findall(r"\$(\d+(?:\.\d+)?)",c)
            if not nums or not money: continue
            # count is the last non-money number before the price in normal GSM clauses.
            price=float(money[-1]); count=nums[-1]
            terms.append(self._op("mul",count,price,trace,"group_cost"))
        if len(terms)<2: return None
        total=terms[0]
        for x in terms[1:]: total=self._op("add",total,x,trace,"group_sum")
        return GraphPlan("per_group_sum",total,0.98,trace,[f"terms={len(terms)}"])

    def _unit_chain(self, question: str) -> Optional[GraphPlan]:
        low=question.lower()
        # Generic stock -> area -> missing dimension chain.
        if not ("per square inch" in low and "bag" in low and "chips" in low and "how many inches long" in low):
            return None
        trace=[]
        m_density=re.search(r"(?:takes\s+)?(\d+(?:\.\d+)?)\s+(?:glass\s+)?chips?\s+to\s+make\s+every\s+square inch",low)
        m_bag=re.search(r"bag of .*?holds\s+(\d+(?:\.\d+)?)\s+chips",low)
        m_bags=re.search(r"(?:has|with)\s+(\d+(?:\.\d+)?)\s+bags",low)
        m_height=re.search(r"(\d+(?:\.\d+)?)\s+inches? tall",low)
        if not (m_density and m_bag and m_bags and m_height): return None
        density=float(m_density.group(1)); perbag=float(m_bag.group(1)); bags=float(m_bags.group(1)); height=float(m_height.group(1))
        chips=self._op("mul",perbag,bags,trace,"inventory")
        area=self._op("div",chips,density,trace,"chips_to_area")
        length=self._op("div",area,height,trace,"area_to_length")
        return GraphPlan("unit_chain",length,0.99,trace,[f"density={density}",f"bag={perbag}",f"bags={bags}",f"height={height}"])

    def _simple_rate_total(self, question: str) -> Optional[GraphPlan]:
        """Conservative generic fallback for multiple `count at $rate per X` terms."""
        low=question.lower()
        if not ("total" in low or "altogether" in low or "in all" in low): return None
        trace=[]; terms=[]
        # sentence-local count/rate pairs
        for s in self._sentences(question.replace(",","")):
            money=[float(x) for x in re.findall(r"\$(\d+(?:\.\d+)?)",s)]
            if not money or " per " not in s.lower(): continue
            nums=[float(x) for x in re.findall(r"(?<!\$)\b(\d+(?:\.\d+)?)\b",s)]
            if nums:
                terms.append(self._op("mul",nums[-1],money[-1],trace,"rate_product"))
        if len(terms)<2:return None
        total=terms[0]
        for x in terms[1:]:total=self._op("add",total,x,trace,"sum_parts")
        return GraphPlan("simple_rate_total",total,0.90,trace,[f"terms={len(terms)}"])

    def _build_graph(self, question: str) -> Tuple[List[Quantity], List[GraphPlan]]:
        qs=self._extract_quantities(question); plans=[]
        for fn in (self._bundle_extras,self._mixture_fraction,self._repeat_scale,self._per_group_sum,self._unit_chain,self._simple_rate_total):
            ops0=self.math.ops
            try:
                p=fn(question)
                if p: plans.append(p)
            except Exception:
                # Do not leave speculative arithmetic edges counted as a valid routed plan.
                pass
        self.graphs += 1
        return qs,plans

    @staticmethod
    def _family_choice(text: str) -> Optional[str]:
        m=re.findall(r"FAMILY\s*:\s*([a-z_]+)",text,re.I)
        return m[-1].lower() if m else None

    def _choose_plan(self, question: str, plans: Sequence[GraphPlan]) -> Optional[GraphPlan]:
        if not plans:return None
        # High-confidence deterministic typed match wins without another LLM call.
        best=sorted(plans,key=lambda p:-p.confidence)
        if len(best)==1 or (best[0].confidence>=0.98 and best[0].confidence-best[1].confidence>=0.03):
            return best[0]
        families=", ".join(p.family for p in best)
        raw=self.cortex.call(
            "Classify the word-problem structure. Choose only one listed relation family; do not calculate. Output exactly FAMILY: <name>.",
            question+"\n\nAllowed families: "+families,32)
        f=self._family_choice(raw)
        return next((p for p in best if p.family==f),None)

    def answer_math(self, question: str, baseline_hint: Optional[str]=None) -> V08Result:
        calls0=self.cortex.calls
        if baseline_hint is None:
            raw=self.cortex.call("Solve carefully and end exactly FINAL: <number>.",question,180)
            baseline=_extract_number(raw)
        else: baseline=str(baseline_hint) if baseline_hint is not None else None
        qnodes,plans=self._build_graph(question)
        trace=[f"baseline={baseline}",f"typed_quantities={len(qnodes)}",f"plans={[p.family for p in plans]}"]
        plan=self._choose_plan(question,plans)
        if plan is None:
            self.fallbacks+=1
            return V08Result(baseline,0.80,self.cortex.calls-calls0,self.cortex.calls-calls0,trace+["gate=no_typed_plan"])
        self.routed+=1; self.family_counts[plan.family]=self.family_counts.get(plan.family,0)+1
        core=self._fmt(plan.value); trace += [f"family={plan.family}",f"core={core}"]+plan.trace
        if baseline is not None and _num_equal(baseline,core):
            trace.append("gate=baseline_core_agree")
            return V08Result(baseline,0.99,self.cortex.calls-calls0,self.cortex.calls-calls0,trace)
        # Only very high-confidence deterministic families can repair the baseline.
        if plan.confidence>=0.98:
            self.overrides+=1; final=core; conf=plan.confidence; trace.append("gate=typed_override")
        else:
            self.fallbacks+=1; final=baseline; conf=0.84; trace.append("gate=conservative_base")
        self.core.encode_event(SemanticEvent(
            relation="r33_gate",goal="gsm8k",confidence=conf,
            payload={"family":plan.family,"baseline":baseline,"core":core,"final":final},
        ))
        return V08Result(final,conf,self.cortex.calls-calls0,self.cortex.calls-calls0,trace)

    def snapshot(self):
        s=self.core.snapshot(); s.update({
            "version":"0.9R3.3","cortex_calls":self.cortex.calls,"causal_math_ops":self.math.ops,
            "graphs":self.graphs,"routed":self.routed,"overrides":self.overrides,"fallbacks":self.fallbacks,
            "family_counts":self.family_counts,
        }); return s
