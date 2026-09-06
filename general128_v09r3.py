from __future__ import annotations

"""General-128 V0.9R3: dual constrained semantic-plan compiler.

R1 proved causal execution but accepted bad free-form plans. R2 prevented those
plans from overwriting the baseline. R3 changes the interface itself:

  question -> two independently prompted *bounded arithmetic plans*
           -> structural/constant grounding validation
           -> General-128 operator-family execution
           -> accept only if both plans independently reach the same number

The plan compiler never receives benchmark labels.  Literal numeric constants
must be grounded in the question, except small semantic factors explicitly
licensed by words such as twice/half/dozen/percent.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from general128_v07 import General128V07, SemanticEvent
from general128_v08 import LanguageCortex, V08Result, _extract_number, _num_equal
from general128_v09 import CausalArithmetic128


OPS = {"add":"g_add", "sub":"g_sub", "mul":"g_mul", "div":"g_div", "pow":"g_pow"}


@dataclass
class PlanResult:
    value: Optional[str]
    valid: bool
    reason: str
    ops: int
    plan: Optional[dict]


class General128V09R3:
    def __init__(self, cortex_endpoint: str):
        self.cortex = LanguageCortex(cortex_endpoint)
        self.core = General128V07()
        self.math = CausalArithmetic128(self.core)
        self.accepted_plans = 0
        self.rejected_plans = 0

    @staticmethod
    def _fmt(x: float) -> str:
        if abs(x-round(x)) < 1e-9:
            return str(int(round(x)))
        return ("%.10f" % x).rstrip("0").rstrip(".")

    @staticmethod
    def _question_numbers(question: str) -> List[float]:
        vals=[]
        for x in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", question.replace(',', '')):
            try: vals.append(float(x))
            except: pass
        return vals

    @staticmethod
    def _licensed_constants(question: str) -> List[float]:
        q=question.lower(); out=[]
        rules=[
            (r"\b(twice|double)\b",2.0),(r"\btriple\b",3.0),(r"\bhalf\b",0.5),
            (r"\bquarter\b",0.25),(r"\bdozen\b",12.0),(r"\bpercent\b|%",100.0),
        ]
        for pat,val in rules:
            if re.search(pat,q): out.append(val)
        return out

    @staticmethod
    def _extract_plan(text: str) -> Optional[dict]:
        # Prefer fenced/marked JSON but tolerate surrounding prose.
        m=re.search(r"PLAN\s*:\s*(\{.*\})",text,re.I|re.S)
        candidates=[]
        if m: candidates.append(m.group(1))
        l=text.find('{'); r=text.rfind('}')
        if l>=0 and r>l: candidates.append(text[l:r+1])
        for s in candidates:
            try:
                obj=json.loads(s)
                if isinstance(obj,dict): return obj
            except Exception:
                continue
        return None

    def _grounded(self, x: float, question: str) -> bool:
        allowed=self._question_numbers(question)+self._licensed_constants(question)
        return any(abs(x-y)<=1e-9*max(1.0,abs(y)) for y in allowed)

    def _operand(self, x: Any, env: Dict[str,float], question: str) -> Tuple[Optional[float],str]:
        if isinstance(x,(int,float)) and not isinstance(x,bool):
            v=float(x)
            return (v,"literal") if self._grounded(v,question) else (None,"ungrounded_literal")
        if isinstance(x,str):
            s=x.strip()
            if s in env: return env[s],"ref"
            try:
                v=float(s)
                return (v,"literal") if self._grounded(v,question) else (None,"ungrounded_literal")
            except Exception:
                return None,"bad_ref"
        return None,"bad_operand"

    def execute_plan(self, plan: Optional[dict], question: str) -> PlanResult:
        if not isinstance(plan,dict):
            self.rejected_plans+=1; return PlanResult(None,False,"no_plan",0,plan)
        steps=plan.get("steps"); answer_ref=plan.get("answer")
        if not isinstance(steps,list) or not (1 <= len(steps) <= 6):
            self.rejected_plans+=1; return PlanResult(None,False,"bad_step_count",0,plan)
        env: Dict[str,float]={}; ops0=self.math.ops
        for i,st in enumerate(steps):
            if not isinstance(st,dict):
                self.rejected_plans+=1; return PlanResult(None,False,"bad_step",self.math.ops-ops0,plan)
            op=str(st.get("op","")).lower(); out=str(st.get("out",f"r{i}"))
            if op not in OPS or not re.fullmatch(r"r\d+",out) or out in env:
                self.rejected_plans+=1; return PlanResult(None,False,"bad_op_or_out",self.math.ops-ops0,plan)
            a,ra=self._operand(st.get("a"),env,question); b,rb=self._operand(st.get("b"),env,question)
            if a is None or b is None:
                self.rejected_plans+=1; return PlanResult(None,False,f"operand:{ra}/{rb}",self.math.ops-ops0,plan)
            try:
                r=self.core.numeric(OPS[op],a,b)
                if not r.verified or not isinstance(r.answer,(int,float)):
                    raise ValueError("unverified")
                v=float(r.answer)
                if abs(v)>1e15: raise ValueError("range")
                env[out]=v
                self.math.ops+=1
                self.core.encode_event(SemanticEvent(
                    relation=f"r3:{op}",quantity=v,goal="validated_plan",
                    confidence=r.confidence,payload={"a":a,"b":b,"out":out,"step":i},
                ))
            except Exception:
                self.rejected_plans+=1; return PlanResult(None,False,"execution",self.math.ops-ops0,plan)
        if not isinstance(answer_ref,str) or answer_ref not in env:
            self.rejected_plans+=1; return PlanResult(None,False,"bad_answer_ref",self.math.ops-ops0,plan)
        self.accepted_plans+=1
        return PlanResult(self._fmt(env[answer_ref]),True,"ok",self.math.ops-ops0,plan)

    def _ask_plan(self, question: str, variant: int) -> Tuple[Optional[dict],str]:
        if variant==1:
            sys=(
              "Translate the word problem into a short arithmetic program. Do not solve mentally. "
              "Use JSON only with this schema: PLAN: {\"steps\":[{\"op\":\"add|sub|mul|div\",\"a\":NUMBER_OR_rN,\"b\":NUMBER_OR_rN,\"out\":\"rN\"}],\"answer\":\"rN\"}. "
              "Every numeric literal must come from the problem text. Use earlier rN results for intermediate values. Maximum 6 steps."
            )
        else:
            sys=(
              "Build an independent minimal calculation graph for the requested quantity. Return only JSON preceded by PLAN:. "
              "Schema exactly: {\"steps\":[{\"op\":\"add|sub|mul|div\",\"a\":NUMBER_OR_rN,\"b\":NUMBER_OR_rN,\"out\":\"rN\"}],\"answer\":\"rN\"}. "
              "Ground every number in the question and make dependencies explicit. Maximum 6 steps."
            )
        raw=self.cortex.call(sys,question,220)
        return self._extract_plan(raw),raw

    def answer_math(self, question: str, baseline_hint: Optional[str]=None) -> V08Result:
        calls0=self.cortex.calls
        if baseline_hint is None:
            raw=self.cortex.call("Solve carefully and end exactly FINAL: <number>.",question,180)
            baseline=_extract_number(raw)
        else:
            baseline=str(baseline_hint) if baseline_hint is not None else None

        p1,_=self._ask_plan(question,1); p2,_=self._ask_plan(question,2)
        r1=self.execute_plan(p1,question); r2=self.execute_plan(p2,question)
        trace=[f"baseline={baseline}",f"plan1={r1.value}/{r1.reason}/ops{r1.ops}",f"plan2={r2.value}/{r2.reason}/ops{r2.ops}"]

        if r1.valid and r2.valid and _num_equal(r1.value,r2.value):
            core=r1.value
            if baseline is None or not _num_equal(baseline,core):
                final=core; gate="dual_plan_override"; conf=0.95
            else:
                final=baseline; gate="triple_agree"; conf=0.98
        else:
            final=baseline; gate="baseline_fallback"; conf=0.80 if baseline is not None else 0.30
        trace.append("gate="+gate)
        self.core.encode_event(SemanticEvent(
            relation="r3_gate",goal="gsm8k",confidence=conf,
            payload={"baseline":baseline,"p1":r1.value,"p2":r2.value,"final":final,"gate":gate},
        ))
        return V08Result(final,conf,self.cortex.calls-calls0,self.cortex.calls-calls0,trace)

    def snapshot(self):
        s=self.core.snapshot(); s.update({
            "version":"0.9R3","cortex_calls":self.cortex.calls,
            "causal_math_ops":self.math.ops,"accepted_plans":self.accepted_plans,
            "rejected_plans":self.rejected_plans,
        }); return s
