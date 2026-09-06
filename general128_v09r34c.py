from __future__ import annotations

"""General-128 V0.9R3.4C: final parser closure before a fresh blind seed.

Only three surface parser defects from the development set are changed here:
(1) choose the second person's inventory, not the first 'has' occurrence;
(2) parse two-period sales sentence-locally;
(3) extract multiple consumption rates independently in a shared sentence.
No new arithmetic relation family or answer constant is introduced.
"""

import re
from typing import Optional

from general128_v09r33 import GraphPlan
from general128_v09r34b import General128V09R34B


class General128V09R34C(General128V09R34B):
    def _infer_container_capacity(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        m=re.search(r"(\d+(?:\.\d+)?)\s+boxes full of\s+([a-z]+)s?\s+and\s+(\d+(?:\.\d+)?)\s+loose .*? total of\s+(\d+(?:\.\d+)?)",s)
        if not m:return None
        total=float(m.group(4)); noun=self._singular(m.group(2))
        # Take grounded inventories of the same item; the final distinct inventory is the other person.
        inventories=[]
        for mm in re.finditer(r"\bhas\s+(\d+(?:\.\d+)?)\s+([a-z]+)\b",s):
            if self._singular(mm.group(2))==noun: inventories.append(float(mm.group(1)))
        # If the first person's quantity is expressed as a total rather than 'has N noun', keep only a later person's count.
        other=inventories[-1] if inventories else None
        if other is None or abs(other-total)<1e-9:
            # explicit fallback anchored near sister/brother/friend/name clause
            mo=re.search(r"(?:sister|brother|friend).*?has\s+(\d+(?:\.\d+)?)\s+"+re.escape(noun)+r"s?",s)
            other=float(mo.group(1)) if mo else None
        if other is None:return None
        nbox=float(m.group(1)); loose=float(m.group(3))
        if nbox<=0:return None
        tr=[]; packed=self._op("sub",total,loose,tr,"packed_items"); cap=self._op("div",packed,nbox,tr,"items_per_box")
        alln=self._op("add",total,other,tr,"all_items"); boxes=self._op("div",alln,cap,tr,"boxes_needed")
        return GraphPlan("infer_container_capacity",boxes,.99,tr,[f"capacity={cap}",f"other={other}"])

    def _two_period_sales(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        sentences=self._sentences(s)
        sat=next((x for x in sentences if "on saturday" in x and "sold" in x),None)
        sun=next((x for x in sentences if x.startswith("on sunday") and "sold" in x),None)
        price_sent=next((x for x in sentences if x.count("$")>=2 and "cost" in x),None)
        if not sat or not sun or not price_sent or "two days" not in s:return None
        msat=re.search(r"sold\s+(\d+(?:\.\d+)?)\s+boxes of\s+(.+?)\s+and\s+(\d+(?:\.\d+)?)\s+fewer boxes of\s+(.+?)\s+than on sunday",sat)
        msun=re.search(r"sold\s+(\d+(?:\.\d+)?)\s+more boxes of\s+(.+?)\s+than on saturday\s+and\s+(\d+(?:\.\d+)?)\s+boxes of\s+(.+?)(?:\.|$)",sun)
        price_pairs=re.findall(r"(?:the\s+)?([a-z][a-z ]*?)\s+cost\s+\$(\d+(?:\.\d+)?)",price_sent)
        if not msat or not msun or len(price_pairs)<2:return None
        a0=float(msat.group(1)); a_name=msat.group(2).strip(); fewer=float(msat.group(3)); b_name=msat.group(4).strip()
        more=float(msun.group(1)); b1=float(msun.group(3))
        def price_for(name,default):
            for k,v in price_pairs:
                k=k.strip()
                if name in k or k in name:return float(v)
            return float(default)
        pa=price_for(a_name,price_pairs[0][1]); pb=price_for(b_name,price_pairs[-1][1])
        tr=[]; a1=self._op("add",a0,more,tr,"next_period_a"); b0=self._op("sub",b1,fewer,tr,"previous_period_b")
        ac=self._op("add",a0,a1,tr,"a_count_sum"); bc=self._op("add",b0,b1,tr,"b_count_sum")
        ar=self._op("mul",ac,pa,tr,"a_revenue"); br=self._op("mul",bc,pb,tr,"b_revenue")
        return GraphPlan("two_period_sales",self._op("add",ar,br,tr,"revenue_sum"),.985,tr,[a_name,b_name])

    def _daily_consumption_week(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        counts=re.search(r"keeps\s+(\d+(?:\.\d+)?)\s+.+?\s+and\s+(\d+(?:\.\d+)?)\s+.+?\.",s)
        rates=[float(x) for x in re.findall(r"consumes\s+(\d+(?:\.\d+)?)\s+kilograms?",s)]
        if not counts or len(rates)<2 or "in a week" not in s:return None
        n1=float(counts.group(1)); n2=float(counts.group(2)); r1,r2=rates[:2]
        tr=[]; a=self._op("mul",n1,r1,tr,"group_daily_1"); b=self._op("mul",n2,r2,tr,"group_daily_2")
        daily=self._op("add",a,b,tr,"daily_total")
        return GraphPlan("daily_consumption_week",self._op("mul",daily,7,tr,"week_scale"),.99,tr,[])

    def snapshot(self):
        s=super().snapshot(); s["version"]="0.9R3.4C"; return s
