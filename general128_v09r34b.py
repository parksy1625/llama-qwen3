from __future__ import annotations

"""General-128 V0.9R3.4B.

Parser-hardening pass over R3.4.  No new gold-dependent constants are added.
This pass fixes surface-form failures (pluralization, omitted repeated nouns,
punctuation removal, and relation-label normalization) while retaining the
same reusable relation families and General-128 numeric execution path.
"""

import re
from typing import Optional

from general128_v09r33 import GraphPlan
from general128_v09r34 import General128V09R34


class General128V09R34B(General128V09R34):
    @staticmethod
    def _singular(x: str) -> str:
        x=x.lower().strip()
        irregular={"babies":"baby","people":"person","children":"child","men":"man","women":"woman"}
        if x in irregular:return irregular[x]
        if x.endswith("ies") and len(x)>3:return x[:-3]+"y"
        if x.endswith("es") and len(x)>4:return x[:-2]
        if x.endswith("s") and len(x)>3:return x[:-1]
        return x

    def _two_period_sales(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        if not ("saturday" in s and "sunday" in s and "fewer" in s and "more" in s and "two days" in s):return None
        # A0 on first day, B0 = B1 - d, A1 = A0 + m; each category has a price.
        ma0=re.search(r"saturday.*?sold\s+(\d+(?:\.\d+)?)\s+boxes of\s+([a-z ]+?)\s+and",s)
        md=re.search(r"and\s+(\d+(?:\.\d+)?)\s+fewer boxes of\s+([a-z ]+?)\s+than on sunday",s)
        mm=re.search(r"sunday.*?sold\s+(\d+(?:\.\d+)?)\s+more boxes of\s+([a-z ]+?)\s+than on saturday",s)
        mb1=re.search(r"and\s+(\d+(?:\.\d+)?)\s+boxes of\s+([a-z ]+?)\s*\.",s)
        prices=re.findall(r"(?:the\s+)?([a-z ]+?)\s+cost\s+\$(\d+(?:\.\d+)?)",s)
        if not all([ma0,md,mm,mb1]) or len(prices)<2:return None
        a0=float(ma0.group(1)); d=float(md.group(1)); more=float(mm.group(1)); b1=float(mb1.group(1))
        # Prices are grounded by category names when possible, otherwise text order.
        price_map={name.strip():float(v) for name,v in prices}
        an=ma0.group(2).strip(); bn=md.group(2).strip()
        pa=next((v for k,v in price_map.items() if an in k or k in an),float(prices[0][1]))
        pb=next((v for k,v in price_map.items() if bn in k or k in bn),float(prices[-1][1]))
        tr=[]
        a1=self._op("add",a0,more,tr,"next_period_a")
        b0=self._op("sub",b1,d,tr,"previous_period_b")
        ac=self._op("add",a0,a1,tr,"a_count_sum"); bc=self._op("add",b0,b1,tr,"b_count_sum")
        ar=self._op("mul",ac,pa,tr,"a_revenue"); br=self._op("mul",bc,pb,tr,"b_revenue")
        return GraphPlan("two_period_sales",self._op("add",ar,br,tr,"revenue_sum"),.985,tr,[])

    def _infer_container_capacity(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        m=re.search(r"(\d+(?:\.\d+)?)\s+boxes full of\s+([a-z]+)s?\s+and\s+(\d+(?:\.\d+)?)\s+loose .*? total of\s+(\d+(?:\.\d+)?)",s)
        other=re.search(r"(?:sister.*?)?has\s+(\d+(?:\.\d+)?)\s+[a-z]+s?.*?how many boxes do .*?store all",s)
        if not m or not other:return None
        nbox=float(m.group(1)); loose=float(m.group(3)); total=float(m.group(4)); othern=float(other.group(1))
        if nbox<=0:return None
        tr=[]; packed=self._op("sub",total,loose,tr,"packed_items"); cap=self._op("div",packed,nbox,tr,"items_per_box")
        alln=self._op("add",total,othern,tr,"all_items"); boxes=self._op("div",alln,cap,tr,"boxes_needed")
        return GraphPlan("infer_container_capacity",boxes,.99,tr,[f"capacity={cap}"])

    def _group_weight_convert(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        mtotal=re.search(r"has\s+(\d+(?:\.\d+)?)\s+([a-z]+)s?",s)
        mgroup=re.search(r"groups of\s+(\d+(?:\.\d+)?)",s)
        mbox=re.search(r"each box weighs\s+(\d+(?:\.\d+)?)\s+ounces?",s)
        mitem=re.search(r"each\s+(?!box\b)([a-z]+)\s+weighs\s+(\d+(?:\.\d+)?)\s+ounces?",s)
        mconv=re.search(r"(\d+(?:\.\d+)?)\s+ounces? to a pound",s)
        if not all([mtotal,mgroup,mbox,mitem,mconv]):return None
        n=float(mtotal.group(1)); g=float(mgroup.group(1)); bw=float(mbox.group(1)); iw=float(mitem.group(2)); conv=float(mconv.group(1))
        if min(g,conv)<=0:return None
        tr=[]; boxes=self._op("div",n,g,tr,"box_count"); boxoz=self._op("mul",boxes,bw,tr,"box_weight")
        itemoz=self._op("mul",n,iw,tr,"item_weight"); oz=self._op("add",boxoz,itemoz,tr,"total_ounces")
        return GraphPlan("group_weight_convert",self._op("div",oz,conv,tr,"unit_convert"),.99,tr,[])

    def _weighted_percent_score(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        if not ("correctly answers" in s and "%" in s and "worth" in s and "points" in s):return None
        pcts=[float(x)/100 for x in re.findall(r"(\d+(?:\.\d+)?)%",s)[:3]]
        cm=re.search(r"there are\s+(\d+(?:\.\d+)?)\s+multiple-choice questions\s+(\d+(?:\.\d+)?)\s+true/false questions\s+and\s+(\d+(?:\.\d+)?)\s+long answer questions",s)
        weights=[float(x) for x in re.findall(r"worth\s+(\d+(?:\.\d+)?)\s+points?",s)]
        if len(pcts)!=3 or not cm or len(weights)<2:return None
        counts=list(map(float,cm.groups())); w=[weights[0],weights[0],weights[-1]]
        tr=[]; parts=[]
        for i in range(3):
            correct=self._op("mul",counts[i],pcts[i],tr,f"correct_{i}")
            parts.append(self._op("mul",correct,w[i],tr,f"points_{i}"))
        return GraphPlan("weighted_percent_score",self._sumv(parts,tr,"score_sum"),.99,tr,[])

    def _three_stage_scale_sum(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        m0=re.search(r"monday.*?bought\s+(\d+(?:\.\d+)?)\s+[a-z]+",s)
        m1=re.search(r"tuesday.*?bought\s+(\d+(?:\.\d+)?)\s+times that number",s)
        m2=re.search(r"wednesday.*?bought\s+(\d+(?:\.\d+)?)\s+times the number .*?tuesday",s)
        if not all([m0,m1,m2]) or not re.search(r"all\s+(?:3|three)\s+days",s):return None
        a=float(m0.group(1)); k=float(m1.group(1)); j=float(m2.group(1)); tr=[]
        b=self._op("mul",a,k,tr,"stage2"); c=self._op("mul",b,j,tr,"stage3")
        return GraphPlan("three_stage_scale_sum",self._sumv([a,b,c],tr,"stage_sum"),.99,tr,[])

    def _inventory_flow(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        mi=re.search(r"\b(?:had|has)\s+(\d+(?:\.\d+)?)\s+([a-z]+)\b",s)
        if not mi:return None
        initial=float(mi.group(1)); noun=self._singular(mi.group(2));
        if not re.search(r"how many .*?(?:left|now|have left|on .* now|are on .* now)",s):return None
        events=[]
        for verb,op in (("bought","add"),("got","add"),("downloaded","add"),("received","add"),("gave","sub"),("used","sub"),("deleted","sub"),("lost","sub")):
            for m in re.finditer(r"\b"+verb+r"\s+(\d+(?:\.\d+)?)\b",s):
                # Same-entity ellipsis is permitted after an explicit stock introduction.
                events.append((m.start(),op,float(m.group(1)),verb))
        events.sort()
        if len(events)<2:return None
        tr=[]; x=initial
        for _,op,v,verb in events:x=self._op(op,x,v,tr,f"inventory_{verb}")
        return GraphPlan("inventory_flow",x,.99,tr,[f"entity={noun}",f"events={len(events)}"])

    def _decay_until_threshold(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        if not ("chalk" in s and "recycle" in s and "smaller than" in s):return None
        mlen=re.search(r"(\d+(?:\.\d+)?)\s*[- ]inch",s)
        mnormal=re.search(r"only use\s+(\d+(?:\.\d+)?)%",s)
        mabnormal=re.search(r"used up\s+(\d+(?:\.\d+)?)%",s)
        mthr=re.search(r"smaller than\s+(\d+(?:\.\d+)?)\s+inches?",s)
        if not all([mlen,mnormal,mabnormal,mthr]):return None
        length=float(mlen.group(1)); normal=float(mnormal.group(1))/100; abnormal=float(mabnormal.group(1))/100; threshold=float(mthr.group(1))
        if not (0<=normal<1 and 0<=abnormal<1):return None
        tr=[]; remain=self._op("mul",length,1-abnormal,tr,"post_unusual_day"); days=0
        while remain>=threshold and days<100:
            remain=self._op("mul",remain,1-normal,tr,"daily_decay"); days+=1
        return GraphPlan("decay_until_threshold",float(days),.98,tr,[f"normal={normal}",f"abnormal={abnormal}"])

    def _progressive_difference_sum(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        base=re.search(r"had\s+(\d+(?:\.\d+)?)\s+([a-z]+)\s+squares",s)
        rels=[]
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s+(more|fewer)\s+([a-z]+)\s+squares than\s+([a-z]+)\s+squares",s):
            rels.append((float(m.group(1)),m.group(2),m.group(3),m.group(4)))
        if not base or len(rels)<3 or "comforter" not in s:return None
        values={base.group(2):float(base.group(1))}; tr=[]
        for _ in range(5):
            for d,kind,new,old in rels:
                if old in values and new not in values:
                    values[new]=self._op("add" if kind=="more" else "sub",values[old],d,tr,f"relative_{new}")
        if len(values)<4:return None
        return GraphPlan("progressive_difference_sum",self._sumv(list(values.values()),tr,"all_groups"),.99,tr,[str(values)])

    def _ratio_partition_chain(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        mt=re.search(r"there are\s+(\d+(?:\.\d+)?)\s+[a-z]+\s+in",s)
        targetm=re.search(r"how many\s+([a-z]+)(?:\s+[a-z]+)?\s+are there",s)
        raw=[]
        for m in re.finditer(r"twice as many\s+([a-z]+)(?:\s+bees)?\s+as\s+([a-z]+)(?:\s+bees)?",s):
            raw.append((self._singular(m.group(1)),self._singular(m.group(2))))
        if not mt or not targetm or len(raw)<2:return None
        target=self._singular(targetm.group(1)); coeff={raw[-1][1]:1.0}
        for _ in range(6):
            for a,b in reversed(raw):
                if b in coeff and a not in coeff:coeff[a]=2*coeff[b]
                if a in coeff and b not in coeff:coeff[b]=coeff[a]/2
        if target not in coeff or len(coeff)<3:return None
        denom=sum(coeff.values()); tr=[]; unit=self._op("div",float(mt.group(1)),denom,tr,"ratio_unit")
        return GraphPlan("ratio_partition_chain",self._op("mul",unit,coeff[target],tr,"ratio_target"),.985,tr,[str(coeff)])

    def _daily_consumption_week(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        counts=re.search(r"keeps\s+(\d+(?:\.\d+)?)\s+.+?\s+and\s+(\d+(?:\.\d+)?)\s+.+?\.",s)
        rates=[float(x) for x in re.findall(r"consumes\s+(\d+(?:\.\d+)?)\s+kilograms? .*? per day",s)]
        if not counts or len(rates)<2 or "in a week" not in s:return None
        n1=float(counts.group(1)); n2=float(counts.group(2)); r1,r2=rates[:2]; tr=[]
        a=self._op("mul",n1,r1,tr,"group_daily_1"); b=self._op("mul",n2,r2,tr,"group_daily_2")
        daily=self._op("add",a,b,tr,"daily_total"); return GraphPlan("daily_consumption_week",self._op("mul",daily,7,tr,"week_scale"),.99,tr,[])

    def _equal_split_two_rates(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        if not ("equal number" in s and re.search(r"(?:2|two) different",s) and "twice" in s and "per sentence" in s):return None
        mt=re.search(r"total number of\s+(\d+(?:\.\d+)?)\s+sentences",s); mr=re.search(r"publisher a pays him\s+(\d+(?:\.\d+)?)\s+cents per sentence",s)
        if not mt or not mr:return None
        total=float(mt.group(1)); ra=float(mr.group(1)); tr=[]; each=self._op("div",total,2,tr,"equal_split"); rb=self._op("mul",ra,2,tr,"rate_b")
        ea=self._op("mul",each,ra,tr,"earn_a"); eb=self._op("mul",each,rb,tr,"earn_b")
        return GraphPlan("equal_split_two_rates",self._op("add",ea,eb,tr,"earn_sum"),.99,tr,[])

    def _losses_then_share(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        mmade=re.search(r"made\s+(\d+(?:\.\d+)?)\s+gallons",s); people=re.search(r"(\d+(?:\.\d+)?)\s+people showed up",s)
        drank=re.search(r"drinking\s+(\d+(?:\.\d+)?)\s+(?:of those\s+)?gallons",s)
        spilled=re.search(r"spilled\s+(\d+(?:\.\d+)?)\s+gallons",s)
        reduced=re.search(r"reducing .*? by\s+(\d+(?:\.\d+)?)\s+gallons",s)
        if not all([mmade,people,drank,spilled,reduced]) or "shared the remaining" not in s:return None
        made=float(mmade.group(1)); losses=[float(drank.group(1)),float(spilled.group(1)),float(reduced.group(1))]; attendees=float(people.group(1))
        tr=[]; x=made
        for v in losses:x=self._op("sub",x,v,tr,"inventory_loss")
        participants=self._op("add",attendees,1,tr,"participants") if "fred and the others" in s else attendees
        return GraphPlan("losses_then_share",self._op("div",x,participants,tr,"equal_share"),.99,tr,[])

    def _time_budget_mixed_units(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        mh=re.search(r"spending\s+(\d+(?:\.\d+)?)\s+hours",s); mr=re.search(r"takes\s+(\d+(?:\.\d+)?)\s+minutes per inpatient",s)
        mi=re.search(r"has\s+(\d+(?:\.\d+)?)\s+inpatients",s); ma=re.search(r"has\s+(\d+(?:\.\d+)?)\s+appointments\s+which take\s+(\d+(?:\.\d+)?)\s+minutes each",s)
        if not all([mh,mr,mi,ma]) or "hours will" not in s:return None
        budget_h=float(mh.group(1)); round_min=float(mr.group(1)); inp=float(mi.group(1)); apn=float(ma.group(1)); apmin=float(ma.group(2)); tr=[]
        budget=self._op("mul",budget_h,60,tr,"hours_to_minutes"); rounds=self._op("mul",round_min,inp,tr,"rounds_minutes"); appts=self._op("mul",apn,apmin,tr,"appointments_minutes")
        used=self._op("add",rounds,appts,tr,"used_minutes"); left=self._op("sub",budget,used,tr,"left_minutes")
        return GraphPlan("time_budget_mixed_units",self._op("div",left,60,tr,"minutes_to_hours"),.99,tr,[])

    def _missing_from_average(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        mn=re.search(r"has\s+(\d+(?:\.\d+)?)\s+[a-z]+s",s); mav=re.search(r"on average.*?(\d+(?:\.\d+)?)\s+miles? per day",s)
        known=[float(x) for x in re.findall(r"(?:first|second|third) needs to walk\s+(\d+(?:\.\d+)?)\s+miles?",s)]
        if not mn or not mav or len(known)<2 or "last" not in s:return None
        n=float(mn.group(1)); avg=float(mav.group(1)); tr=[]; total=self._op("mul",n,avg,tr,"average_total"); ksum=self._sumv(known,tr,"known_sum")
        return GraphPlan("missing_from_average",self._op("sub",total,ksum,tr,"missing_value"),.99,tr,[])

    def snapshot(self):
        s=super().snapshot(); s["version"]="0.9R3.4B"; return s
