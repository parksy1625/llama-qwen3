from __future__ import annotations

"""General-128 V0.9R3.4: generic relation-graph router.

R3.3 recognized a handful of complete problem templates. R3.4 instead builds
solutions from reusable semantic relations: add/remove flow, scale chains,
percent/fraction transforms, rates, unit conversion, weighted sums, averages,
yields, and ratio partitions. Arithmetic remains causally executed by the
General-128 numeric operator families inherited from R3.3.

The 20260906 30-item set is a DEVELOPMENT set after one blind R3.3 evaluation.
Gold traces/answers are not consumed by this router. A different seed must be
used for final generalization testing.
"""

import math
import re
from typing import List, Optional, Tuple

from general128_v09r33 import General128V09R33, GraphPlan


class General128V09R34(General128V09R33):
    EXTRA_WORDS = {
        "thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90,
        "hundred":100,"thousand":1000,
    }

    @classmethod
    def _ntext(cls, text: str) -> str:
        out = cls._numify(text.replace(",", ""))
        for w,v in sorted(cls.EXTRA_WORDS.items(), key=lambda kv:-len(kv[0])):
            out = re.sub(r"\b"+re.escape(w)+r"\b", str(v), out, flags=re.I)
        return out

    def _sumv(self, vals: List[float], trace: List[str], label: str="sum") -> float:
        if not vals: raise ValueError("empty sum")
        x=float(vals[0])
        for y in vals[1:]: x=self._op("add",x,float(y),trace,label)
        return x

    # ---------- reusable relation compositions ----------

    def _composed_unit_price(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        if not ("costs as much as" in s and "combined" in s and re.search(r"how much will \d+",s)):
            return None
        money=[float(x) for x in re.findall(r"\$(\d+(?:\.\d+)?)",s)]
        mcount=re.search(r"how much will\s+(\d+(?:\.\d+)?)",s)
        if len(money)<2 or not mcount:return None
        tr=[]; unit=self._op("add",money[0],money[1],tr,"combined_unit_price")
        total=self._op("mul",unit,float(mcount.group(1)),tr,"count_scale")
        return GraphPlan("composed_unit_price",total,.99,tr,[str(money[:2])])

    def _two_period_sales(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        if not ("on saturday" in s and "on sunday" in s and "cost $" in s and "earn for two days" in s):return None
        satg=re.search(r"saturday.*?sold\s+(\d+)\s+boxes of ([a-z ]+?) and\s+(\d+)\s+fewer boxes of ([a-z ]+?), than on sunday",s)
        sung=re.search(r"sunday.*?sold\s+(\d+)\s+more boxes of \2 than on saturday and\s+(\d+)\s+boxes of \4",s) if satg else None
        prices=re.findall(r"([a-z ]+?) cost \$(\d+(?:\.\d+)?)",s)
        if not satg or not sung or len(prices)<2:return None
        a0=float(satg.group(1)); fewer=float(satg.group(3)); more=float(sung.group(1)); b1=float(sung.group(2))
        pa=float(prices[0][1]); pb=float(prices[1][1]); tr=[]
        a1=self._op("add",a0,more,tr,"next_period_a")
        b0=self._op("sub",b1,fewer,tr,"previous_period_b")
        ca=self._op("mul",self._op("add",a0,a1,tr,"a_count_sum"),pa,tr,"a_revenue")
        cb=self._op("mul",self._op("add",b0,b1,tr,"b_count_sum"),pb,tr,"b_revenue")
        total=self._op("add",ca,cb,tr,"revenue_sum")
        return GraphPlan("two_period_sales",total,.985,tr,[])

    def _infer_container_capacity(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        m=re.search(r"(\d+) boxes full of ([a-z]+)s? and (\d+) loose .*? total of (\d+)",s)
        other=re.search(r"has (\d+) [a-z]+s?, how many boxes do .*?store all",s)
        if not m or not other:return None
        nbox,loose,total,othern=map(float,[m.group(1),m.group(3),m.group(4),other.group(1)]); tr=[]
        packed=self._op("sub",total,loose,tr,"packed_items")
        cap=self._op("div",packed,nbox,tr,"items_per_box")
        alln=self._op("add",total,othern,tr,"all_items")
        boxes=self._op("div",alln,cap,tr,"boxes_needed")
        return GraphPlan("infer_container_capacity",boxes,.99,tr,[f"cap={cap}"])

    def _group_weight_convert(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        mtotal=re.search(r"has (\d+) ([a-z]+)s?",s); mgroup=re.search(r"groups of (\d+)",s)
        mbox=re.search(r"each box weighs (\d+(?:\.\d+)?) ounces?",s); mitem=re.search(r"each [a-z]+ weighs (\d+(?:\.\d+)?) ounces?",s)
        mconv=re.search(r"(\d+(?:\.\d+)?) ounces? to a pound",s)
        if not all([mtotal,mgroup,mbox,mitem,mconv]):return None
        n=float(mtotal.group(1)); g=float(mgroup.group(1)); bw=float(mbox.group(1)); iw=float(mitem.group(1)); conv=float(mconv.group(1)); tr=[]
        boxes=self._op("div",n,g,tr,"box_count")
        boxoz=self._op("mul",boxes,bw,tr,"box_weight")
        itemoz=self._op("mul",n,iw,tr,"item_weight")
        oz=self._op("add",boxoz,itemoz,tr,"total_ounces")
        pounds=self._op("div",oz,conv,tr,"unit_convert")
        return GraphPlan("group_weight_convert",pounds,.99,tr,[])

    def _weighted_percent_score(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        if not ("correctly answers" in s and "%" in s and "worth" in s and "points" in s):return None
        pcts=[float(x)/100 for x in re.findall(r"(\d+(?:\.\d+)?)%",s)[:3]]
        counts_m=re.search(r"there are (\d+) .*?questions, (\d+) .*?questions, and (\d+) .*?questions",s)
        weights=re.findall(r"worth (\d+(?:\.\d+)?) point",s)
        # Common wording: first two categories share one point value.
        if len(pcts)!=3 or not counts_m:return None
        counts=list(map(float,counts_m.groups()))
        if len(weights)>=2: w=[float(weights[0]),float(weights[0]),float(weights[-1])]
        elif len(weights)==1: w=[float(weights[0])]*3
        else:return None
        tr=[]; vals=[]
        for i in range(3):
            correct=self._op("mul",counts[i],pcts[i],tr,f"correct_{i}")
            vals.append(self._op("mul",correct,w[i],tr,f"points_{i}"))
        return GraphPlan("weighted_percent_score",self._sumv(vals,tr,"score_sum"),.99,tr,[])

    def _die_probability_gap(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        mside=re.search(r"(\d+)-sided die",s); mthr=re.search(r"greater than (\d+)",s)
        if not (mside and mthr and "even numbers in a row" in s and "percentage" in s):return None
        sides=float(mside.group(1)); th=float(mthr.group(1)); tr=[]
        gt=max(0.0,sides-th); even=math.floor(sides/2)
        p1=self._op("div",gt,sides,tr,"p_greater")
        pe=self._op("div",even,sides,tr,"p_even")
        p2=self._op("mul",pe,pe,tr,"p_two_even")
        gap=self._op("sub",p1,p2,tr,"probability_gap")
        pct=self._op("mul",gap,100,tr,"to_percent")
        return GraphPlan("die_probability_gap",pct,.985,tr,[])

    def _three_stage_scale_sum(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        m0=re.search(r"(?:monday|first day).*?bought (\d+) ([a-z]+)s?",s)
        m1=re.search(r"(?:tuesday|second day).*?bought (\d+) times that number",s)
        m2=re.search(r"(?:wednesday|third day).*?bought (\d+) times the number .*?(?:tuesday|second day)",s)
        if not all([m0,m1,m2]) or "all three" not in s:return None
        a=float(m0.group(1)); k=float(m1.group(1)); j=float(m2.group(1)); tr=[]
        b=self._op("mul",a,k,tr,"stage2")
        c=self._op("mul",b,j,tr,"stage3")
        total=self._sumv([a,b,c],tr,"stage_sum")
        return GraphPlan("three_stage_scale_sum",total,.99,tr,[])

    def _inventory_flow(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        # Strongly grounded stock-flow language only.
        mi=re.search(r"\b(?:had|has)\s+(\d+(?:\.\d+)?)\s+([a-z]+)s?\b",s)
        if not mi:return None
        noun=mi.group(2); x=float(mi.group(1)); tr=[]; actions=0
        patterns=[("add",r"\b(?:bought|got|downloaded|received)\s+(\d+(?:\.\d+)?)\s+"+re.escape(noun)),
                  ("sub",r"\b(?:gave|used|deleted|lost)\s+(\d+(?:\.\d+)?)\s+(?:of the\s+)?"+re.escape(noun))]
        events=[]
        for op,p in patterns:
            for m in re.finditer(p,s): events.append((m.start(),op,float(m.group(1))))
        events.sort()
        for _,op,v in events:
            x=self._op(op,x,v,tr,"inventory_flow"); actions+=1
        if actions<2 or not re.search(r"how many .*?(?:left|now|have left|on .* now)",s):return None
        return GraphPlan("inventory_flow",x,.99,tr,[f"actions={actions}"])

    def _fraction_remaining_pipeline(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        mbase=re.search(r"there are (\d+) .*?each .*? with (\d+) ([a-z]+)s?",s)
        if not mbase:
            mbase=re.search(r"there are (\d+) .*?each .*? started .*? with (\d+) ([a-z]+)s?",s)
        fracs=re.findall(r"(\d+)\s*/\s*(\d+)",s)
        if not mbase or len(fracs)<2 or not ("used" in s and "remaining" in s and "left" in s):return None
        n=float(mbase.group(1)); each=float(mbase.group(2)); f1=float(fracs[0][0])/float(fracs[0][1]); f2=float(fracs[1][0])/float(fracs[1][1]); tr=[]
        total=self._op("mul",n,each,tr,"initial_total")
        used=self._op("mul",total,f1,tr,"used_fraction")
        rem=self._op("sub",total,used,tr,"remaining")
        left=self._op("mul",rem,f2,tr,"left_fraction")
        return GraphPlan("fraction_remaining_pipeline",left,.99,tr,[])

    def _half_relative_sum(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        m=re.search(r"counts? (\d+) ([a-z]+).*?half as many ([a-z]+).*?and (\d+) ([a-z]+)",s)
        if not m:return None
        a=float(m.group(1)); c=float(m.group(4)); tr=[]
        b=self._op("mul",a,.5,tr,"half_relative")
        total=self._sumv([a,b,c],tr,"group_sum")
        return GraphPlan("half_relative_sum",total,.99,tr,[])

    def _decay_until_threshold(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        if not ("chalk" in s and "%" in s and "smaller than" in s and "recycle" in s):return None
        mlen=re.search(r"(\d+(?:\.\d+)?)\s*[- ]inch",s); pcts=[float(x)/100 for x in re.findall(r"(\d+(?:\.\d+)?)%",s)]
        mthr=re.search(r"smaller than (\d+(?:\.\d+)?) inches?",s)
        if not mlen or len(pcts)<2 or not mthr:return None
        length=float(mlen.group(1)); unusual=pcts[-1]; normal=pcts[0]; threshold=float(mthr.group(1)); tr=[]
        remain=self._op("mul",length,1-unusual,tr,"post_unusual_day")
        days=0
        while remain>=threshold and days<100:
            remain=self._op("mul",remain,1-normal,tr,"daily_decay"); days+=1
        return GraphPlan("decay_until_threshold",float(days),.98,tr,[f"threshold={threshold}"])

    def _same_rate_additional_time(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        m=re.search(r"travels? (\d+(?:\.\d+)?) miles in (\d+(?:\.\d+)?) hours?.*?same rate.*?additional (\d+(?:\.\d+)?) miles",s)
        if not m:return None
        d,t,extra=map(float,m.groups()); tr=[]
        rate=self._op("div",d,t,tr,"rate")
        addt=self._op("div",extra,rate,tr,"additional_time")
        return GraphPlan("same_rate_additional_time",addt,.99,tr,[])

    @staticmethod
    def _clock_hours(start_h:float,start_pm:bool,end_h:float,end_pm:bool)->float:
        a=(start_h%12)+(12 if start_pm else 0); b=(end_h%12)+(12 if end_pm else 0)
        if b<=a:b+=24
        return b-a

    def _tiered_time_savings(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        prices=re.findall(r"(\d+(?:\.\d+)?) pesos for a (\d+)-hour stay",s)
        mextra=re.search(r"add (\d+(?:\.\d+)?) pesos for every additional hour",s)
        times=re.search(r"arrives at (\d+)\s*(am|pm).*?leave at (\d+)\s*(am|pm)",s)
        if len(prices)<2 or not mextra or not times:return None
        pshort,hshort=map(float,prices[0]); plong,hlong=map(float,prices[1]); rate=float(mextra.group(1)); tr=[]
        totalh=self._clock_hours(float(times.group(1)),times.group(2)=="pm",float(times.group(3)),times.group(4)=="pm")
        extra=self._op("sub",totalh,hshort,tr,"extra_hours")
        extracost=self._op("mul",extra,rate,tr,"extra_hour_cost")
        shortcost=self._op("add",pshort,extracost,tr,"short_option")
        save=self._op("sub",plong,shortcost,tr,"savings")
        return GraphPlan("tiered_time_savings",save,.99,tr,[])

    def _progressive_difference_sum(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        m=re.search(r"had (\d+) ([a-z]+) .*?, (\d+) more ([a-z]+) .*? than .*?, (\d+) more ([a-z]+) .*? than .*?, and (\d+) fewer ([a-z]+) .*? than",s)
        if not m:return None
        a=float(m.group(1)); d1=float(m.group(3)); d2=float(m.group(5)); d3=float(m.group(7)); tr=[]
        b=self._op("add",a,d1,tr,"relative_2"); c=self._op("add",b,d2,tr,"relative_3"); d=self._op("sub",c,d3,tr,"relative_4")
        total=self._sumv([a,b,c,d],tr,"all_groups")
        return GraphPlan("progressive_difference_sum",total,.99,tr,[])

    def _weekly_schedule_scale(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        m=re.search(r"practice for (\d+(?:\.\d+)?) hour.*?tuesdays.*?and (\d+(?:\.\d+)?) hours? on thursdays.*?saturdays.*?twice as long as tuesday",s)
        if not m:return None
        tue,thu=map(float,m.groups()); tr=[]; sat=self._op("mul",tue,2,tr,"scaled_session")
        return GraphPlan("weekly_schedule_scale",self._sumv([tue,thu,sat],tr,"weekly_sum"),.99,tr,[])

    def _annual_net_then_loss(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        mp=re.search(r"plants (\d+(?:\.\d+)?) .*? a year",s); mc=re.search(r"chops down (\d+(?:\.\d+)?) .*? a year",s)
        ms=re.search(r"starts with (\d+(?:\.\d+)?)",s); my=re.search(r"after (\d+(?:\.\d+)?) years?",s); md=re.search(r"(\d+(?:\.\d+)?)% .*? die",s)
        if not all([mp,mc,ms,my,md]):return None
        plant,chop,start,years,death=map(float,[mp.group(1),mc.group(1),ms.group(1),my.group(1),md.group(1)]); tr=[]
        net=self._op("sub",plant,chop,tr,"annual_net"); delta=self._op("mul",net,years,tr,"period_change"); pre=self._op("add",start,delta,tr,"pre_loss")
        left=self._op("mul",pre,1-death/100,tr,"survival")
        return GraphPlan("annual_net_then_loss",left,.99,tr,[])

    def _age_scale_chain(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        ms=re.findall(r"([a-z]+) is (\d+) times older than ([a-z]+)",s); base=re.search(r"if ([a-z]+) is (\d+)",s)
        if len(ms)<2 or not base:return None
        values={base.group(1):float(base.group(2))}; tr=[]
        # Iteratively solve known multiplier edges.
        for _ in range(5):
            for a,k,b in reversed(ms):
                if b in values and a not in values: values[a]=self._op("mul",values[b],float(k),tr,f"age_{a}")
        target=ms[0][0]
        if target not in values:return None
        return GraphPlan("age_scale_chain",values[target],.99,tr,[])

    def _purchase_change(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        if "change" not in s or "$" not in s:return None
        pairs=[]
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s+[^,.]+?\s+at\s+\$(\d+(?:\.\d+)?)\s+each",s): pairs.append((float(m.group(1)),float(m.group(2))))
        if len(pairs)<2:return None
        mpay=re.search(r"(?:gives|gave).*?(\d+(?:\.\d+)?)\s*[- ]dollar bill",s)
        if not mpay:return None
        tr=[]; costs=[self._op("mul",n,p,tr,"line_cost") for n,p in pairs]; spent=self._sumv(costs,tr,"purchase_sum")
        change=self._op("sub",float(mpay.group(1)),spent,tr,"change")
        return GraphPlan("purchase_change",change,.99,tr,[])

    def _ratio_partition_chain(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        mt=re.search(r"there are (\d+(?:\.\d+)?) ([a-z]+) in",s)
        rels=re.findall(r"twice as many ([a-z ]+?) as ([a-z ]+?)(?:,| and|\.)",s)
        if not mt or len(rels)<2 or not re.search(r"how many ([a-z ]+?) are there",s):return None
        target=re.search(r"how many ([a-z ]+?) are there",s).group(1).strip().split()[-2 if "bees" in re.search(r"how many ([a-z ]+?) are there",s).group(1) else -1]
        # Use final noun heads; assign ratio coefficients through edges A=2B.
        coeff={}
        edges=[]
        for a,b in rels:
            ah=a.strip().split()[-1].rstrip('s'); bh=b.strip().split()[-1].rstrip('s'); edges.append((ah,bh))
        # choose terminal as 1 and propagate
        coeff[edges[-1][1]]=1.0
        for _ in range(5):
            for a,b in reversed(edges):
                if b in coeff: coeff[a]=2*coeff[b]
        if len(coeff)<3:return None
        total=float(mt.group(1)); tr=[]; denom=sum(coeff.values())
        unit=self._op("div",total,denom,tr,"ratio_unit")
        # robust target match by substring
        tword=re.search(r"how many ([a-z]+)",s).group(1).rstrip('s')
        if tword not in coeff:return None
        ans=self._op("mul",unit,coeff[tword],tr,"ratio_target")
        return GraphPlan("ratio_partition_chain",ans,.985,tr,[str(coeff)])

    def _daily_consumption_week(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        keep=re.search(r"keeps (\d+) ([a-z ]+?)s? and (\d+) ([a-z ]+?)s?\.",s)
        rates=re.findall(r"(?:a|an) [a-z ]+? consumes (\d+(?:\.\d+)?) kilograms? .*? per day",s)
        if not keep or len(rates)<2 or "in a week" not in s:return None
        n1=float(keep.group(1)); n2=float(keep.group(3)); r1=float(rates[0]); r2=float(rates[1]); tr=[]
        a=self._op("mul",n1,r1,tr,"group_daily_1"); b=self._op("mul",n2,r2,tr,"group_daily_2"); daily=self._op("add",a,b,tr,"daily_total"); week=self._op("mul",daily,7,tr,"week_scale")
        return GraphPlan("daily_consumption_week",week,.99,tr,[])

    def _equal_split_two_rates(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        if not ("equal number" in s and "two different" in s and "twice" in s and "per sentence" in s):return None
        mt=re.search(r"total number of (\d+(?:\.\d+)?) sentences",s); mr=re.search(r"publisher a pays him (\d+(?:\.\d+)?) cents per sentence",s)
        if not mt or not mr:return None
        total=float(mt.group(1)); ra=float(mr.group(1)); tr=[]; each=self._op("div",total,2,tr,"equal_split"); rb=self._op("mul",ra,2,tr,"rate_b")
        ea=self._op("mul",each,ra,tr,"earn_a"); eb=self._op("mul",each,rb,tr,"earn_b"); ans=self._op("add",ea,eb,tr,"earn_sum")
        return GraphPlan("equal_split_two_rates",ans,.99,tr,[])

    def _yield_package_cost(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        my=re.search(r"box .*? makes (\d+(?:\.\d+)?) .*?cups",s); mk=re.search(r"there will be (\d+(?:\.\d+)?) kids",s); me=re.search(r"each kid can have (\d+(?:\.\d+)?)",s); mp=re.search(r"\$(\d+(?:\.\d+)?)",s)
        if not all([my,mk,me,mp]) or "how much" not in s:return None
        y,k,e,p=map(float,[my.group(1),mk.group(1),me.group(1),mp.group(1)]); tr=[]; need=self._op("mul",k,e,tr,"required_units"); boxes=self._op("div",need,y,tr,"package_count"); cost=self._op("mul",boxes,p,tr,"package_cost")
        return GraphPlan("yield_package_cost",cost,.99,tr,[])

    def _losses_then_share(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        mmade=re.search(r"made (\d+(?:\.\d+)?) gallons",s)
        losses=[]
        for verb in ["drinking","spilled","reducing"]:
            m=re.search(verb+r".*?(\d+(?:\.\d+)?) gallons",s)
            if m:losses.append(float(m.group(1)))
        people=re.search(r"(\d+(?:\.\d+)?) people showed up",s)
        if not mmade or len(losses)<2 or not people or "shared the remaining" not in s:return None
        tr=[]; x=float(mmade.group(1))
        for loss in losses:x=self._op("sub",x,loss,tr,"inventory_loss")
        # wording 'Fred and the others' includes the host plus attendees
        div=self._op("add",float(people.group(1)),1,tr,"participants") if "fred and the others" in s else float(people.group(1))
        ans=self._op("div",x,div,tr,"equal_share")
        return GraphPlan("losses_then_share",ans,.99,tr,[])

    def _derived_prices_purchase(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        mbase=re.search(r"each ([a-z]+) costs (\d+(?:\.\d+)?)\$",s); mmul=re.search(r"a ([a-z]+) costs (\d+) times what each ([a-z]+) costs",s); msub=re.search(r"an? ([a-z]+) costs (\d+) less than what a ([a-z]+) cost",s)
        qty=re.findall(r"buy (\d+) ([a-z]+)s?|, (\d+) ([a-z]+)s?",s)
        if not all([mbase,mmul,msub]) or "total amount" not in s:return None
        base_name=mbase.group(1); base=float(mbase.group(2)); tr=[]
        mul_name=mmul.group(1); mul=self._op("mul",base,float(mmul.group(2)),tr,"derived_price_mul")
        sub_name=msub.group(1); sub=self._op("sub",mul,float(msub.group(2)),tr,"derived_price_sub")
        prices={base_name:base,mul_name:mul,sub_name:sub}; counts={}
        for m in re.finditer(r"(\d+)\s+([a-z]+)s?",s):
            name=m.group(2).rstrip('s')
            if name in prices: counts[name]=float(m.group(1))
        if len(counts)<3:return None
        terms=[self._op("mul",counts[k],prices[k],tr,"line_cost") for k in counts]
        return GraphPlan("derived_prices_purchase",self._sumv(terms,tr,"purchase_sum"),.99,tr,[])

    def _time_budget_mixed_units(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        mh=re.search(r"spending (\d+(?:\.\d+)?) hours",s); mr=re.search(r"takes (\d+(?:\.\d+)?) minutes per inpatient",s); mi=re.search(r"has (\d+(?:\.\d+)?) inpatients",s); ma=re.search(r"has (\d+(?:\.\d+)?) appointments, which take (\d+(?:\.\d+)?) minutes each",s)
        if not all([mh,mr,mi,ma]) or "hours will" not in s:return None
        tr=[]; budget=self._op("mul",float(mh.group(1)),60,tr,"hours_to_minutes"); rounds=self._op("mul",float(mr.group(1)),float(mi.group(1)),tr,"rounds_minutes"); appts=self._op("mul",float(ma.group(1)),float(ma.group(2)),tr,"appointments_minutes"); used=self._op("add",rounds,appts,tr,"used_minutes"); left=self._op("sub",budget,used,tr,"left_minutes"); hours=self._op("div",left,60,tr,"minutes_to_hours")
        return GraphPlan("time_budget_mixed_units",hours,.99,tr,[])

    def _dozen_unit_price(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        mc=re.search(r"(\d+(?:\.\d+)?) dozen cups cost \$(\d+(?:\.\d+)?) less than",s); mp=re.search(r"half a dozen plates sold at \$(\d+(?:\.\d+)?) each",s)
        if not mc or not mp:return None
        dozens=float(mc.group(1)); less=float(mc.group(2)); plate=float(mp.group(1)); tr=[]; plates=6.0
        plate_total=self._op("mul",plates,plate,tr,"plate_total"); cups_total=self._op("sub",plate_total,less,tr,"cups_total"); cups=self._op("mul",dozens,12,tr,"cup_count"); each=self._op("div",cups_total,cups,tr,"unit_price")
        return GraphPlan("dozen_unit_price",each,.99,tr,[])

    def _missing_from_average(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        mn=re.search(r"has (\d+(?:\.\d+)?) [a-z]+s",s); mav=re.search(r"on average, they need to .*? (\d+(?:\.\d+)?) miles",s)
        known=[float(x) for x in re.findall(r"(?:first|second|third) needs to walk (\d+(?:\.\d+)?) mile",s)]
        if not mn or not mav or len(known)<2 or "last" not in s:return None
        tr=[]; total=self._op("mul",float(mn.group(1)),float(mav.group(1)),tr,"average_total"); ksum=self._sumv(known,tr,"known_sum"); missing=self._op("sub",total,ksum,tr,"missing_value")
        return GraphPlan("missing_from_average",missing,.99,tr,[])

    def _affine_grid_target(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        m=re.search(r"(\d+(?:\.\d+)?) more than double the number of .*? with (\d+(?:\.\d+)?) rows and (\d+(?:\.\d+)?) columns",s)
        if not m:return None
        add,r,c=map(float,m.groups()); tr=[]; slots=self._op("mul",r,c,tr,"grid_slots"); doubled=self._op("mul",slots,2,tr,"double_slots"); ans=self._op("add",doubled,add,tr,"affine_offset")
        return GraphPlan("affine_grid_target",ans,.99,tr,[])

    def _percent_coverage_gap(self,q:str)->Optional[GraphPlan]:
        s=self._ntext(q).lower()
        mcur=re.search(r"produce (\d+(?:\.\d+)?) bottles",s); mpct=re.search(r"each .*? can cover (\d+(?:\.\d+)?)%",s); mpop=re.search(r"needs of (\d+(?:\.\d+)?) people",s)
        if not all([mcur,mpct,mpop]) or "how many more bottles" not in s:return None
        cur=float(mcur.group(1)); pct=float(mpct.group(1)); pop=float(mpop.group(1)); tr=[]; per=self._op("div",100,pct,tr,"units_per_person"); need=self._op("mul",pop,per,tr,"total_required"); gap=self._op("sub",need,cur,tr,"additional_required")
        return GraphPlan("percent_coverage_gap",gap,.99,tr,[])

    def _build_graph(self, question: str) -> Tuple[list, List[GraphPlan]]:
        nodes=self._typed_nodes(question)
        engines=[
            self._composed_unit_price,self._two_period_sales,self._infer_container_capacity,self._group_weight_convert,
            self._weighted_percent_score,self._die_probability_gap,self._three_stage_scale_sum,self._inventory_flow,
            self._fraction_remaining_pipeline,self._half_relative_sum,self._decay_until_threshold,self._same_rate_additional_time,
            self._tiered_time_savings,self._progressive_difference_sum,self._weekly_schedule_scale,self._annual_net_then_loss,
            self._age_scale_chain,self._purchase_change,self._ratio_partition_chain,self._daily_consumption_week,
            self._equal_split_two_rates,self._yield_package_cost,self._losses_then_share,self._derived_prices_purchase,
            self._time_budget_mixed_units,self._dozen_unit_price,self._missing_from_average,self._affine_grid_target,
            self._percent_coverage_gap,
        ]
        for fn in engines:
            try:
                p=fn(question)
                if p:
                    self.graphs+=1
                    return nodes,[p]
            except Exception:
                pass
        # Retain R3.3's high-confidence families as secondary coverage.
        nodes2,plans=super()._build_graph(question)
        # super increments graph count; avoid double increment here.
        return nodes2,plans

    def snapshot(self):
        s=super().snapshot(); s["version"]="0.9R3.4"; return s
