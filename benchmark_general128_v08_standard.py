from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import load_dataset

from benchmark_llm_standard_v01 import (
    eval_mc, eval_gsm8k, load_mmlu, load_arc, load_hellaswag,
    load_winogrande, load_truthfulqa, fixed_sample, gsm8k_gold,
)
from general128_v08 import General128V08


def wilson(p: float, n: int, z: float = 1.96):
    import math
    if n <= 0: return [0.0, 0.0]
    den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return [max(0.0,center-half),min(1.0,center+half)]


def eval_hybrid_mc(g: General128V08, rows: List[Tuple[str,List[str],int]], task: str) -> Dict[str,Any]:
    correct=0; parsed=0; raw=[]; calls0=g.cortex.calls; t0=time.time(); cycles=[]
    for i,(q,choices,gold) in enumerate(rows):
        r=g.answer_mc(q,choices)
        pred=r.answer if isinstance(r.answer,int) else None
        parsed += int(pred is not None)
        correct += int(pred==gold)
        cycles.append(r.cycles)
        raw.append({"i":i,"gold":gold,"pred":pred,"cycles":r.cycles,"calls":r.cortex_calls,"trace":r.trace})
    n=len(rows); acc=correct/n if n else None
    return {"task":task,"n":n,"correct":correct,"accuracy":acc,"parse_rate":parsed/n if n else None,
            "seconds":time.time()-t0,"cortex_calls":g.cortex.calls-calls0,
            "mean_cycles":sum(cycles)/len(cycles) if cycles else 0,"examples":raw,"accuracy_ci95":wilson(acc,n)}


def eval_hybrid_gsm(g: General128V08,n:int,seed:int)->Dict[str,Any]:
    ds=load_dataset("openai/gsm8k","main",split="test")
    xs=fixed_sample(ds,n,seed); correct=0; parsed=0; raw=[]; calls0=g.cortex.calls; t0=time.time(); cycles=[]
    for i,x in enumerate(xs):
        r=g.answer_math(x["question"])
        pred=str(r.answer) if r.answer is not None else None
        gold=gsm8k_gold(x["answer"])
        parsed += int(pred is not None)
        try: ok=pred is not None and abs(float(pred)-float(gold)) <= 1e-7*max(1.0,abs(float(gold)))
        except Exception: ok=pred==gold
        correct += int(ok); cycles.append(r.cycles)
        raw.append({"i":i,"gold":gold,"pred":pred,"cycles":r.cycles,"calls":r.cortex_calls,"trace":r.trace})
    acc=correct/len(xs) if xs else None
    return {"task":"GSM8K","n":len(xs),"correct":correct,"accuracy":acc,"parse_rate":parsed/len(xs) if xs else None,
            "seconds":time.time()-t0,"cortex_calls":g.cortex.calls-calls0,
            "mean_cycles":sum(cycles)/len(cycles) if cycles else 0,"examples":raw,"accuracy_ci95":wilson(acc,len(xs))}


def run_base(endpoint:str,samples:int,seed:int,name:str)->Dict[str,Any]:
    tasks=[]
    loaders=[("MMLU",load_mmlu),("ARC-Challenge",load_arc),("HellaSwag",load_hellaswag),
             ("WinoGrande",load_winogrande),("TruthfulQA-MC1",load_truthfulqa)]
    for j,(task,loader) in enumerate(loaders):
        r=eval_mc(endpoint,loader(samples,seed+101*j),task)
        r["accuracy_ci95"]=wilson(r["accuracy"],r["n"]); tasks.append(r)
    r=eval_gsm8k(endpoint,samples,seed+999); r["accuracy_ci95"]=wilson(r["accuracy"],r["n"]); tasks.append(r)
    return {"model":name,"tasks":tasks,"macro_accuracy":sum(x["accuracy"] for x in tasks)/len(tasks)}


def run_hybrid(endpoint:str,samples:int,seed:int)->Dict[str,Any]:
    g=General128V08(endpoint); tasks=[]
    loaders=[("MMLU",load_mmlu),("ARC-Challenge",load_arc),("HellaSwag",load_hellaswag),
             ("WinoGrande",load_winogrande),("TruthfulQA-MC1",load_truthfulqa)]
    for j,(task,loader) in enumerate(loaders): tasks.append(eval_hybrid_mc(g,loader(samples,seed+101*j),task))
    tasks.append(eval_hybrid_gsm(g,samples,seed+999))
    return {"model":"General-128 V0.8 + SmolLM2-360M-Instruct-Q4_K_M","tasks":tasks,
            "macro_accuracy":sum(x["accuracy"] for x in tasks)/len(tasks),"snapshot":g.snapshot()}


def compact(m:Dict[str,Any]):
    return {"model":m["model"],"macro_accuracy":m["macro_accuracy"],
            "tasks":{x["task"]:{"accuracy":x["accuracy"],"correct":x["correct"],"n":x["n"],
                                 "cortex_calls":x.get("cortex_calls"),"mean_cycles":x.get("mean_cycles")}
                     for x in m["tasks"]}}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--smol-endpoint",required=True); ap.add_argument("--qwen-endpoint",required=True)
    ap.add_argument("--samples",type=int,default=30); ap.add_argument("--seed",type=int,default=20260906); ap.add_argument("--out",default="v08_standard_results.json")
    a=ap.parse_args()
    q=run_base(a.qwen_endpoint,a.samples,a.seed,"Qwen2.5-0.5B-Instruct-Q4_K_M")
    s=run_base(a.smol_endpoint,a.samples,a.seed,"SmolLM2-360M-Instruct-Q4_K_M")
    h=run_hybrid(a.smol_endpoint,a.samples,a.seed)
    out={"protocol":{"name":"general128-v08-standard-sampled-v0.1","samples_per_task":a.samples,"seed":a.seed,
                     "note":"Same sampled standard datasets. Qwen/Smol are single-pass; V0.8 uses adaptive 2-3 cortex calls and an internal safe arithmetic core."},
         "models":[q,s,h],"compact":[compact(q),compact(s),compact(h)]}
    Path(a.out).write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out["compact"],indent=2))

if __name__=="__main__": main()
