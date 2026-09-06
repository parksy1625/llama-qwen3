from __future__ import annotations
import argparse,json,time
from pathlib import Path
from datasets import load_dataset
from benchmark_llm_standard_v01 import fixed_sample,gsm8k_gold,call_chat,parse_final_number
from general128_v09r32 import General128V09R32

def eq(a,b):
    try:return a is not None and b is not None and abs(float(a)-float(b))<=1e-7*max(1.0,abs(float(b)))
    except:return a==b

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--endpoint',required=True); ap.add_argument('--samples',type=int,default=5); ap.add_argument('--seed',type=int,default=20261905); ap.add_argument('--out',default='v09r32_gsm_results.json'); a=ap.parse_args()
    ds=load_dataset('openai/gsm8k','main',split='test'); xs=fixed_sample(ds,a.samples,a.seed)
    g=General128V09R32(a.endpoint); rows=[]; base=hyb=0; t0=time.time()
    for i,x in enumerate(xs):
        raw=call_chat(a.endpoint,'Solve grade-school math carefully.',x['question']+'\nSolve the problem. You may reason briefly, but end with exactly: FINAL: <number>',180)
        bp=parse_final_number(raw); gold=gsm8k_gold(x['answer']); base+=int(eq(bp,gold))
        ops0=g.math.ops; over0=g.overrides; cons0=g.consensus
        r=g.answer_math(x['question'],baseline_hint=bp); hp=str(r.answer) if r.answer is not None else None; hyb+=int(eq(hp,gold))
        row={'i':i,'gold':gold,'baseline':bp,'v09r32':hp,'base_ok':eq(bp,gold),'v09r32_ok':eq(hp,gold),'causal_ops':g.math.ops-ops0,'consensus':g.consensus-cons0,'override':g.overrides-over0,'trace':r.trace}
        rows.append(row)
        print(i,gold,bp,hp,'base',base,'r32',hyb,'ops',row['causal_ops'],'cons',row['consensus'],'over',row['override'],'TRACE',json.dumps(r.trace),flush=True)
    result={'n':len(xs),'baseline_correct':base,'baseline_accuracy':base/len(xs),'v09r32_correct':hyb,'v09r32_accuracy':hyb/len(xs),'delta':(hyb-base)/len(xs),'seconds':time.time()-t0,'snapshot':g.snapshot(),'rows':rows}
    Path(a.out).write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__': main()
