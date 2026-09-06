from __future__ import annotations
import argparse,json,time
from pathlib import Path
from datasets import load_dataset
from benchmark_llm_standard_v01 import fixed_sample,gsm8k_gold,call_chat,parse_final_number
from general128_v09r2 import General128V09R2

def eq(a,b):
    try:return a is not None and b is not None and abs(float(a)-float(b))<=1e-7*max(1.0,abs(float(b)))
    except:return a==b

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--endpoint',required=True); ap.add_argument('--samples',type=int,default=5); ap.add_argument('--seed',type=int,default=20261905); ap.add_argument('--out',default='v09r2_gsm_results.json'); a=ap.parse_args()
    ds=load_dataset('openai/gsm8k','main',split='test'); xs=fixed_sample(ds,a.samples,a.seed)
    g=General128V09R2(a.endpoint); rows=[]; base=hyb=0; t0=time.time()
    for i,x in enumerate(xs):
        out=call_chat(a.endpoint,'Solve grade-school math carefully.',x['question']+'\nSolve the problem. You may reason briefly, but end with exactly: FINAL: <number>',180)
        bp=parse_final_number(out); gold=gsm8k_gold(x['answer']); base+=int(eq(bp,gold))
        ops0=g.math.ops; tools0=g.tool_calls
        r=g.answer_math(x['question'], baseline_hint=bp)
        hp=str(r.answer) if r.answer is not None else None; hyb+=int(eq(hp,gold))
        ops=g.math.ops-ops0; tools=g.tool_calls-tools0
        rows.append({'i':i,'gold':gold,'baseline':bp,'v09r2':hp,'base_ok':eq(bp,gold),'v09r2_ok':eq(hp,gold),'cortex_calls':r.cortex_calls,'tool_calls':tools,'causal_ops':ops,'trace':r.trace})
        print(i,gold,bp,hp,'base',base,'r2',hyb,'tools',tools,'ops',ops,'gate',r.trace[-1] if r.trace else '',flush=True)
    result={'n':len(xs),'baseline_correct':base,'baseline_accuracy':base/len(xs),'v09r2_correct':hyb,'v09r2_accuracy':hyb/len(xs),'delta':(hyb-base)/len(xs),'seconds':time.time()-t0,'snapshot':g.snapshot(),'rows':rows}
    Path(a.out).write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__': main()
