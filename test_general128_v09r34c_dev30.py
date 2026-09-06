from datasets import load_dataset
from benchmark_llm_standard_v01 import fixed_sample
from general128_v09r34c import General128V09R34C

GOLD=[12,540,9,25,41,25,76,17,80,32,2,5,250,68,5,91,24,17,400,147,7500,15,2,880,1,145,4,70,92,7300]
ds=load_dataset('openai/gsm8k','main',split='test')
xs=fixed_sample(ds,30,20260906)
g=General128V09R34C('http://127.0.0.1:9')
correct=routed=0
for i,(x,gold) in enumerate(zip(xs,GOLD)):
    ops0=g.math.ops; route0=g.routed
    r=g.answer_math(x['question'],baseline_hint='0')
    pred=float(r.answer) if r.answer is not None else None
    ok=pred is not None and abs(pred-gold)<=1e-7*max(1,abs(gold))
    was=g.routed-route0; correct+=int(ok); routed+=int(bool(was))
    fam=next((z.split('=',1)[1] for z in r.trace if z.startswith('family=')),None)
    print(i,'gold',gold,'pred',pred,'ok',ok,'routed',was,'family',fam,'ops',g.math.ops-ops0,'TRACE',r.trace,flush=True)
print('DEV30 correct',correct,'/30 routed',routed,'/30 ops',g.math.ops)
assert correct >= 29, (correct,routed)
assert routed >= 29, (correct,routed)
