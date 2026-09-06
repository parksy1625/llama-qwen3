from __future__ import annotations
import argparse,json,re,time,urllib.request
from pathlib import Path
from benchmark_general128_v07 import build_suite,run_general128,score_model,write_outputs

def call(ep,messages):
    body=json.dumps({"model":"local-model","messages":messages,"temperature":0,"max_tokens":220,"stream":False}).encode()
    req=urllib.request.Request(ep,data=body,headers={"Content-Type":"application/json"},method="POST")
    with urllib.request.urlopen(req,timeout=300) as r: data=json.loads(r.read().decode())
    return data["choices"][0]["message"]["content"]

def obj(text):
    text=text.strip()
    try:
        x=json.loads(text); return x if isinstance(x,dict) else {}
    except Exception: pass
    m=re.search(r"\{.*\}",text,re.S)
    if not m:return {}
    try:
        x=json.loads(m.group()); return x if isinstance(x,dict) else {}
    except Exception:return {}

def batches(xs,n=4):
    for i in range(0,len(xs),n):yield xs[i:i+n]

def ask_batches(ep,system,base,queries,keys,kind,raw,pred):
    for i,b in enumerate(batches(queries)):
        qs=[{k:q[k] for k in keys} for q in b]
        text=call(ep,[{"role":"system","content":system},{"role":"user","content":base+json.dumps(qs,separators=(",",":"))}])
        raw[f"{kind}_{i}"]=text
        x=obj(text)
        for q in b:
            if q["id"] in x:pred[q["id"]]=x[q["id"]]

def run_qwen(s,ep):
    raw,pred,times={},{},{}
    demos=json.dumps(s["operator_demos"],separators=(",",":"))
    t=time.time(); ask_batches(ep,"Infer arbitrary numeric operators. JSON only.","DEMOS="+demos+". Every answer is NUMBER. Return flat JSON id:number. QUERIES=",s["numeric_queries"],["id","symbol","a","b"],"numeric",raw,pred); times["numeric"]=time.time()-t
    t=time.time(); ask_batches(ep,"Compute compositions. JSON only.","DEMOS="+demos+". Apply ALL steps in order. Every answer is NUMBER. Return flat JSON id:number. QUERIES=",s["composition_queries"],["id","initial","steps"],"composition",raw,pred); times["composition"]=time.time()-t
    facts=json.dumps(s["graph_domains"],separators=(",",":")); t=time.time(); ask_batches(ep,"Answer graph reachability as JSON booleans only.","FACTS="+facts+". before,causes,is_a,greater are transitive. Edge [A,B] means A relation B. Answer true iff b reachable from a for exact kind, else false. Return flat JSON id:true/false. QUERIES=",s["graph_queries"],["id","kind","a","b"],"graph",raw,pred); times["graph"]=time.time()-t
    c=s["continual"]; p1=json.dumps(c["phase1_demos"],separators=(",",":")); p2=json.dumps(c["phase2_demos"],separators=(",",":")); t=time.time()
    for i,b in enumerate(batches(c["final_queries"])):
        qs=[{k:q[k] for k in ("id","symbol","a","b")} for q in b]
        ms=[{"role":"system","content":"Learn numeric operators; final output JSON only."},{"role":"user","content":"PHASE1="+p1},{"role":"assistant","content":"OK"},{"role":"user","content":"PHASE2="+p2},{"role":"assistant","content":"OK"},{"role":"user","content":"Using both phases, every answer is NUMBER. Return flat JSON id:number. QUERIES="+json.dumps(qs,separators=(",",":"))}]
        text=call(ep,ms);raw[f"continual_{i}"]=text;x=obj(text)
        for q in b:
            if q["id"] in x:pred[q["id"]]=x[q["id"]]
    times["continual"]=time.time()-t
    return {"model":"Qwen2.5-0.5B-Instruct-Q4_K_M-fair","predictions":pred,"raw_outputs":raw,"latency_seconds":times}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--outdir",default="fair_results");ap.add_argument("--qwen-endpoint",required=True);a=ap.parse_args()
    s=build_suite();g=run_general128(s);q=run_qwen(s,a.qwen_endpoint);scores=[score_model(s,g),score_model(s,q)];write_outputs(Path(a.outdir),s,[g,q],scores);print(json.dumps(scores,indent=2));print("RAW",json.dumps(q["raw_outputs"]))
if __name__=="__main__":main()
