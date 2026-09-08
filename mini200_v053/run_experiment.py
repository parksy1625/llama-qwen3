import os, re, json, math, random, time, gc
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from mini200_v053 import Mini200V053, Mini200V053Config

SEED=int(os.getenv('SEED','20260909'))
STEPS=int(os.getenv('STEPS','150'))
GSM_GEN_N=int(os.getenv('GSM_GEN_N','10'))
GSM_NLL_N=int(os.getenv('GSM_NLL_N','40'))
MMLU_N=int(os.getenv('MMLU_N','40'))
LR=float(os.getenv('LR','0.00025'))
MODEL_ID=os.getenv('MODEL_ID','HuggingFaceTB/SmolLM2-135M-Instruct')
OUT=Path(os.getenv('OUTDIR','mini200_ci_out'))
OUT.mkdir(parents=True,exist_ok=True)
random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1,min(4,os.cpu_count() or 2)))

def p(msg): print(msg,flush=True)

def keymap(k):
    fixed={'model.embed_tokens.weight':'embed_tokens.weight','model.norm.weight':'norm.weight','lm_head.weight':'lm_head.weight'}
    if k in fixed: return fixed[k]
    if not k.startswith('model.layers.'): return None
    parts=k.split('.'); i=parts[2]; tail='.'.join(parts[3:])
    table={
      'input_layernorm.weight':f'layers.{i}.n1.weight',
      'post_attention_layernorm.weight':f'layers.{i}.n2.weight',
      'self_attn.q_proj.weight':f'layers.{i}.a.q.weight',
      'self_attn.k_proj.weight':f'layers.{i}.a.k.weight',
      'self_attn.v_proj.weight':f'layers.{i}.a.v.weight',
      'self_attn.o_proj.weight':f'layers.{i}.a.o.weight',
      'mlp.gate_proj.weight':f'layers.{i}.m.g.weight',
      'mlp.up_proj.weight':f'layers.{i}.m.u.weight',
      'mlp.down_proj.weight':f'layers.{i}.m.d.weight'}
    return table.get(tail)

def transplant_from_hf(hf,mini):
    src=hf.state_dict(); dst=mini.state_dict(); copied=0
    with torch.no_grad():
        for sk,sv in src.items():
            dk=keymap(sk)
            if dk is None: continue
            if dk in dst:
                if dst[dk].shape!=sv.shape: raise RuntimeError(f'shape mismatch {sk}->{dk}: {sv.shape} vs {dst[dk].shape}')
                dst[dk].copy_(sv.to(dst[dk].dtype)); copied+=1
        mini.lm_head.weight.copy_(mini.embed_tokens.weight)
    return copied

def plain_prompt(q): return f"Question: {q}\nGive only the final numeric answer.\nAnswer:"
def chat_prompt(tok,q):
    msgs=[{'role':'user','content':f"Solve this problem and give only the final numeric answer.\n\n{q}"}]
    try: return tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    except Exception: return plain_prompt(q)

def extract_final(answer):
    m=re.search(r'####\s*([^\n]+)',answer)
    s=(m.group(1) if m else answer.splitlines()[-1]).strip().replace(',','').replace('$','')
    m2=re.search(r'-?\d+(?:\.\d+)?',s)
    return m2.group(0) if m2 else s

def intermediate_values(answer):
    vals=[]
    for v in re.findall(r'<<[^<>]*=([^<>]+)>>',answer):
        v=v.strip().replace(',','').replace('$','').replace('%','')
        try: vals.append(float(v))
        except Exception: pass
    return vals

def signed_log(v): return math.copysign(math.log1p(abs(float(v))),float(v))

def anchor_until_sidecar(m,ids):
    x=m.embed_tokens(ids)
    for i,b in enumerate(m.layers):
        x=b(x)
        if i+1==m.cfg.insert_after_layer: return x,i+1
    raise RuntimeError('insert point not reached')

def workspace_states(m,anchor,R=4):
    sc=m.sidecar
    seed=sc.seed_proj(sc.anchor_norm(anchor[:,-1:]))
    w=sc.slots.expand(anchor.size(0),-1,-1)+sc.cfg.workspace_init_scale*seed
    states=[]
    for _ in range(R):
        for b in sc.blocks: w=b(w,anchor)
        states.append(w[:,0])
    return states

def logits_from_state(m,anchor,start,state):
    sc=m.sidecar; delta=sc.read_proj(sc.read_norm(state))
    h=anchor.clone(); h[:,-1]=h[:,-1]+torch.tanh(sc.read_gate)*delta
    for j in range(start,len(m.layers)): h=m.layers[j](h)
    return m.lm_head(m.norm(h[:,-1]))

def forward_last_depths(m,ids,depths=(1,2,4,8)):
    with torch.no_grad():
        anchor,start=anchor_until_sidecar(m,ids)
        states=workspace_states(m,anchor,max(depths)); out={}
        for d in depths: out[d]=logits_from_state(m,anchor,start,states[d-1])
    return out

def parity(tok,hf,mini):
    texts=['The capital of France is','2 + 3 =','A train travels 60 miles in']; rows=[]
    hf.eval(); mini.eval()
    with torch.no_grad():
        for t in texts:
            ids=tok(t,return_tensors='pt',add_special_tokens=True).input_ids
            a=hf(ids).logits.float(); b=mini(ids,recurrences=1).float(); c=mini(ids,recurrences=8).float()
            rows.append({'text':t,'hf_vs_mini_max_abs':float((a-b).abs().max()),'hf_vs_mini_mean_abs':float((a-b).abs().mean()),'argmax_equal':bool(torch.equal(a.argmax(-1),b.argmax(-1))),'r1_vs_r8_max_abs_gate0':float((b-c).abs().max())})
    return rows

def numeric_generate(m,tok,q,depth,max_new=8):
    prompt=chat_prompt(tok,q); ids=tok(prompt,return_tensors='pt',add_special_tokens=True).input_ids; base_len=ids.shape[1]; cur=ids
    for _ in range(max_new):
        lg=forward_last_depths(m,cur,(depth,))[depth]; nxt=lg.argmax(-1,keepdim=True); cur=torch.cat([cur,nxt],dim=1)
        txt=tok.decode(cur[0,base_len:],skip_special_tokens=True)
        if '\n' in txt or len(txt)>=20: break
    txt=tok.decode(cur[0,base_len:],skip_special_tokens=True).strip(); nums=re.findall(r'-?\d+(?:\.\d+)?',txt.replace(',',''))
    return (nums[-1] if nums else ''),txt

def gsm_gold_nll(m,tok,rows,depths=(1,2,4,8),limit=40):
    losses={d:[] for d in depths}; first={d:0 for d in depths}; used=0
    for row in rows[:limit]:
        gold=extract_final(row['answer']); prompt=chat_prompt(tok,row['question']); pids=tok(prompt,add_special_tokens=True).input_ids; aids=tok(gold,add_special_tokens=False).input_ids
        if not aids: continue
        ids=torch.tensor([pids],dtype=torch.long); lgs=forward_last_depths(m,ids,depths); target=torch.tensor([aids[0]])
        for d in depths:
            losses[d].append(float(F.cross_entropy(lgs[d],target))); first[d]+=int(int(lgs[d].argmax(-1))==aids[0])
        used+=1
    return {'n':used,'first_token_nll':{str(d):sum(losses[d])/max(1,len(losses[d])) for d in depths},'first_token_accuracy':{str(d):first[d]/max(1,used) for d in depths}}

def gsm_exact(m,tok,rows,depths=(1,2,4,8),limit=10):
    result={str(d):{'correct':0,'details':[]} for d in depths}
    for i,row in enumerate(rows[:limit]):
        gold=extract_final(row['answer'])
        for d in depths:
            pred,text=numeric_generate(m,tok,row['question'],d); ok=pred==gold
            result[str(d)]['correct']+=int(ok); result[str(d)]['details'].append({'i':i,'gold':gold,'pred':pred,'text':text})
    for d in depths: result[str(d)]['accuracy']=result[str(d)]['correct']/max(1,limit)
    return result

def mmlu_eval(m,tok,limit=40,depths=(1,2,4,8)):
    try: ds=load_dataset('cais/mmlu','management',split='test')
    except Exception as e: return {'error':repr(e)}
    labels=['A','B','C','D']; label_ids=[]
    for x in labels:
        ids=tok(x,add_special_tokens=False).input_ids
        if len(ids)!=1: ids=tok(' '+x,add_special_tokens=False).input_ids
        label_ids.append(ids[-1])
    counts={str(d):{x:0 for x in labels} for d in depths}; correct={str(d):0 for d in depths}; rows=[]
    for i,row in enumerate(ds.select(range(min(limit,len(ds))))):
        q=row['question']; choices=row['choices']; gold=int(row['answer'])
        prompt=f"Question: {q}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer: "
        ids=tok(prompt,return_tensors='pt',add_special_tokens=True).input_ids; lgs=forward_last_depths(m,ids,depths); rr={'i':i,'gold':labels[gold],'pred':{}}
        for d in depths:
            scores=lgs[d][0,label_ids]; pi=int(scores.argmax()); pred=labels[pi]
            counts[str(d)][pred]+=1; correct[str(d)]+=int(pi==gold); rr['pred'][str(d)]=pred
        rows.append(rr)
    n=len(rows); return {'n':n,'accuracy':{str(d):correct[str(d)]/max(1,n) for d in depths},'prediction_counts':counts,'rows':rows}

def freeze_backbone(m):
    for p0 in m.parameters(): p0.requires_grad=False
    for p0 in m.sidecar.parameters(): p0.requires_grad=True

def train_sidecar(m,tok,train_rows,steps=150):
    freeze_backbone(m); aux=nn.Sequential(nn.LayerNorm(m.cfg.hidden_size),nn.Linear(m.cfg.hidden_size,1))
    params=[p0 for p0 in m.sidecar.parameters() if p0.requires_grad]+list(aux.parameters())
    opt=torch.optim.AdamW(params,lr=LR,weight_decay=.01); rng=random.Random(SEED+77); logs=[]; t0=time.time(); m.train(); aux.train()
    for step in range(1,steps+1):
        row=train_rows[rng.randrange(len(train_rows))]; q=row['question']; answer=row['answer']; final=extract_final(answer); vals=intermediate_values(answer); prompt=chat_prompt(tok,q)
        ans_ids=tok(final,add_special_tokens=False).input_ids
        if not ans_ids: continue
        k=rng.randrange(min(len(ans_ids),4)); prefix=tok(prompt,add_special_tokens=True).input_ids+ans_ids[:k]; prefix=prefix[-192:]
        ids=torch.tensor([prefix],dtype=torch.long); target=torch.tensor([ans_ids[k]],dtype=torch.long)
        with torch.no_grad(): anchor,start=anchor_until_sidecar(m,ids)
        states=workspace_states(m,anchor,4); traj=[]
        if vals:
            for r in range(4): traj.append(vals[min(r,len(vals)-1)])
        else:
            try: fv=float(final); traj=[fv]*4
            except Exception: traj=[0.0]*4
        auxloss=torch.tensor(0.0)
        for r in range(4):
            y=torch.tensor([signed_log(traj[r])],dtype=states[r].dtype); auxloss=auxloss+F.smooth_l1_loss(aux(states[r]).squeeze(-1),y)
        auxloss=auxloss/4
        logits4=logits_from_state(m,anchor,start,states[3]); lmloss=F.cross_entropy(logits4,target); loss=lmloss+0.15*auxloss
        opt.zero_grad(set_to_none=True); loss.backward(); gn=torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
        if step==1 or step%10==0 or step==steps:
            item={'step':step,'loss':float(loss.detach()),'lm':float(lmloss.detach()),'aux':float(auxloss.detach()),'grad_norm':float(gn),'read_gate':float(m.sidecar.read_gate.detach())}
            logs.append(item); p('TRAIN '+json.dumps(item))
    return logs,time.time()-t0,aux

def save_sidecar_fp16(m,path):
    sd={k:v.detach().cpu().half() for k,v in m.sidecar.state_dict().items()}; torch.save({'version':'Mini-200 V0.5.3','source':MODEL_ID,'sidecar_state_dict':sd,'config':m.cfg.__dict__},path)

def main():
    t_all=time.time(); p(f'Loading {MODEL_ID}')
    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    hf=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float32,low_cpu_mem_usage=True).eval()
    mini=Mini200V053(Mini200V053Config()).float().eval(); copied=transplant_from_hf(hf,mini)
    p(f'Copied tensors: {copied}; mini params={sum(x.numel() for x in mini.parameters())}')
    par=parity(tok,hf,mini); p('PARITY '+json.dumps(par)); del hf; gc.collect()
    p('Loading GSM8K'); gsm=load_dataset('openai/gsm8k','main'); train_rows=list(gsm['train']); test_rows=list(gsm['test'])
    baseline_nll=gsm_gold_nll(mini,tok,test_rows,limit=GSM_NLL_N); baseline_exact=gsm_exact(mini,tok,test_rows,depths=(1,),limit=GSM_GEN_N); baseline_mmlu=mmlu_eval(mini,tok,limit=MMLU_N,depths=(1,))
    p('BASE_GSM_NLL '+json.dumps(baseline_nll)); p('BASE_GSM_EXACT '+json.dumps({k:{'accuracy':v['accuracy']} for k,v in baseline_exact.items()})); p('BASE_MMLU '+json.dumps({k:v for k,v in baseline_mmlu.items() if k!='rows'}))
    logs,secs,aux=train_sidecar(mini,tok,train_rows,STEPS); mini.eval(); aux.eval()
    post_nll=gsm_gold_nll(mini,tok,test_rows,limit=GSM_NLL_N); post_exact=gsm_exact(mini,tok,test_rows,limit=GSM_GEN_N); post_mmlu=mmlu_eval(mini,tok,limit=MMLU_N)
    p('POST_GSM_NLL '+json.dumps(post_nll)); p('POST_GSM_EXACT '+json.dumps({k:{'accuracy':v['accuracy']} for k,v in post_exact.items()})); p('POST_MMLU '+json.dumps({k:v for k,v in post_mmlu.items() if k!='rows'}))
    save_sidecar_fp16(mini,OUT/'sidecar_fp16.pt')
    result={'version':'Mini-200 V0.5.3 GitHub Actions actual SmolLM2 experiment','seed':SEED,'model_id':MODEL_ID,'parameters':sum(x.numel() for x in mini.parameters()),'training':{'steps':STEPS,'lr':LR,'seconds':secs,'log':logs,'objective':'GSM8K trajectory aux at R1..R4 + final-answer LM loss only at R4; R8 unseen during training'},'parity':par,'baseline':{'gsm_nll':baseline_nll,'gsm_exact':baseline_exact,'mmlu':baseline_mmlu},'post':{'gsm_nll':post_nll,'gsm_exact':post_exact,'mmlu':post_mmlu},'published_gates':{'MobileLLM_R1_5_140M_GSM8K_0shot':8.3,'SmolLM2_135M_base_MMLU_cloze':31.5},'validity':'GSM exact and MMLU are small real test slices, NOT full leaderboard scores. Full benchmark only after structural signal is positive.','total_seconds':time.time()-t_all}
    (OUT/'results.json').write_text(json.dumps(result,indent=2)); p('FINAL_SUMMARY '+json.dumps({'params':result['parameters'],'baseline_gsm_exact':{k:v['accuracy'] for k,v in baseline_exact.items()},'post_gsm_exact':{k:v['accuracy'] for k,v in post_exact.items()},'baseline_mmlu':baseline_mmlu.get('accuracy'),'post_mmlu':post_mmlu.get('accuracy'),'post_gsm_nll':post_nll,'seconds':result['total_seconds']}))
if __name__=='__main__': main()
