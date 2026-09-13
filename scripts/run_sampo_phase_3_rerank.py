"""Private, fixed-subset Phase 3 selective reranking diagnostic (no MAS)."""
from __future__ import annotations

import asyncio
import csv
import json
import random
import re
from pathlib import Path

from litellm import acompletion
from sampo_baselines import (
    bm25_token_ranked,
    tfidf_char_ngrams_ranked,
    tfidf_char_word_hybrid_ranked,
    tfidf_construction_token_ranked,
    tfidf_word_ranked,
)

ROOT=Path(__file__).resolve().parents[1]; PUBLIC=ROOT/'artifacts/sampo_benchmark'; GT=ROOT/'artifacts/sampo_audit/private_ground_truth.csv'; OUT=ROOT/'artifacts/sampo_phase_3'; MODEL='openai/gpt-5.6-luna'; SEED=2025; SIZE=150
METHODS={'char':tfidf_char_ngrams_ranked,'word':tfidf_word_ranked,'hybrid':tfidf_char_word_hybrid_ranked,'bm25':bm25_token_ranked,'construction':tfidf_construction_token_ranked}
def read(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def candidates(rankings,i,k):
    depth={5:2,10:5,20:10}[k]; seen=[]
    for r in rankings.values():
        for label,score in r[i][:depth]:
            if label not in seen:seen.append(label)
    fused=[]
    for label in seen:
        score=sum(1/(60+next(j for j,(x,_) in enumerate(r[i],1) if x==label)) for r in rankings.values() if any(x==label for x,_ in r[i]))
        fused.append((label,score))
    ordered=[label for label,_ in sorted(fused,key=lambda x:(-x[1],x[0]))]
    return [(x,next(score for r in rankings.values() for label,score in r[i] if label==x)) for x in ordered[:k]]
def pmt(raw,c):return 'Choose only the best supplied label. Return JSON only: {"top_1_index": integer}.\nraw_work_name: '+json.dumps(raw,ensure_ascii=False)+'\ncandidates: '+json.dumps([{'candidate_index':j,'label':x,'lexical_rank':j,'lexical_score':round(s,8)} for j,(x,s) in enumerate(c,1)],ensure_ascii=False)
async def call(raw,c,sem):
 async with sem:
  r=await acompletion(model=MODEL,messages=[{'role':'user','content':pmt(raw,c)}],temperature=0,response_format={'type':'json_object'})
 txt=r.choices[0].message.content or ''; j=json.loads(re.search(r'\{.*\}',txt,re.DOTALL).group()); ix=int(j['top_1_index']); u=getattr(r,'usage',None)
 return c[ix-1][0],int(getattr(u,'prompt_tokens',0)or 0),int(getattr(u,'completion_tokens',0)or 0)
async def run():
    pilot=read(PUBLIC/'pilot_inputs.csv'); labels=[x['target_label'] for x in read(PUBLIC/'allowed_target_labels.csv')]; gt={str(i):x['target_granular_name'] for i,x in enumerate(read(GT),1)}; subset=random.Random(SEED).sample(pilot,SIZE); allrank={n:f([x['raw_work_name'] for x in subset],labels,50) for n,f in METHODS.items()}
    margins=[r[0][1]-r[1][1] for r in allrank['hybrid']]; q25=sorted(margins)[SIZE//4]; q50=sorted(margins)[SIZE//2]; disagreement={i for i in range(SIZE) if len({r[i][0][0] for r in allrank.values()})>1}; policies={'all':set(range(SIZE)),'bottom_25_margin':{i for i,m in enumerate(margins) if m<=q25},'bottom_50_margin':{i for i,m in enumerate(margins) if m<=q50},'retriever_disagreement':disagreement}; out={'policies':{},'candidate_tradeoff':{},'heuristics':{'margin_q25':q25,'margin_q50':q50,'disagreement_examples':len(disagreement)}}
    async def evaluate(name,indices,k=10):
        cs=[candidates(allrank,i,k) for i in range(SIZE)]; sem=asyncio.Semaphore(8); got=await asyncio.gather(*[call(subset[i]['raw_work_name'],cs[i],sem) for i in indices]); selected=dict(zip(indices,got)); lex=[c[0][0] for c in cs]; pred=[selected[i][0] if i in selected else lex[i] for i in range(SIZE)]; target=[gt[x['example_id']] for x in subset]; rescue=sum(l!=t and p==t for l,p,t in zip(lex,pred,target)); harm=sum(l==t and p!=t for l,p,t in zip(lex,pred,target)); return {'overall_top_1':sum(p==t for p,t in zip(pred,target))/SIZE,'calls':len(indices),'prompt_tokens':sum(x[1] for x in got),'completion_tokens':sum(x[2] for x in got),'rescued':rescue,'harmed':harm,'net_gain':rescue-harm,'gain_per_100_calls':(rescue-harm)*100/len(indices) if indices else 0,'tokens_per_net_corrected':(sum(x[1]+x[2] for x in got)/(rescue-harm)) if rescue>harm else None,'retrieval_recall':sum(t in [x for x,_ in c] for t,c in zip(target,cs))/SIZE,'oracle_top_1':sum(pred[i]==target[i] for i,c in enumerate(cs) if target[i] in [x for x,_ in c])/sum(target[i] in [x for x,_ in c] for i,c in enumerate(cs))}
    for n,ix in policies.items():out['policies'][n]=await evaluate(n,ix)
    for k in (5,10,20):out['candidate_tradeoff'][f'k_{k}']=await evaluate(f'k{k}',set(range(SIZE)),k)
    (OUT/'rerank_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':asyncio.run(run())
