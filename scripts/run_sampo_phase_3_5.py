"""Private Phase 3.5 fixed reranking selection; never runs MAS."""
from __future__ import annotations
import asyncio,csv,json,random,re,time
from pathlib import Path
from litellm import acompletion
from sampo_baselines import bm25_token_ranked,tfidf_char_ngrams_ranked,tfidf_char_word_hybrid_ranked,tfidf_construction_token_ranked,tfidf_word_ranked
from sampo_evaluation import evaluate_predictions
ROOT=Path(__file__).resolve().parents[1]; PUBLIC=ROOT/'artifacts/sampo_benchmark'; GT=ROOT/'artifacts/sampo_audit/private_ground_truth.csv'; OUT=ROOT/'artifacts/sampo_phase_3_5'; MODEL='openai/gpt-5.6-luna'; SEED=2025
METHODS={'char':tfidf_char_ngrams_ranked,'word':tfidf_word_ranked,'hybrid':tfidf_char_word_hybrid_ranked,'bm25':bm25_token_ranked,'construction':tfidf_construction_token_ranked}
def read(p):
 with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def rank_all(rows,labels):return {n:f([x['raw_work_name'] for x in rows],labels,50) for n,f in METHODS.items()}
def fused(rs,i,k):
 score={}
 for r in rs.values():
  for rank,(label,_) in enumerate(r[i][:max(2,k//2)],1):score[label]=score.get(label,0)+1/(60+rank)
 return [(x,s) for x,s in sorted(score.items(),key=lambda z:(-z[1],z[0]))[:k]]
def raw(rs,i):
 out=[]
 for name,r in rs.items():
  for rank,(label,score) in enumerate(r[i][:5],1):
   found=next((x for x in out if x['label']==label),None)
   if found:found['provenance'].append({'retriever':name,'rank':rank,'score':round(score,7)})
   else:out.append({'label':label,'provenance':[{'retriever':name,'rank':rank,'score':round(score,7)}]})
 return out
def prompt(raw,candidates,prov=False):
 items=[{'candidate_index':i,'label':x['label'] if prov else x[0],'lexical_rank':i,'lexical_score':round((x['provenance'][0]['score'] if prov else x[1]),7),**({'retrieval_provenance':x['provenance']} if prov else {})} for i,x in enumerate(candidates,1)]
 return 'Choose only the best supplied label. Return JSON only: {"top_1_index": integer}.\nraw_work_name: '+json.dumps(raw,ensure_ascii=False)+'\ncandidates: '+json.dumps(items,ensure_ascii=False)
async def llm(raw,c,sem,prov=False):
 async with sem:r=await acompletion(model=MODEL,messages=[{'role':'user','content':prompt(raw,c,prov)}],temperature=0,response_format={'type':'json_object'})
 ix=int(json.loads(re.search(r'\{.*\}',r.choices[0].message.content or '',re.S).group())['top_1_index']);u=getattr(r,'usage',None);return (c[ix-1]['label'] if prov else c[ix-1][0]),int(getattr(u,'prompt_tokens',0)or 0),int(getattr(u,'completion_tokens',0)or 0)
async def evaluate(rows,rs,gt,labels,kind,indices):
 cs=[raw(rs,i) if kind=='raw' else fused(rs,i,5) for i in range(len(rows))];lex=[c[0]['label'] if kind=='raw' else c[0][0] for c in cs];sem=asyncio.Semaphore(8);start=time.perf_counter();got=await asyncio.gather(*[llm(rows[i]['raw_work_name'],cs[i],sem,kind=='raw') for i in indices]);chosen=dict(zip(indices,got));pred=[chosen[i][0] if i in chosen else lex[i] for i in range(len(rows))];targets=[gt[x['example_id']]['target_granular_name'] for x in rows];oracle=[i for i,c in enumerate(cs) if targets[i] in ([x['label'] for x in c] if kind=='raw' else [x for x,_ in c])];res=sum(l!=t and p==t for l,p,t in zip(lex,pred,targets));harm=sum(l==t and p!=t for l,p,t in zip(lex,pred,targets));top3=[{'example_id':x['example_id'],'top_1':p,'top_2':next(y for y in lex if y!=p),'top_3':labels[0] if labels[0] not in {p,next(y for y in lex if y!=p)} else labels[1]} for x,p in zip(rows,pred)]
 m=evaluate_predictions([gt[x['example_id']] for x in rows],top3,labels);return {'retrieval_recall':len(oracle)/len(rows),'lexical_top_1':sum(l==t for l,t in zip(lex,targets))/len(rows),'final_top_1':m['top_1_accuracy'],'top_3':m['top_3_accuracy'],'macro_f1_observed_labels':m['macro_f1_observed_labels'],'oracle_top_1':sum(pred[i]==targets[i] for i in oracle)/len(oracle),'calls':len(indices),'rescues':res,'harms':harm,'net_gain':res-harm,'prompt_tokens':sum(x[1] for x in got),'completion_tokens':sum(x[2] for x in got),'tokens_per_net_correction':sum(x[1]+x[2] for x in got)/(res-harm) if res>harm else None,'runtime_seconds':time.perf_counter()-start}
async def main():
 OUT.mkdir(parents=True,exist_ok=True);pilot=read(PUBLIC/'pilot_inputs.csv');labels=[x['target_label'] for x in read(PUBLIC/'allowed_target_labels.csv')];grows=read(GT);gt={str(i):{'example_id':str(i),'target_granular_name':x['target_granular_name']} for i,x in enumerate(grows,1)};sub=random.Random(SEED).sample(pilot,150);rs=rank_all(sub,labels);dis={i for i in range(150) if len({r[i][0][0] for r in rs.values()})>1};r={'k5_disagreement':await evaluate(sub,rs,gt,labels,'fused',dis),'raw_union_all':await evaluate(sub,rs,gt,labels,'raw',set(range(150)))}; prior=json.loads((ROOT/'artifacts/sampo_phase_3/rerank_results.json').read_text());r['k5_all']=prior['candidate_tradeoff']['k_5'];r['k10_disagreement']=prior['policies']['retriever_disagreement'];r['k10_all']=prior['policies']['all'];winner=max(('k5_disagreement','raw_union_all'),key=lambda n:(r[n]['final_top_1'], -r[n]['calls'], -(r[n]['prompt_tokens']+r[n]['completion_tokens'])));r['selected']=winner;rsfull=rank_all(pilot,labels);indices={i for i in range(1000) if len({x[i][0][0] for x in rsfull.values()})>1} if winner=='k5_disagreement' else set(range(1000));r['pilot_selected']=await evaluate(pilot,rsfull,gt,labels,'fused' if winner=='k5_disagreement' else 'raw',indices);(OUT/'results.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');(OUT/'report.md').write_text('# SAMPO Phase 3.5\n\nSelected `'+winner+'`; see `results.json` for the fixed-subset and one-shot pilot metrics.\n')
if __name__=='__main__':asyncio.run(main())
