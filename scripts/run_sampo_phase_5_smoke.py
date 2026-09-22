"""Process-isolated, harness-owned 3x20 smoke test for selected Phase 5 config."""
from __future__ import annotations
import csv,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from sampo_phase_5_policy import evaluate_policy_conformance
B=Path(os.environ['PHASE5_BATCH']).resolve() if os.environ.get('PHASE5_BATCH') else ROOT/'artifacts/sampo_phase_5/structural_review/batch_7aea72723bfd461585a8d6cd67857954'
CONFIG=B/f"config_{int(os.environ.get('PHASE5_CONFIG','1')):02d}"; OUT=ROOT/'artifacts/sampo_phase_5'/f"smoke_{B.name}_{CONFIG.name}{os.environ.get('PHASE5_SMOKE_SUFFIX','')}"
def rows():
 with (ROOT/'artifacts/sampo_benchmark/pilot_inputs.csv').open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def stored(path): return {json.loads(x)['example_id'] for x in path.open() if x.strip()} if path.exists() else set()
def duplicate_persistence(raw, allowed_ids):
 candidate_ids=set(); review_decisions={}
 for call in raw.get('tool_calls',[]):
  if call.get('tool') == 'save_candidate_predictions':
   ids=set(call.get('ids') or [])
   if not ids <= allowed_ids: continue
   if ids & candidate_ids: return True
   candidate_ids.update(ids)
  elif call.get('tool') == 'save_review_decisions':
   for decision in call.get('decisions') or []:
    example_id=decision.get('example_id'); indices=tuple(decision.get('candidate_indices') or [])
    if example_id in review_decisions and review_decisions[example_id] == indices: return True
    review_decisions[example_id]=indices
 return False
def repeated_prepare(raw):
 durable_write=False
 for call in raw.get('tool_calls',[]):
  if call.get('tool') in {'save_candidate_predictions','save_review_decisions'}: durable_write=True
  if durable_write and call.get('tool')=='prepare_candidate_batch': return True
 return False
def malformed_tool_call(raw, allowed_ids):
 accepted={'bm25_token','char_tfidf','char_word_fusion','construction_token_tfidf','word_tfidf','bm25','bm25_token_ranked','lexical','lexical_bm25','tfidf_char','tfidf_char_ngrams','char_ngram_tfidf','tfidf_char_word_hybrid','char_word_hybrid','tfidf_construction_token','construction_tfidf','tfidf_word','tfidf_word_ngrams'}
 for call in raw.get('tool_calls',[]):
  if call.get('tool')=='prepare_candidate_batch' and any(method not in accepted for method in (call.get('retrieval') or {}).get('methods') or []): return True
  if call.get('tool')=='save_candidate_predictions' and not set(call.get('ids') or []) <= allowed_ids: return True
  if call.get('tool')=='save_review_decisions' and any(not isinstance(decision.get('candidate_indices'),list) or len(decision['candidate_indices'])!=3 or len(set(decision['candidate_indices']))!=3 for decision in call.get('decisions') or []): return True
  if call.get('tool')=='get_candidate_evidence' and (len(call.get('ids') or [])>4 or (call.get('evidence') or {}).get('selection')!='diverse_round_robin' or not 5 <= (call.get('evidence') or {}).get('candidate_limit',0) <= 10): return True
 return False
def main():
 if OUT.exists(): raise RuntimeError(f'Refusing to overwrite {OUT}')
 OUT.mkdir(parents=True); meta=json.loads((CONFIG/'metadata.json').read_text()); run=os.environ.get('PHASE5_RUN_ID',meta['run_id']); prediction=ROOT/f'artifacts/sampo_benchmark/mas_runs/{run}.jsonl'
 if prediction.exists() or prediction.with_suffix('.staged.jsonl').exists(): raise RuntimeError('Smoke run_id is not clean')
 allrows=rows(); report={'config':str(CONFIG.relative_to(ROOT)),'run_id':run,'private_ground_truth_used':False,'fresh_sessions':True,'batches':[]}
 for offset in (0,20,40):
  assigned=[r['example_id'] for r in allrows[offset:offset+20]]; before=stored(prediction); trace=OUT/f'batch_{offset:04d}_trace.json'; sentinel=OUT/f'batch_{offset:04d}_complete.json'; result=OUT/f'batch_{offset:04d}_result.json'; start=time.monotonic()
  env=dict(os.environ,PHASE5_CONFIG=CONFIG.name.removeprefix('config_'),PHASE5_RUN_ID=run,PHASE5_ASSIGNED_IDS=json.dumps(assigned),PHASE5_OFFSET=str(offset),PHASE5_ENFORCE_COMPLETE='1',PHASE5_COMPLETION_SENTINEL=str(sentinel),PHASE5_TRACE_PATH=str(trace),PHASE5_RESULT_PATH=str(result),PHASE5_POLICY_SUFFIX=os.environ.get('PHASE5_POLICY_SUFFIX',''))
  proc=subprocess.Popen([sys.executable,str(ROOT/'scripts/run_sampo_phase_5_qualification.py')],cwd=ROOT,env=env,start_new_session=True)
  while proc.poll() is None and time.monotonic()-start<300:
   if set(assigned)<=stored(prediction):
    time.sleep(.2)
    try: os.killpg(proc.pid,signal.SIGTERM)
    except (ProcessLookupError, PermissionError): pass
    break
   time.sleep(.1)
  if proc.poll() is None:
   try: os.killpg(proc.pid,signal.SIGKILL)
   except (ProcessLookupError, PermissionError): pass
  proc.wait(); final=stored(prediction); raw=json.loads(trace.read_text()) if trace.exists() else {'model_calls':[],'tool_calls':[],'failures':[]}; outside=(final-before)-set(assigned)
  policy=evaluate_policy_conformance(raw,assigned)
  item={'offset':offset,'assigned_ids':assigned,'runtime_seconds':time.monotonic()-start,'stored_ids':sorted(final & set(assigned)),'outside_ids':sorted(outside),'max_prompt_tokens':max([x['prompt_tokens'] for x in raw['model_calls']],default=0),'model_calls':len(raw['model_calls']),'tool_calls':len(raw['tool_calls']),'duplicate_persistence':duplicate_persistence(raw,set(assigned)),'repeated_prepare':repeated_prepare(raw),'malformed_tool_call':malformed_tool_call(raw,set(assigned)),'policy_conformance':policy,'complete':set(assigned)<=final,'sentinel':sentinel.exists(),'failures':raw['failures']}
  report['batches'].append(item)
  if not item['complete'] or item['outside_ids'] or item['duplicate_persistence'] or item['repeated_prepare'] or item['malformed_tool_call'] or not policy['pass'] or item['max_prompt_tokens']>64000 or item['model_calls']>30 or item['tool_calls']>60 or item['failures']:
   report['status']='failed'; report['failure_reason']=f'batch {offset} failed'; (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n'); raise RuntimeError(f'batch {offset} failed')
 report['status']='passed'; (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__': main()
