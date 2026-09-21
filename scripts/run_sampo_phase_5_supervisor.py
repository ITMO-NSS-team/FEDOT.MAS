"""Process-isolated supervisor for bounded Phase 5 behavioral qualification."""
from __future__ import annotations
import hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
B=Path(os.environ['PHASE5_BATCH']).resolve() if os.environ.get('PHASE5_BATCH') else ROOT/'artifacts/sampo_phase_5/structural_review/batch_7aea72723bfd461585a8d6cd67857954'
OUT=ROOT/'artifacts/sampo_phase_5/behavioral_qualification'/B.name
def malformed_tool_calls(raw):
 failures=[]
 for call in raw.get('tool_calls',[]):
  if call.get('tool')=='save_review_decisions':
   for decision in call.get('decisions') or []:
    indices=decision.get('candidate_indices')
    if not isinstance(indices,list) or len(indices)!=3 or len(set(indices))!=3 or not all(isinstance(index,int) for index in indices): failures.append('malformed_review_decision')
 return failures
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 ids=[row.split(',')[0] for row in (ROOT/'artifacts/sampo_benchmark/pilot_inputs.csv').read_text(encoding='utf-8').splitlines()[1:6]]
 manifest={'batch_id':B.name,'assigned_ids':ids,'limits':{'wall_clock_seconds':300,'model_calls':30,'tool_calls':60,'prompt_tokens':64000},'config_sha256':{f'config_{i:02d}':hashlib.sha256((B/f'config_{i:02d}/config.json').read_bytes()).hexdigest() for i in range(1,6)},'mcp_server_sha256':hashlib.sha256((ROOT/'mcp-servers/sampo-benchmark/src/mcp_sampo_benchmark/server.py').read_bytes()).hexdigest(),'private_ground_truth_used':False}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 results=[]
 for index in range(1,6):
  path=OUT/f'config_{index:02d}.json'; trace_path=OUT/f'config_{index:02d}.live_trace.json'; sentinel=OUT/f'config_{index:02d}.complete.json'
  if path.exists():
   result=json.loads(path.read_text())
   raw=json.loads(trace_path.read_text()) if trace_path.exists() else {}
   malformed=malformed_tool_calls(raw)
   if malformed:
    result['behavioral_pass']=False; result['failed_criteria']=sorted(set(result.get('failed_criteria',[])+malformed))
    path.write_text(json.dumps(result,indent=2)+'\n')
   results.append(result)
   continue
  static=json.loads((B/'structural_review.json').read_text())['configs'][f'config_{index:02d}']['pass']
  if not static:
   result={'config':str((B/f'config_{index:02d}').relative_to(ROOT)),'static_sanity':False,'behavioral_pass':False,'failed_criteria':['static_sanity'],'trace':{'agents':[],'model_calls':[],'tool_calls':[],'max_prompt_tokens':0,'artifact_ids':[],'final_stored_ids':[],'runtime_seconds':0,'termination_reason':'static_sanity_failed'}}
   path.write_text(json.dumps(result,indent=2)+'\n'); results.append(result)
   continue
  run=json.loads((B/f'config_{index:02d}/metadata.json').read_text())['run_id']; prediction=ROOT/f'artifacts/sampo_benchmark/mas_runs/{run}.jsonl'
  if sentinel.exists() and trace_path.exists():
   raw=json.loads(trace_path.read_text()); final={json.loads(x)['example_id'] for x in prediction.open() if x.strip()} if prediction.exists() else set(); failures=list(raw.get('failures',[]))+malformed_tool_calls(raw); candidate_ids=set(); review_decisions={}
   for call in raw.get('tool_calls',[]):
    if call['tool']=='save_candidate_predictions':
     current=set(call.get('ids') or [])
     if current & candidate_ids: failures.append('duplicate_persistence_attempt')
     candidate_ids.update(current)
    elif call['tool']=='save_review_decisions':
     for decision in call.get('decisions') or []:
      example_id=decision.get('example_id'); indices=tuple(decision.get('candidate_indices') or [])
      if example_id in review_decisions and review_decisions[example_id] == indices: failures.append('duplicate_persistence_attempt')
      review_decisions[example_id]=indices
   if not set(ids)<=final: failures.append('incomplete_coverage')
   result={'config':str((B/f'config_{index:02d}').relative_to(ROOT)),'static_sanity':static,'behavioral_pass':static and not failures,'failed_criteria':sorted(set(failures)),'trace':{**raw,'max_prompt_tokens':max([x['prompt_tokens'] for x in raw.get('model_calls',[])],default=0),'artifact_ids':sorted({x['artifact_id'] for x in raw.get('tool_calls',[]) if x.get('artifact_id')}),'final_stored_ids':sorted(final),'runtime_seconds':None,'termination_reason':'recovered_harness_batch_complete'}}
   path.write_text(json.dumps(result,indent=2)+'\n'); results.append(result)
   continue
  trace_path.unlink(missing_ok=True); sentinel.unlink(missing_ok=True); env=dict(os.environ,PHASE5_CONFIG=str(index),PHASE5_TRACE_PATH=str(trace_path),PHASE5_ENFORCE_COMPLETE='1',PHASE5_COMPLETION_SENTINEL=str(sentinel)); start=time.monotonic(); completed=False
  proc=subprocess.Popen([sys.executable,str(ROOT/'scripts/run_sampo_phase_5_qualification.py')],cwd=ROOT,env=env,start_new_session=True)
  while proc.poll() is None and time.monotonic()-start < 300:
   stored={json.loads(x)['example_id'] for x in prediction.open() if x.strip()} if prediction.exists() else set()
   if sentinel.exists() or set(ids) <= stored:
    completed=True; time.sleep(0.2)
    try: os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError: pass
    break
   time.sleep(1)
  if proc.poll() is None:
   try: os.killpg(proc.pid,signal.SIGKILL)
   except ProcessLookupError: pass
  proc.wait(); elapsed=time.monotonic()-start
  completed = completed or sentinel.exists()
  if not path.exists():
   raw=json.loads(trace_path.read_text()) if trace_path.exists() else {'agents':[],'model_calls':[],'tool_calls':[],'failures':[]}; final={json.loads(x)['example_id'] for x in prediction.open() if x.strip()} if prediction.exists() else set(); failures=list(raw.get('failures',[])); outside=final-set(ids)
   failures.extend(malformed_tool_calls(raw))
   if outside: failures.append('outside_batch_write')
   if not set(ids)<=final: failures.append('incomplete_coverage')
   candidate_ids=set(); review_decisions={}
   for call in raw.get('tool_calls',[]):
    if call['tool']=='save_candidate_predictions':
     current=set(call.get('ids') or [])
     if current & candidate_ids: failures.append('duplicate_persistence_attempt')
     candidate_ids.update(current)
    elif call['tool']=='save_review_decisions':
     for decision in call.get('decisions') or []:
      example_id=decision.get('example_id'); indices=tuple(decision.get('candidate_indices') or [])
      if example_id in review_decisions and review_decisions[example_id] == indices: failures.append('duplicate_persistence_attempt')
      review_decisions[example_id]=indices
   if not completed: failures.extend(['runtime_limit','non_normal_termination'])
   result={'config':str((B/f'config_{index:02d}').relative_to(ROOT)),'static_sanity':static,'behavioral_pass':static and completed and not failures,'failed_criteria':sorted(set(failures)),'trace':{**raw,'max_prompt_tokens':max([x['prompt_tokens'] for x in raw.get('model_calls',[])],default=0),'artifact_ids':sorted({x['artifact_id'] for x in raw.get('tool_calls',[]) if x.get('artifact_id')}),'final_stored_ids':sorted(final),'runtime_seconds':elapsed,'termination_reason':'harness_batch_complete' if completed else 'external_wall_clock_timeout'}}
   path.write_text(json.dumps(result,indent=2)+'\n')
  results.append(json.loads(path.read_text()))
 report={**manifest,'accuracy_used':False,'configs':results,'selected_config':next((r['config'] for r in results if r['behavioral_pass']),None)}
 (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 (ROOT/'artifacts/sampo_phase_5/decision_report.md').write_text('# Phase 5 decision report\n\nSelection used behavioral qualification only; no private ground truth or accuracy.\n\nSelected: '+(report['selected_config'] or 'none')+'\n')
if __name__=='__main__': main()
