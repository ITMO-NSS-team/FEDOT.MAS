"""Behaviorally qualify the five already-generated bounded-batch configs."""
from __future__ import annotations
import asyncio,csv,json,os,threading,time
from pathlib import Path
from typing import Any
from fedotmas import MAS
from fedotmas.mas.models import MASConfig
from google.adk.plugins import BasePlugin
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.events import Event
from google.adk.runners import InvocationContext
ROOT=Path(__file__).resolve().parents[1]; B=Path(os.environ['PHASE5_BATCH']).resolve() if os.environ.get('PHASE5_BATCH') else ROOT/'artifacts/sampo_phase_5/structural_review/batch_7aea72723bfd461585a8d6cd67857954'; OUT=ROOT/'artifacts/sampo_phase_5/behavioral_qualification'/B.name
class Trace(BasePlugin):
 def __init__(self, path: Path | None = None): super().__init__(name='phase5_behavior_trace'); self.calls=[]; self.tools=[]; self.agents=[]; self.fail=[]; self.path=path
 def flush(self):
  if self.path: self.path.write_text(json.dumps({'agents':self.agents,'model_calls':self.calls,'tool_calls':self.tools,'failures':self.fail})+'\n')
 async def before_agent_callback(self,*,agent,callback_context): self.agents.append(agent.name); self.flush()
 async def after_model_callback(self,*,callback_context,llm_response:LlmResponse):
  u=llm_response.usage_metadata; p=(u.prompt_token_count if u else 0) or 0; c=(u.candidates_token_count if u else 0) or 0; self.calls.append({'agent':callback_context.agent_name,'prompt_tokens':p,'completion_tokens':c})
  self.flush()
  if p>64000:self.fail.append('prompt_over_64k'); self.flush(); raise RuntimeError('prompt_over_64k')
  if len(self.calls)>30:self.fail.append('model_call_limit'); self.flush(); raise RuntimeError('model_call_limit')
 async def on_event_callback(self,*,invocation_context:InvocationContext,event:Event):
  if not event.partial:
   for x in event.get_function_calls():
    args=x.args or {}; self.tools.append({'agent':event.author,'tool':x.name,'ids':args.get('example_ids') or [d.get('example_id') for d in args.get('decisions',[])], 'decisions':[{'example_id':d.get('example_id'),'candidate_indices':d.get('candidate_indices')} for d in args.get('decisions',[])], 'artifact_id':args.get('artifact_id'), 'retrieval':({'offset':args.get('offset'),'limit':args.get('limit'),'methods':args.get('methods'),'k':args.get('k'),'fusion':args.get('fusion')} if x.name=='prepare_candidate_batch' else None)})
    self.flush()
    if len(self.tools)>60:self.fail.append('tool_call_limit'); self.flush(); raise RuntimeError('tool_call_limit')
def stored(run):
 p=ROOT/f'artifacts/sampo_benchmark/mas_runs/{run}.jsonl'
 return {json.loads(x)['example_id'] for x in p.open() if x.strip()} if p.exists() else set()
def _completion_guard(run: str, assigned: list[str], sentinel: Path) -> None:
 while True:
  if set(assigned) <= stored(run):
   sentinel.write_text(json.dumps({'reason':'harness_batch_complete','stored_ids':sorted(stored(run))})+'\n')
   os._exit(0)
  time.sleep(0.1)
async def one(i,assigned,offset=0):
 d=B/f'config_{i:02d}'; meta=json.loads((d/'metadata.json').read_text()); run=meta['run_id']; trace=Trace(Path(os.environ['PHASE5_TRACE_PATH']) if os.environ.get('PHASE5_TRACE_PATH') else None); before=stored(run); start=time.perf_counter(); reason='normal'
 if os.environ.get('PHASE5_ENFORCE_COMPLETE'):
  sentinel=Path(os.environ['PHASE5_COMPLETION_SENTINEL']); threading.Thread(target=_completion_guard,args=(run,assigned,sentinel),daemon=True).start()
 task=(d/'task.txt').read_text()+f'\nHARNESS ASSIGNMENT: process exactly offset {offset} and IDs {assigned}. Do not process any other IDs; do not finalize the pilot.'
 try: await MAS(mcp_servers=['sampo-benchmark','sandbox-light'],plugins=[trace]).build_and_run(MASConfig.model_validate_json((d/'config.json').read_text()),task,timeout=300)
 except Exception as e: reason=f'{type(e).__name__}: {e}'
 elapsed=time.perf_counter()-start; after=stored(run); added=after-before; failed=list(trace.fail); outside=added-set(assigned)
 if outside: failed.append('outside_batch_write')
 if set(assigned)-after: failed.append('incomplete_coverage')
 saves=[t for t in trace.tools if t['tool'] in {'save_candidate_predictions','save_review_decisions'}]; seen=set()
 for t in saves:
  ids=set(t['ids'] or [])
  if ids & seen: failed.append('duplicate_persistence_attempt')
  seen.update(ids)
 preps=[t for t in trace.tools if t['tool']=='prepare_candidate_batch']
 if len(preps)>1: failed.append('repeated_completed_unit')
 if elapsed>=299.5: reason='wall_clock_timeout'; failed.append('runtime_limit')
 if reason!='normal': failed.append('non_normal_termination')
 static=json.loads((B/'structural_review.json').read_text())['configs'][f'config_{i:02d}']['pass']
 return {'config':str(d.relative_to(ROOT)),'static_sanity':static,'behavioral_pass':static and not failed,'failed_criteria':sorted(set(failed)),'trace':{'agents':trace.agents,'model_calls':trace.calls,'tool_calls':trace.tools,'max_prompt_tokens':max([x['prompt_tokens'] for x in trace.calls],default=0),'artifact_ids':sorted({x['artifact_id'] for x in trace.tools if x['artifact_id']}),'final_stored_ids':sorted(after),'runtime_seconds':elapsed,'termination_reason':reason}}
async def main():
 OUT.mkdir(parents=True,exist_ok=True)
 with (ROOT/'artifacts/sampo_benchmark/pilot_inputs.csv').open(encoding='utf-8',newline='') as f: default_ids=[r['example_id'] for r in list(csv.DictReader(f))[:5]]
 ids=json.loads(os.environ['PHASE5_ASSIGNED_IDS']) if os.environ.get('PHASE5_ASSIGNED_IDS') else default_ids
 offset=int(os.environ.get('PHASE5_OFFSET','0'))
 indexes=[int(os.environ['PHASE5_CONFIG'])] if os.environ.get('PHASE5_CONFIG') else range(1,6)
 results=[]
 for i in indexes:
  result=await one(i,ids,offset); results.append(result)
  result_path=Path(os.environ['PHASE5_RESULT_PATH']) if os.environ.get('PHASE5_RESULT_PATH') else OUT/f'config_{i:02d}.json'
  result_path.write_text(json.dumps(result,indent=2)+'\n')
 if os.environ.get('PHASE5_CONFIG'): return
 selected=next((x['config'] for x in results if x['behavioral_pass']),None)
 report={'batch_id':B.name,'assigned_ids':ids,'private_ground_truth_used':False,'accuracy_used':False,'configs':results,'selected_config':selected}
 (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 (ROOT/'artifacts/sampo_phase_5/decision_report.md').write_text('# Phase 5 decision report\n\nSelection used behavioral qualification only; no private ground truth or accuracy.\n\nSelected: '+(selected or 'none')+'\n')
if __name__=='__main__': asyncio.run(main())
