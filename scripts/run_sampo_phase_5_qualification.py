"""Behaviorally qualify the five already-generated bounded-batch configs."""
from __future__ import annotations
import asyncio,csv,json,os,threading,time,uuid
from pathlib import Path
from typing import Any
from fedotmas import MAS
from fedotmas.mas.models import MASConfig
from google.adk.plugins import BasePlugin
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.events import Event
from google.adk.runners import InvocationContext
from sampo_phase_5_policy import evaluate_policy_conformance
from sampo_phase_5_trace import write_trace_atomic
ROOT=Path(__file__).resolve().parents[1]; B=Path(os.environ['PHASE5_BATCH']).resolve() if os.environ.get('PHASE5_BATCH') else ROOT/'artifacts/sampo_phase_5/structural_review/batch_7aea72723bfd461585a8d6cd67857954'; QUALIFICATION_ROOT=ROOT/'artifacts/sampo_phase_5/behavioral_qualification'/B.name; OUT=QUALIFICATION_ROOT/f"attempt_{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
# Fixed public-only mixed sample: 10477, 1825, 4507 have retriever agreement;
# 410 and 12150 have retriever disagreement in the deterministic public artifact.
QUALIFICATION_IDS=['10477','1825','410','12150','4507']
class Trace(BasePlugin):
 def __init__(self, path: Path | None = None): super().__init__(name='phase5_behavior_trace'); self.calls=[]; self.tools=[]; self.agents=[]; self.fail=[]; self.path=path; self.durable_ids=set(); self.durable_paths=set(); self.partition_ids=set(); self.evidence_seen=set(); self.terminal=False
 def flush(self):
  if self.path: write_trace_atomic(self.path, {'agents':self.agents,'model_calls':self.calls,'tool_calls':self.tools,'failures':self.fail})
 async def before_agent_callback(self,*,agent,callback_context): self.agents.append(agent.name); self.flush()
 async def after_model_callback(self,*,callback_context,llm_response:LlmResponse):
  u=llm_response.usage_metadata; p=(u.prompt_token_count if u else 0) or 0; c=(u.candidates_token_count if u else 0) or 0; self.calls.append({'agent':callback_context.agent_name,'prompt_tokens':p,'completion_tokens':c})
  self.flush()
  if p>64000:self.fail.append('prompt_over_64k'); self.flush(); raise RuntimeError('prompt_over_64k')
  if len(self.calls)>30:self.fail.append('model_call_limit'); self.flush(); raise RuntimeError('model_call_limit')
 async def on_event_callback(self,*,invocation_context:InvocationContext,event:Event):
  if not event.partial:
   for x in event.get_function_calls():
    if self.partition_ids and self.durable_ids >= self.partition_ids:
     self.fail.append('tool_after_durable_write'); self.flush(); raise RuntimeError('tool_after_durable_write')
    args=x.args or {}; ids=args.get('example_ids') or [d.get('example_id') for d in args.get('decisions',[])]
    if x.name=='partition_candidate_batch': self.partition_ids.update(ids)
    if x.name=='get_candidate_evidence':
     if self.evidence_seen.intersection(ids): self.fail.append('duplicate_evidence_call'); self.flush(); raise RuntimeError('duplicate_evidence_call')
     self.evidence_seen.update(ids)
    self.tools.append({'agent':event.author,'tool':x.name,'ids':ids, 'decisions':[{'example_id':d.get('example_id'),'candidate_indices':d.get('candidate_indices')} for d in args.get('decisions',[])], 'fill_retrieval_tail':args.get('fill_retrieval_tail',False), 'artifact_id':args.get('artifact_id'), 'retrieval':({'offset':args.get('offset'),'limit':args.get('limit'),'methods':args.get('methods'),'k':args.get('k'),'fusion':args.get('fusion')} if x.name=='prepare_candidate_batch' else None), 'evidence':({'candidate_limit':args.get('candidate_limit'),'selection':args.get('selection')} if x.name=='get_candidate_evidence' else None)})
    if x.name in {'save_review_decisions','save_candidate_predictions'}:
     self.durable_ids.update(ids); self.durable_paths.add(x.name)
   self.flush()
   if len(self.tools)>60:self.fail.append('tool_call_limit'); self.flush(); raise RuntimeError('tool_call_limit')
   if event.get_function_responses() and self.partition_ids and self.durable_ids >= self.partition_ids and self.durable_paths == {'save_review_decisions','save_candidate_predictions'}:
    self.terminal=True; invocation_context.end_invocation=True; self.flush()
def stored(run):
 p=ROOT/f'artifacts/sampo_benchmark/mas_runs/{run}.jsonl'
 if not p.exists(): return set()
 for _ in range(20):
  try: return {json.loads(x)['example_id'] for x in p.open() if x.strip()}
  except (UnicodeDecodeError,json.JSONDecodeError): time.sleep(0.05)
 raise RuntimeError(f'Prediction file remained unreadable during status check: {p}')
def _completion_guard(run: str, assigned: list[str], sentinel: Path) -> None:
 while True:
  if set(assigned) <= stored(run):
   sentinel.write_text(json.dumps({'reason':'harness_batch_complete','stored_ids':sorted(stored(run))})+'\n')
   os._exit(0)
  time.sleep(0.1)
async def one(i,assigned,offset=0):
 d=B/f'config_{i:02d}'; meta=json.loads((d/'metadata.json').read_text()); run=os.environ.get('PHASE5_RUN_ID',f"{meta['run_id']}_qual_{uuid.uuid4().hex[:12]}"); trace=Trace(Path(os.environ['PHASE5_TRACE_PATH']) if os.environ.get('PHASE5_TRACE_PATH') else None); before=stored(run); start=time.perf_counter(); reason='normal'
 if os.environ.get('PHASE5_ENFORCE_COMPLETE'):
  sentinel=Path(os.environ['PHASE5_COMPLETION_SENTINEL']); threading.Thread(target=_completion_guard,args=(run,assigned,sentinel),daemon=True).start()
 task=(d/'task.txt').read_text()+f'\nHARNESS RUN-ID OVERRIDE: use durable prediction run_id {run}; this supersedes any run_id in the saved task. HARNESS ASSIGNMENT: process exactly offset {offset} and IDs {assigned}. Do not process any other IDs; do not finalize the pilot.'+os.environ.get('PHASE5_POLICY_SUFFIX','')
 try: await MAS(mcp_servers=['sampo-benchmark','sandbox-light'],plugins=[trace]).build_and_run(MASConfig.model_validate_json((d/'config.json').read_text()),task,timeout=300)
 except Exception as e: reason=f'{type(e).__name__}: {e}'
 if trace.terminal: reason='phase5_terminal_after_durable_writes'
 elapsed=time.perf_counter()-start; after=stored(run); added=after-before; failed=list(trace.fail); outside=added-set(assigned)
 if outside: failed.append('outside_batch_write')
 if set(assigned)-after: failed.append('incomplete_coverage')
 saves=[t for t in trace.tools if t['tool'] in {'save_candidate_predictions','save_review_decisions'}]; seen=set()
 for t in saves:
  ids=set(t['ids'] or [])
  if ids & seen: failed.append('duplicate_persistence_attempt')
  seen.update(ids)
 if any(t['tool']=='stage_candidate_predictions' for t in trace.tools): failed.append('staging_not_allowed')
 # Multiple retrieval calls are allowed before persistence (for complementary
 # signals or bounded evidence). Repeated work is detected only when the same
 # ID is included in overlapping durable persistence calls above.
 if elapsed>=299.5: reason='wall_clock_timeout'; failed.append('runtime_limit')
 policy=evaluate_policy_conformance({'tool_calls':trace.tools}, assigned)
 if not policy['pass']: failed.extend('policy_conformance:'+reason for reason in policy['reasons'])
 if reason not in {'normal','phase5_terminal_after_durable_writes'}: failed.append('non_normal_termination')
 static=json.loads((B/'structural_review.json').read_text())['configs'][f'config_{i:02d}']['pass']
 return {'config':str(d.relative_to(ROOT)),'static_sanity':static,'behavioral_pass':static and not failed,'failed_criteria':sorted(set(failed)),'policy_conformance':policy,'trace':{'agents':trace.agents,'model_calls':trace.calls,'tool_calls':trace.tools,'max_prompt_tokens':max([x['prompt_tokens'] for x in trace.calls],default=0),'artifact_ids':sorted({x['artifact_id'] for x in trace.tools if x['artifact_id']}),'final_stored_ids':sorted(after),'runtime_seconds':elapsed,'termination_reason':reason}}
async def main():
 OUT.mkdir(parents=True,exist_ok=False)
 default_ids=QUALIFICATION_IDS
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
 if selected:
  (ROOT/'artifacts/sampo_phase_5/decision_report.md').write_text('# Phase 5 decision report\n\nSelection used behavioral qualification only; no private ground truth or accuracy.\n\nSelected: '+selected+'\nQualification report: '+str((OUT/'report.json').relative_to(ROOT))+'\n')
if __name__=='__main__': asyncio.run(main())
