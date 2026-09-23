"""DEPRECATED / NON-OFFICIAL. Use run_sampo_phase_5_smoke.py instead."""
from __future__ import annotations
import asyncio,csv,json,time
from pathlib import Path
from fedotmas import MAS
from fedotmas.mas.models import MASConfig
ROOT=Path(__file__).resolve().parents[1]
BATCH=ROOT/'artifacts/sampo_phase_5/structural_review/batch_68c6ec377966471cbd688efd43e43107'
CONFIG=BATCH/'config_01'; OUT=ROOT/'artifacts/sampo_phase_5/smoke_config_01'
def rows():
 with (ROOT/'artifacts/sampo_benchmark/pilot_inputs.csv').open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def stored(run):
 p=ROOT/f'artifacts/sampo_benchmark/mas_runs/{run}.jsonl'
 return {json.loads(x)['example_id'] for x in p.open(encoding='utf-8') if x.strip()} if p.exists() else set()
async def main():
 if OUT.exists(): raise RuntimeError('Refusing to overwrite smoke output')
 OUT.mkdir(parents=True); meta=json.loads((CONFIG/'metadata.json').read_text()); run=meta['run_id']; config=MASConfig.model_validate_json((CONFIG/'config.json').read_text()); base=(CONFIG/'task.txt').read_text(); allrows=rows(); report={'config':str(CONFIG.relative_to(ROOT)),'run_id':run,'batches':[],'private_gt_exposed':False,'fresh_sessions':True}
 for offset in (0,20,40):
  assigned=[r['example_id'] for r in allrows[offset:offset+20]]; before=stored(run); start=time.perf_counter()
  task=base+f'\nHARNESS ASSIGNMENT: process only offset {offset}; exactly these IDs: {assigned}. Do not process any other ID. Do not finalize the pilot.'
  state=await MAS(mcp_servers=['sampo-benchmark','sandbox-light']).build_and_run(config,task,timeout=900)
  after=stored(run); added=after-before; outside=added-set(assigned); missing=set(assigned)-after
  item={'offset':offset,'ids':assigned,'runtime_seconds':time.perf_counter()-start,'added_ids':sorted(added),'missing_ids':sorted(missing),'outside_ids':sorted(outside),'state':str(state)}; report['batches'].append(item)
  if outside or missing: raise RuntimeError(f'batch {offset} failed: missing={sorted(missing)} outside={sorted(outside)}')
 (OUT/'smoke_report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__': asyncio.run(main())
