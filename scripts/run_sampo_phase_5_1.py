"""Generate exactly five bounded-batch SAMPO MAS configurations and review structure."""
from __future__ import annotations
import asyncio, hashlib, json, subprocess, uuid
from pathlib import Path
from fedotmas import MAS
from fedotmas._settings import get_meta_model

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/sampo_phase_5/structural_review'
TASK='''Map a bounded batch of public historical construction work names to the allowed labels. Use public artifact-backed candidate retrieval and safe prediction storage. Combine cheap retrieval evidence where useful and use semantic reasoning selectively for uncertain cases. Produce valid ranked predictions for every example in the supplied batch while minimizing unnecessary LLM reasoning. Once a delegated unit is durably complete, do not repeat it. Verify completion from tool-backed state, route only unresolved work, and terminate when the assigned unit is fully covered. Complete any intended review before the durable write; after a durable write, treat that unit as terminal unless an explicit correction is requested. Use only parameter values explicitly advertised by a tool catalogue or its documented compatibility aliases. Keep detailed evidence requests to small unresolved subsets; do not place full per-example evidence for an entire batch into a model message.'''
FILES=[ROOT/'mcp-servers/sampo-benchmark/src/mcp_sampo_benchmark/server.py',ROOT/'scripts/sampo_baselines.py',ROOT/'packages/fedotmas/src/fedotmas/meta/mas_prompts.py']
CATALOGUE=['list_methods','prepare_candidate_batch','get_candidate_evidence','stage_candidate_predictions','save_candidate_predictions','save_review_decisions','get_prediction_status','get_run_status','finalize_predictions','get_pilot_manifest']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def commit(): return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def review(config):
    text=config.model_dump_json().casefold(); workers=config.workers
    checks={
      'tool_backed_deterministic_candidates':any('sampo-benchmark' in w.tools for w in workers),
      'multiple_signals_possible':'multiple' in text or 'complementary' in text or 'independent' in text,
      'selective_semantic_review':'uncertain' in text or 'conflict' in text or 'ambigu' in text,
      'artifact_backed_bulk_evidence':'artifact' in text,
      'bounded_semantic_evidence':'bounded' in text,
      'artifact_id_prediction_persistence':'artifact' in text and ('candidate index' in text or 'candidate_indices' in text or 'example id' in text),
      'no_persistence_only_worker':not any(('persist' in w.description.casefold() or 'finaliz' in w.description.casefold()) and not ('retriev' in w.description.casefold() or 'semantic' in w.description.casefold() or 'review' in w.description.casefold()) for w in workers),
      'bounded_single_invocation':'supplied batch' in text or 'bounded batch' in text,
      'no_full_pilot_coordinator':'entire pilot' not in config.coordinator.instruction.casefold() and 'every pilot id' not in config.coordinator.instruction.casefold(),
    }
    return {'checks':checks,'pass':all(checks.values())}
async def one(index,batch):
    run_id=f'phase5_{index}_{uuid.uuid4().hex}'
    task=TASK+f'\nUse durable prediction run_id {run_id}. The harness will supply exactly one batch offset and IDs; do not process any other examples and do not finalize the whole pilot.'
    mas=MAS(meta_model=get_meta_model(),worker_models=['openai/gpt-5.6-luna','openai/gpt-5-mini'],mcp_servers=['sampo-benchmark','sandbox-light'])
    config=await mas.generate_config(task); d=batch/f'config_{index:02d}'; d.mkdir()
    (d/'config.json').write_text(config.model_dump_json(indent=2)+'\n'); (d/'task.txt').write_text(task+'\n')
    (d/'metadata.json').write_text(json.dumps({'index':index,'run_id':run_id,'meta_model':get_meta_model(),'model_assignments':{'coordinator':config.coordinator.model,'workers':{w.name:w.model for w in config.workers}},'git_commit':commit(),'sha256':{str(p.relative_to(ROOT)):sha(p) for p in FILES},'mcp_tool_catalogue':CATALOGUE,'executed':False},indent=2)+'\n')
    return index,review(config)
async def main():
    batch=OUT/f'batch_{uuid.uuid4().hex}'; batch.mkdir(parents=True)
    results=await asyncio.gather(*(one(i,batch) for i in range(1,6)))
    report={'batch_id':batch.name,'configs':{f'config_{i:02d}':r for i,r in results}}
    (batch/'structural_review.json').write_text(json.dumps(report,indent=2)+'\n')
    (batch/'README.md').write_text('# Phase 5 structural review\n\nExactly five configurations were generated; none executed.\n')
if __name__=='__main__': asyncio.run(main())
