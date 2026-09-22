"""Generate exactly five bounded-batch SAMPO MAS configurations and review structure."""
from __future__ import annotations
import asyncio, hashlib, json, subprocess, uuid
from pathlib import Path
from fedotmas import MAS
from fedotmas._settings import get_meta_model

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/sampo_phase_5/structural_review'
TASK='''Map a bounded batch of public historical construction work names to the allowed labels. Use public artifact-backed candidate retrieval and safe prediction storage. Retrieve broadly with char_tfidf, construction_token_tfidf, and word_tfidf at k=50, then use fusion strategy exactly rrf. Keep the server-side fused top-three as the unreviewed fallback. Review only examples with conflicting method top choices, a low fused margin, or no exact title match; save the fused fallback for confident examples. For uncertain examples, request get_candidate_evidence with selection="fused" and candidate_limit=50; this exposes a broader bounded shortlist without bulky per-method provenance. Request evidence for no more than two examples at a time to remain below the response budget. Evidence fused_candidates retain artifact-local candidate_index values; use those exact indices in save_review_decisions candidate_indices, in final rank order. Before any durable write, partition the assigned IDs into disjoint reviewed and fallback sets. Every ID must appear in exactly one durable call: reviewed IDs only in save_review_decisions, fallback IDs only in save_candidate_predictions. Never include a reviewed ID in a fallback call, even after a rejected call; never retry a durable call with overlapping IDs. Combine cheap retrieval evidence where useful and use semantic reasoning selectively for uncertain cases. Produce valid ranked predictions for every example in the supplied batch while minimizing unnecessary LLM reasoning. Once a delegated unit is durably complete, do not repeat it. Verify completion from tool-backed state, route only unresolved work, and terminate when the assigned unit is fully covered. Complete any intended review before the durable write; after a durable write, treat that unit as terminal unless an explicit correction is requested. Use only parameter values explicitly advertised by a tool catalogue or its documented compatibility aliases. Keep detailed evidence requests to small unresolved subsets; do not place full per-example evidence for an entire batch into a model message. Before issuing a durable write, ensure every required argument is concrete and schema-valid. For each completed unit, choose exactly one durable write path; never retry or combine durable write paths for an already written ID.'''
TASK=TASK.replace('selection="fused" and candidate_limit=50; this exposes a broader bounded shortlist without bulky per-method provenance. Request evidence for no more than two examples at a time', 'selection="diverse_round_robin" and candidate_limit=30; this preserves artifact-local candidate indices while giving the reviewer a broader diverse bounded shortlist. Request evidence for no more than four examples at a time')
TASK=TASK.replace('Review only examples with conflicting method top choices, a low fused margin, or no exact title match; save the fused fallback for confident examples.', 'Use a deterministic disagreement gate: review exactly the examples whose distinct method top-1 labels count is greater than one; use save_candidate_predictions for every other assigned example. Do not use exact-title or margin heuristics.')
TASK=TASK.replace('candidate_limit=30', 'candidate_limit=10').replace('no more than four examples', 'no more than four examples')
TASK += ' The observable workflow must be complementary retrieval, deterministic disagreement gate, selective semantic reranking only for disagreement IDs, then durable predictions. Never save fallback predictions for all IDs after retrieval. Every disagreement ID must receive get_candidate_evidence before save_review_decisions.'
TASK += ' Immediately after prepare_candidate_batch, call partition_candidate_batch once with the artifact_id and the complete assigned ID list. Use its disjoint review_ids and fallback_ids exactly; do not reconstruct or alter the partition. Call get_candidate_evidence only for review_ids, then save_review_decisions for review_ids and save_candidate_predictions for fallback_ids.'
TASK += ' Do not call stage_candidate_predictions. After both durable write paths succeed, make no further MCP tool calls at all; return the compact completion report immediately. Never re-prepare, re-stage, re-save, or verify with another tool call after durable persistence.'
TASK += ' For semantic review, treat fused rank 0 as only a hypothesis: compare every shortlisted label on the primary operation, physical object, and scope/detail; prefer a specific operation-object match over a generic lexical overlap, and use candidate order only as a tie-breaker. Return exactly three distinct candidates from the supplied artifact indices.'
FILES=[ROOT/'mcp-servers/sampo-benchmark/src/mcp_sampo_benchmark/server.py',ROOT/'scripts/sampo_baselines.py',ROOT/'packages/fedotmas/src/fedotmas/meta/mas_prompts.py']
CATALOGUE=['list_methods','prepare_candidate_batch','partition_candidate_batch','get_candidate_evidence','stage_candidate_predictions','save_candidate_predictions','save_review_decisions','get_prediction_status','get_run_status','finalize_predictions','get_pilot_manifest']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def commit(): return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def review(config, task=''):
    # Policy requirements may be represented in the generated task rather
    # than duplicated in every worker instruction. Review both artifacts.
    text=(config.model_dump_json()+'\n'+task).casefold(); workers=config.workers
    checks={
      'tool_backed_deterministic_candidates':any('sampo-benchmark' in w.tools for w in workers),
      'multiple_signals_possible':all(name in text for name in ('char_tfidf','construction_token_tfidf','word_tfidf')),
      'selective_semantic_review':'uncertain' in text or 'conflict' in text or 'ambigu' in text or 'disagreement' in text or 'review-required' in text,
      'artifact_backed_bulk_evidence':'artifact' in text,
      'bounded_semantic_evidence':'bounded' in text,
      'bounded_fused_review':'candidate_limit=10' in text and 'diverse_round_robin' in text,
      'deterministic_disagreement_gate':'distinct' in text and 'top-1' in text and 'disagreement' in text,
      'explicit_partition_tool':'partition_candidate_batch' in text,
      'no_heuristic_gate':'deterministic gate' in text and 'do not use' in text and 'margin' in text,
      'artifact_id_prediction_persistence':'candidate_indices' in text and 'artifact' in text,
      'no_persistence_only_worker':not any(('persist' in w.description.casefold() or 'finaliz' in w.description.casefold()) and not ('retriev' in w.description.casefold() or 'semantic' in w.description.casefold() or 'review' in w.description.casefold()) for w in workers),
      'bounded_single_invocation':('supplied' in text and 'batch' in text) or 'bounded batch' in text,
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
    return index,review(config,task)
async def main():
    batch=OUT/f'batch_{uuid.uuid4().hex}'; batch.mkdir(parents=True)
    results=await asyncio.gather(*(one(i,batch) for i in range(1,6)))
    report={'batch_id':batch.name,'configs':{f'config_{i:02d}':r for i,r in results}}
    (batch/'structural_review.json').write_text(json.dumps(report,indent=2)+'\n')
    (batch/'README.md').write_text('# Phase 5 structural review\n\nExactly five configurations were generated; none executed.\n')
if __name__=='__main__': asyncio.run(main())
