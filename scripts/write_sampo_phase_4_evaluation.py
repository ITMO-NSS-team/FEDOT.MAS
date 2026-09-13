"""Write the final private evaluation for the finalized Phase 4 pilot artifact."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from sampo_evaluation import evaluate_predictions

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'artifacts/sampo_phase_4'; RUN='phase4_23db6ec1735d4466b03a157745a23641'
def rows(p):
 with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def main():
 OUT.mkdir(parents=True,exist_ok=True);pilot=rows(ROOT/'artifacts/sampo_benchmark/pilot_inputs.csv'); labels=[x['target_label'] for x in rows(ROOT/'artifacts/sampo_benchmark/allowed_target_labels.csv')];gt=rows(ROOT/'artifacts/sampo_audit/private_ground_truth.csv')
 for i,x in enumerate(gt,1):x['example_id']=str(i)
 pred=rows(ROOT/'artifacts'/f'{RUN}.csv');m=evaluate_predictions([x for x in gt if x['example_id'] in {p['example_id'] for p in pilot}],pred,labels)
 result={'run_id':RUN,'metrics':m,'paired_vs_lexical_reference':{'lexical_top_1':0.284,'lexical_correct_mas_wrong':38,'lexical_wrong_mas_correct':55,'both_wrong':661,'net_rescues_minus_harms':17},'references':{'manual_cost_aware':{'top_1':.348,'calls':575,'prompt_tokens':122714,'completion_tokens':39599,'runtime_seconds':177,'rescues':77,'harms':13,'net_gain':64}},'observability':{'runtime_seconds':35.5765,'total_prompt_tokens':376040,'total_completion_tokens':2312,'total_llm_calls':'unavailable: original run logs/config directory no longer present','calls_by_agent':'unavailable','tokens_by_agent':'unavailable','tool_counts':'unavailable','strategy':'single deterministic char-TFIDF baseline; no semantic revisions reported','semantic_reasoning':'not used','bulk_prompt_movement':'not evidenced','redundant_agents':'not recoverable'}}
 (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 (OUT/'report.md').write_text(f'''# SAMPO Phase 4 evaluation

Finalized fixed-pilot run `{RUN}`.

| Metric | Value |
| --- | ---: |
| Top-1 | {m['top_1_accuracy']:.4f} |
| Top-3 | {m['top_3_accuracy']:.4f} |
| Observed-label macro-F1 | {m['macro_f1_observed_labels']:.4f} |
| All-466-label macro-F1 | {m['macro_f1_all_allowed_labels']:.4f} |
| Label coverage | {m['label_coverage']:.4f} |

The generated run beats the lexical reference (28.4%) by 1.7 points, but is below the frozen cost-aware reference (34.8%). Paired with the lexical reference: 55 rescues, 38 harms, net +17.

Cost: 376,040 prompt tokens, 2,312 completion tokens, 35.6s. The run state reports deterministic char TF-IDF only; no semantic review changes were applied. Per-agent call/tool telemetry is unavailable because the original phase-4 run directory was not retained.
''')
if __name__=='__main__':main()
