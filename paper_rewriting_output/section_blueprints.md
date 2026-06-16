# Section Blueprints

## Whole-Paper Arc

Field problem: PIDSs are valuable but graph-first pipelines are hard to deploy
as immediate alerting systems. Specific gap: many "online" PIDSs are online in
collection or periodic processing, but not in event-arrival alert semantics.
Design response: CS4M scores each arriving provenance event before state update
using compact semantic state, E4 dual target-case scoring, and immediately
emitted explainable evidence.
Evidence: local online alert artifacts, streaming implementation, memory/state
artifacts, cross-OS semantic audits, and conservative evaluation.

## Proposed Manuscript Structure

1. Introduction
2. Background and Motivation
3. Problem Statement and Online Detection Model
4. CS4M Design
5. Real-Time Alerting and Explainability
6. Implementation
7. Evaluation
8. Case Studies and Cross-OS Semantic Analysis
9. Discussion and Limitations
10. Related Work
11. Conclusion

## Section 1: Introduction

Open with the deployment mismatch: defenders need immediate alerts from an
audit stream, while graph-first PIDSs often wait for graph/window/segment
construction. Define pseudo-real-time carefully. Introduce CS4M as a stricter
event-arrival online detector. Contributions should be limited to: online
alerting unit, compact semantic state, score-before-update residual scoring,
explainable evidence, and cross-OS evaluation/audit discipline.

## Section 2: Background and Motivation

Explain provenance events, graph-centered PIDSs, and why graph context is useful
for investigation. Then distinguish detection, attribution, and investigation.
The motivation is not that graphs are useless; it is that full graph
materialization is the wrong computational object for immediate event alerting.

## Section 3: Problem Statement and Online Detection Model

Move and polish the current Section 3. Define the stream, event tuple, alert
decision, score-before-update causality, threat assumptions, and no-leakage
evaluation boundary. This section should make the stricter online model formal.

## Section 4: CS4M Design

Reorganize existing method text around design requirements: semantic tokens,
Word2Vec embeddings, causal semantic state, E4 dual target-case scoring,
low-rank residual prediction, OFSM, and threshold calibration. E2/E5 variants
must not be presented as the main paper method.

## Section 5: Real-Time Alerting and Explainability

Use Algorithm 1 as the core. Explain pair-only suppression, alert output,
node-level evidence, and why endpoint-only suppression is excluded. Include a
small table of alert-output fields after inspecting actual CSV columns.

## Section 6: Implementation

Describe streaming DB ingestion, lazy embedding lookup, memmap caches, online
state runtime, and portable configuration. Use source paths and tests as
evidence. Do not expose local absolute paths as deployment defaults.

## Section 7: Evaluation

Frame evaluation questions before tables:

1. Does CS4M emit event-level alerts in stream order?
2. What is detection quality on CADETS_E3, THEIA_E3, CLEARSCOPE_E3/E5, and
   available Windows/OpTC settings?
3. What is the runtime/memory footprint?
4. Does E4 dual target-case scoring expose calibrated, explainable alert
   evidence?
5. Are alerts explainable and reproducible?

Fill only with verified local metrics. Use placeholders for missing experiments.
Use `paper_rewriting_output/evidence_lock_topk_table.csv`,
`paper_rewriting_output/evidence_lock_runtime_table.csv`, and
`paper_rewriting_output/evidence_lock_optc051_summary.csv` as the first
paper-table sources. ClearScope E5 top-k must remain marked unresolved until
the 51-GT source conflict is recomputed.

## Section 8: Case Studies and Cross-OS Semantic Analysis

Use semantic audits to show why portable PIDS needs OS-aware runtime-visible
semantics. Discuss FreeBSD, Linux, Android, and Windows tokenizer risks. Keep
GT-based audit strictly post-hoc.

## Section 9: Discussion and Limitations

State limitations plainly: attribution is not the primary target; semantic
rules require OS audits; some candidate policies need retraining; ground truth
is evaluation-only; graph reconstruction can still be valuable after alerting.

## Section 10: Related Work

Organize by computational unit:

- Graph-first provenance IDS and investigation.
- Segmentation/window-based online PIDS.
- Attribution and triage systems.
- Lightweight/runtime provenance analysis.
- Sequence/state-space models and semantic anomaly scoring.

## Section 11: Conclusion

Return to the red thread: changing the detection unit from completed provenance
object to arriving event makes CS4M a deployable, explainable, true online PIDS.

## Stage Completion Note: Sections 8-11

The manuscript now contains a complete back half for the current PaperSpine
stage:

- Section 8 uses semantic audit evidence as post-hoc cross-OS diagnostics for
  FreeBSD/Linux, Android, and Windows semantics. It explicitly states that these
  audits are not runtime policy and do not use labels during alert generation.
- Section 9 states the deployment and evidence boundaries: CS4M is an
  event-arrival alerting system, not a complete attribution engine; graph
  reconstruction remains useful after alerts; ClearScope E5 top-k and mixed
  runtime-source normalization remain TODO risks.
- Section 10 organizes related work by alerting unit and objective rather than
  by chronology. The critique remains focused on graph/window/segment alerting
  semantics, not on dismissing prior systems.
- Section 11 returns to the central claim that CS4M changes the detection unit
  from a completed provenance object to the arriving event.

E4 dual remains the paper method. E2/E5 are not introduced as method
contributions; any E5 mention is limited to dataset/evidence-risk context.
