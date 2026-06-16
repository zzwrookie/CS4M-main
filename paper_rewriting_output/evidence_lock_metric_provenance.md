# CS4M Paper Evidence Lock Metric Provenance

GT is post-stream evaluation only. These tables do not define runtime rules.

## Source Status

- THEIA_E3, CADETS_E3, CLEARSCOPE_E3, THEIA_E5: values are user-provided
  paper-table locks and must be traced to final result files before final LaTeX.
- OPTC_051: canonical-aware event and GT-kind metrics are extracted from
  `outputs/diagnostics/optc_windows_v13c_canonical_eval_audit_20260615_224932/summary.json`; node top-k values come from the generated
  `node_topk_metrics.csv` in the OPTC v1.3c full infer result.
- CLEARSCOPE_E5: status is
  `needs_recompute_51_gt_source_conflict` because the user-provided
  dual-channel audit row reports 14 / 377 node TP/FP, while the default
  result-directory `node_topk_metrics.csv` currently reports zero TP under
  its current mapping.

## Required Before Final Manuscript

1. Recompute CLEARSCOPE_E5 top-k under the 51-GT canonical-aware mapping.
2. Replace user-provided locked rows with exact source-file extraction where
   possible, or keep them labeled as user-provided table locks.
3. Use OPTC_051 original GT and unique compact GT denominators separately.
