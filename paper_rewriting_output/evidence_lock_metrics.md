# CS4M Paper Evidence Lock Metrics

## Locked Outputs

- `evidence_lock_topk_table.csv`
- `evidence_lock_runtime_table.csv`
- `evidence_lock_optc051_summary.csv`
- `evidence_lock_metric_provenance.md`

## OPTC_051 Summary

- Original GT hit: `66/114`
- Unique compact GT hit: `19/67`
- Netflow GT hit: `49/49`
- Process GT hit: `15/59`
- File GT hit: `2/6`
- Event TP/FP: `208 / 1158`
- Total alerts: `1366`

## ClearScope E5 Risk

ClearScope E5 top-k remains unresolved for the paper table. Keep the
`needs_recompute_51_gt_source_conflict` status until a 51-GT top-k
canonical-aware table is generated.
