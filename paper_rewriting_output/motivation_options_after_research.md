# Motivation Options After Research

| Option | One-Sentence Motivation | Core Innovation | Why It Is Not Overbroad | Required Evidence | Best-Fit Paper Arc |
|---|---|---|---|---|---|
| A | CS4M makes provenance intrusion detection truly event-arrival online by replacing graph/window inference with score-before-update semantic state. | Change the inference unit from completed graph/window/segment to arriving event. | Does not claim to solve all attribution or investigation tasks; focuses on alert timing and deployable state. | Online alert outputs, streaming implementation, state memory/profiling, no-label-leakage tests. | Recommended: problem gap -> event unit -> state design -> evaluation -> limitations. |
| B | CS4M is a lightweight cross-OS semantic-state detector that preserves deployability across FreeBSD, Linux, Android, and Windows provenance settings. | Cross-OS runtime-visible semantic tokenization plus compact residual scoring. | Does not require all OS variants to be final policy contributions; semantic audits can be evidence and limitations. | Semantic generic audit, dataset-specific outputs, tokenizer tests. | Good secondary arc, but weaker than A as the main USENIX spine. |
| C | CS4M improves online provenance alert usefulness by combining residual event scoring with immediate node evidence. | Explainable alert evidence at scoring time rather than post-hoc graph reconstruction alone. | Does not compete directly with full attribution systems like ORTHRUS; it targets alert generation. | `online_event_alerts.csv`, explanation files, node evidence outputs, case studies. | Useful as a supporting contribution under A. |

## Recommendation

Choose Option A as the controlling motivation. Options B and C should become
supporting contributions: cross-OS semantics and explainable evidence are
important because they make event-arrival online detection deployable.
