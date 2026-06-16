# Confirmed Motivation

| Field | Content |
|---|---|
| Source | User-provided and confirmed in the 2026-06-15 PaperSpine request. |
| Confirmed motivation statement | Existing provenance-based intrusion detection systems are often real-time in log collection, but not real-time in alert semantics. They typically require graph construction, window completion, batching, pruning, segmentation, or periodic inference before an alert can be raised. CS4M targets a stricter deployment model: every provenance event is scored as it arrives, before the event updates detector state, and the system emits immediately explainable alert evidence without materializing a full provenance graph. |
| One-sentence red thread | CS4M changes the online detection unit from a completed provenance graph, window, or segment to the individual arriving event. |
| Field problem | PIDSs are valuable for APT detection and investigation, but graph-centered pipelines can delay alert generation and increase deployment cost. |
| Specific gap | Prior "online" PIDS designs often still depend on graph/window/segment completion or periodic inference; this is pseudo-real-time for event-level alerting. |
| Design response | Maintain compact semantic state, score each event before state update with E4 dual target-case heads, calibrate residual scores, emit event/node evidence immediately, and bound online state memory. |
| Main evidence available | Existing LaTeX draft, online alert artifacts, result directories, reproduction manifest, semantic audits, streaming/pipeline tests, and scripts. |
| Target-venue fit | USENIX Security systems paper: precise deployment gap, concrete system design, reproducible artifacts, conservative claims, limitations. |
| Prioritized claims | True event-arrival online alerting; score-before-update causality; E4 dual target-case semantic scoring; bounded semantic-state memory; explainable online alerts; cross-OS semantic audit discipline. |
| Claims to avoid | Do not claim universal superiority over all PIDSs, final attribution quality, final CADETS semantic v3 performance, or runtime policies derived from test labels. |
| Secondary motivations or boundaries | Cross-OS semantics and explainability support the main deployment claim; they are not separate inflated contributions. |
| Surface language priorities | Title, abstract opening, introduction gap paragraph, contribution list, and method overview should foreground event-arrival online detection. |

## Section Consequences

| Section | What This Motivation Requires | What It Should Avoid |
|---|---|---|
| Abstract | Open with pseudo-real-time mismatch and state CS4M's stricter event-arrival model. | Do not open with Word2Vec or generic anomaly detection. |
| Introduction | Build necessity: graph-first PIDSs are useful but not the right alerting unit for immediate deployment. | Do not make a literature list or dismiss prior work unfairly. |
| Methods | Tie every mechanism to online alerting: semantic state, score-before-update, E4 dual target-case heads, OFSM, thresholds, and pair-only suppression. | Do not read like code documentation or present E2/E5 variants as the main method. |
| Results | Answer whether CS4M alerts online, remains lightweight, works across datasets/OSes, and preserves explainable evidence. | Do not use post-hoc audits as runtime threshold evidence. |
| Discussion | Admit limits: semantics are OS-specific, some candidate rules require retraining, attribution is not the same target as online alerting. | Do not overclaim solved PIDS deployment in all settings. |
