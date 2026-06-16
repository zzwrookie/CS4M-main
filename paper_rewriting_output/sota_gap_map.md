# SOTA Gap Map

| Candidate Contribution | What SOTA Already Does | User Evidence | Real Gap | Claim Strength | Risk |
|---|---|---|---|---|---|
| Event-arrival online PIDS | PIDSs collect audit streams and often run online or periodic analysis over graphs, segments, or reconstructed provenance. | Current draft algorithm and online alert artifacts (`online_event_alerts.csv`). | Existing "online" often refers to log collection or periodic processing, not necessarily immediate per-event alert semantics. | Strong if implementation and outputs prove score-before-update. | Needs careful wording to avoid unfairly dismissing prior systems. |
| Bounded semantic state instead of full graph materialization | Graph PIDSs manage graph growth with pruning, sketching, segmentation, embedding reuse, or graph learning. | OFSM design in draft; config and memory/profiling artifacts in result dirs. | Per-event detection needs a compact state object that does not wait for graph completeness. | Strong design claim. | Must report actual memory/runtime only from verified artifacts. |
| Explainable online alert evidence | Attribution-focused systems improve reconstruction quality after graph analysis. | Event alert/explanation files and node evidence outputs. | Online alerting should emit actionable evidence at scoring time, not only after reconstruction. | Moderate to strong. | Need final columns and examples from outputs. |
| Cross-OS semantic adaptation | PIDSs often evaluate on several datasets, but tokenizer assumptions can be dataset-specific. | Semantic generic audit over FreeBSD/Linux/Android/Windows. | Portable online PIDS requires runtime-visible semantic rules per OS and audits to prevent generic-token collapse. | Moderate. | Some semantic work is still audit/projection, not final runtime result. |
| User-confirmed motivation | User wants to critique pseudo-real-time PIDS and foreground CS4M as true event-level online detection. | This session's confirmed text and requested PaperSpine flow. | The paper needs a single spine instead of scattered Word2Vec/SSM/OFSM claims. | Strong writing-control claim. | Must not inflate into unsupported superiority claims. |

## Gap Summary

The strongest gap is not "better graph learning." It is the mismatch between
graph/window/segment inference units and the deployment requirement of alerting
when each event arrives. CS4M's contribution should be framed as changing the
online detection unit from a completed provenance object to an individual
arriving event, with bounded semantic state and score-before-update causality.
