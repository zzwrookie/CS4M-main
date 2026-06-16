# Exemplar Learning Dossier

## Exemplar Inventory

| Title | Venue / Year | Why Selected |
|---|---|---|
| Sometimes Simpler is Better: A Comprehensive Analysis of State-of-the-Art Provenance-Based Intrusion Detection Systems | USENIX Security 2025 | Directly supports the simplicity, deployability, and PIDS complexity framing. |
| ORTHRUS: Achieving High Quality of Attribution in Provenance-Based Intrusion Detection Systems | USENIX Security 2025 | Shows how an attribution-focused PIDS motivates output quality and analyst usefulness. |
| TAPAS: An Efficient Online APT Detection with Task-Guided Process Provenance Graph Segmentation and Analysis | SOTA PIDS paper, verify final venue | Useful contrast for "online" designs whose inference unit is still a segmented graph/process graph. |
| KAIROS: Practical Intrusion Detection and Investigation using Whole-System Provenance | IEEE S&P 2024 | Strong graph-centered practical PIDS exemplar. |
| FLASH: A Comprehensive Approach to Intrusion Detection via Provenance Graph Representation Learning | IEEE S&P 2024 | Useful contrast for graph representation learning and computational cost. |
| UNICORN / HOLMES / NoDoze | NDSS / IEEE S&P | Foundational runtime provenance, correlation, and triage examples. |

## Structural Patterns

The most relevant pattern is to make the evaluation object explicit. ORTHRUS
organizes around attribution quality rather than only detection accuracy. CS4M
should similarly organize around alerting semantics: the unit of online
decision is the arriving event, not a completed graph. Another reusable pattern
is to define a deployment pain point first, then show that architecture follows
from the pain point rather than from model novelty.

## Rhetorical Patterns

The paper should praise prior work for what it optimizes, then isolate the
unmet deployment model. The contrast should be phrased as a mismatch of
computational objects: graph, segment, or window versus event stream. This is
more defensible than saying prior systems are simply "not real-time."

## Language Patterns

Use precise systems language: "event-arrival alerting," "score-before-update,"
"graph materialization," "window completion," "post-stream evaluation,"
"runtime-visible fields," and "bounded online state." Avoid vague claims such
as "high performance" unless a table immediately supports them.
