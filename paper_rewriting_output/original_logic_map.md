# Original Logic Map

| Original Unit | Current Text Role | Evidence Used | Motivation Link | Problem | Keep / Move / Rewrite / Delete |
|---|---|---|---|---|---|
| Title | Names CS4M as lightweight real-time semantic-state PIDS. | None beyond draft. | Partly aligned with event-arrival online motivation. | Does not foreground "event-arrival" or critique graph/window unit. | Rewrite. |
| Section 3 Problem Statement | Defines PIDS, pseudo-real-time, online model, design objective. | Existing citations and narrative. | Strong link to main motivation. | It starts at Section 3 and assumes prior sections; needs Introduction integration and clearer contribution setup. | Keep core, rewrite structure. |
| Pseudo-real-time subsection | Critiques graph-based alerting. | Bilot/KAIROS references in draft. | Central. | Needs fairer language and stronger distinction between log collection online and alert semantics online. | Rewrite carefully. |
| Existing limitations subsection | Surveys HOLMES, NoDoze, UNICORN, KAIROS, FLASH, ORTHRUS. | Existing bib keys. | Supports gap. | Needs TAPAS and a taxonomy by alerting unit; avoid broad claims without citations. | Rewrite and expand. |
| Online detection model | Defines event stream and alert decision. | Mathematical notation. | Central. | Good but should move earlier as a formal problem statement after motivation. | Keep and polish. |
| Challenges and Solutions | Maps C1-S5. | Design logic. | Strong. | Needs tighter connection to contributions and less repetition with design section. | Keep as design principles or compress. |
| CS4M Design | Semantic modeling, state table, OFSM, E4 dual prediction, scoring, suppression. | Method draft and implementation inspection. | Strong. | Earlier draft mixed E5 variant material into the method; user clarified E4 dual is the paper spine. | Keep, audit equations, reorganize around E4 dual. |
| Algorithm figure | Shows event processing order. | Draft pseudocode. | Very strong. | Should become central figure/algorithm in Section 4 or 5. | Keep and polish. |
| Overview figure placeholder | Describes desired figure. | Draft text only. | Useful. | Not an actual figure; should become a figure plan/caption. | Move to figure plan. |
| Appendix theory | Propositions for ZOH, OFSM, COW, and score-before-update causality. | Mathematical derivations. | Supports rigor. | Remove E5 perturbation material from the main paper path. | Keep after method. |
| Bibliography | Existing references. | Draft bib items. | Supports related work. | Bibliography is manual; must verify TAPAS and recent papers before final. | Keep, verify later. |
