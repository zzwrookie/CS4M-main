# Research Dossier

## Venue Requirements

USENIX Security is a top systems security venue. The 2027 call for papers must
be rechecked before final submission; until then, the latest official CFP is
only a temporary template. The manuscript should follow USENIX-style security
paper norms: clear threat model, concrete system assumptions, reproducible
evaluation, conservative claims, artifact-aware writing, and readable two-column
LaTeX. Any final page, anonymity, ethics, or artifact-submission rules must be
updated from the official 2027 CFP.

## Review Criteria

The paper must convince reviewers that the problem is real, the deployment
model is stricter than prior work, the design follows from that model, and the
evaluation supports the exact claims made. For CS4M, the key review risk is not
whether event-level online detection is interesting; it is whether the paper can
prove that the system actually processes and alerts per arriving event, avoids
label leakage, remains lightweight, and compares fairly with graph-centered
PIDSs.

## Accepted Paper Patterns

Strong systems security papers typically stage the argument as: operational
gap, design principles, system architecture, implementation constraints,
evaluation questions, and limitations. Relevant PIDS exemplars also separate
detection, attribution, and investigation. CS4M should transfer that pattern by
making "alerting unit" the organizing axis: graph/window/segment versus
arriving event.

## Constraints for This Paper

The paper must avoid overclaiming. Cross-OS performance should be stated only
where local artifacts support it. ClearScope E5 dual-channel work and CADETS
semantic v3 projection are valuable evidence, but they are not final policy
claims unless final runtime results exist. Ground truth is post-stream
evaluation evidence only.
