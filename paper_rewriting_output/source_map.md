# Source Map

| Source ID | Path / Source | Content Used | Claim Scope | Status |
|---|---|---|---|---|
| S1 | `docs/CS4M_UsenixSecurity.tex` | Current method draft: problem statement, pseudo-real-time critique, CS4M state model, E4 dual scoring, OFSM, score-before-update, calibration, algorithm sketch, references. | Draft logic and reusable method text. | Local source, verified present. |
| S2 | `docs/reproduction/CS4M_BACKUP_CONFIRMATION_MANIFEST_20260607.md` | Reproducible artifact manifest for CADETS_E3 and THEIA_E3, key online outputs, model/cache locations, memory/profiling files. | Reproducibility and evidence boundary. | Local source, verified present. |
| S3 | `outputs/diagnostics/semantic_generic_audit_20260615_162909/semantic_generic_audit_report.md` | Cross-dataset semantic audit for THEIA_E3, CADETS_E3, CLEARSCOPE_E3, CLEARSCOPE_E5, and OS-specific tokenizer summary. | Cross-OS semantic design and limitations. | Local source, verified present. |
| S4 | `outputs/diagnostics/cadets_e3_e4_freebsd_semantic_audit_20260615_223843/` | CADETS_E3 E4 FreeBSD semantic audit. | Evidence that semantic quality must precede Word2Vec/training claims. | Local source, verified present. |
| S5 | `outputs/diagnostics/cadets_e3_e4_freebsd_semantic_v3_projection_20260615_225549/` | Read-only candidate semantic v3 projection: process generic decreases, file empty path is explicitly bucketed, netflow unchanged. | Future-work/case-study evidence only, not final method claim. | Local source, verified present. |
| S6 | `outputs/results/phase3g_theia_e3_e4_theia_v1_lazy100k_pair_only_full_rerun_20260602_142411/THEIA_E3_PHASE3E_E4_NONE/` | THEIA_E3 online outputs and metrics. | Evaluation evidence for Linux setting. | Local source, verified present. |
| S7 | `outputs/results/tflr_light/CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_FULL_GROUPQ0999_FP_GUARD_V3B_NODE_EVIDENCE/` | ClearScope E3 full result with online alerts, node evidence, metrics, and configuration. | Evaluation evidence for Android setting. | Local source, verified present. |
| S8 | `docs/reports/2026-06-15-clearscope-e5-dual-channel-validation-candidates.md` and related ClearScope E5 reports | Validation-only dual-channel candidate design and post-stream diagnostics. | Design discussion and future policy evaluation, not final runtime claim unless supported by final run evidence. | Local source, verify before citing exact numbers. |
| S9 | `scripts/pipeline/io/db_stream.py`, `scripts/pipeline/state/online_state_runtime.py`, `scripts/pipeline/outputs/alert_output.py` | Streaming event ingestion, online state runtime, and alert output implementation. | Implementation evidence for true online processing. | Local source, inspect before final method claims. |
| S10 | `tests/` | Tests for streaming behavior, no-label outputs, semantic audits, runner boundaries, candidate projections. | Engineering rigor and leakage-control support. | Local source, verify exact tests before final claims. |
| E1 | USENIX Security CFP pages | Venue expectations and formatting constraints. | Target-scene requirements. | Use latest official 2027 CFP when available; 2026 CFP is temporary template. |
| E2 | Bilot et al., "Sometimes Simpler is Better..." | SOTA PIDS deployment critique and simplicity/scalability argument. | Exemplar and related-work support. | Verify bibliography before final LaTeX. |
| E3 | Jiang et al., ORTHRUS | Attribution-focused provenance-based IDS. | Related-work contrast: attribution quality vs event-arrival alerting. | Verify bibliography before final LaTeX. |
| E4 | KAIROS | Whole-system provenance detection/investigation. | Related-work contrast: graph-centered detection/investigation. | Verify bibliography before final LaTeX. |
| E5 | TAPAS | Task-guided process provenance graph segmentation and online APT detection. | Related-work contrast: segmentation/window graph unit vs event unit. | Verify bibliography before final LaTeX. |

## Evidence Rule

Every numerical claim in the rewritten manuscript must point to a local result,
diagnostic report, or reproducibility manifest. External papers teach structure
and support related-work claims; they do not supply CS4M results.
