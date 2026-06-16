# PaperSpine Configuration

| Field | Value |
|---|---|
| Workflow | `rewrite_existing` |
| Scene | `conference` |
| Tier | `pro` |
| Output language | `en` |
| Target | USENIX Security 2027 |
| Draft | `docs/CS4M_UsenixSecurity.tex` |
| Materials | Current repository (`docs`, `outputs`, `scripts`, `tests`) |
| Word output | `none` |
| Translation package | `none` |
| Reference mode | `local_first` |
| Citation target count | `10` for the planning gate; expand before final manuscript |

## Confirmed User Direction

The user wants a USENIX Security-style paper that makes CS4M's deployment
argument explicit: CS4M is a true event-arrival online PIDS. It scores each
incoming provenance event before that event updates detector state, emits
immediately explainable evidence, and avoids full graph materialization,
window completion, and periodic batch inference.

This stage builds PaperSpine planning artifacts first. It does not directly
modify `docs/CS4M_UsenixSecurity.tex`.

## Hard Boundaries

- Do not fabricate results, datasets, citations, figures, or p-values.
- Do not turn post-hoc label audits into runtime policy claims.
- Do not claim final CADETS semantic v3 results; current v3 work is projection
  evidence only.
- Do not run training, inference, or full experiments during the spec stage.
- Use citations conservatively and verify bibliography before final writing.
- The planning-stage citation bank uses a target count of 10, requiring 30
  candidates. Before final Related Work, expand and verify the bank if the
  manuscript needs a broader bibliography.
