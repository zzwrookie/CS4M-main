# ClearScope E5 V33B Full Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run independent ClearScope E5 v33b full Word2Vec, Phase3E,
Phase3G, and pipeline artifacts without reading E3 or bounded smoke artifacts.

**Architecture:** Add one E5 full-run shell runner that mirrors the proven E5
bounded runner but switches identities to `FULL`, removes bounded event limits,
and increases training epochs. Keep tokenizer, threshold, policy, evaluation,
and label logic unchanged.

**Tech Stack:** Bash runners, Python `unittest`, `legacy/tools/train_residual_word2vec_models.py`, `scripts.pipeline.entrypoints.conditional_e4`, PostgreSQL-backed ClearScope dataset streaming.

---

## Files

- Create: `tests/test_clearscope_e5_full_runner.py`
  - Responsibility: TDD coverage for full runner dry-run routing and safety preflight.
- Create: `scripts/run/run_clearscope_e5_v33b_full_phase3e_phase3g.sh`
  - Responsibility: E5 full Word2Vec, Phase3E, Phase3G, and inference orchestration.
- Generated, not committed: `outputs/models/residual_word2vec/CLEARSCOPE_E5_V33B_FULL_LATENT64*`, `outputs/cache/phase3e/*/CLEARSCOPE_E5_v33b_full*`, `outputs/cache/phase3g_*/*CLEARSCOPE_E5_v33b_full*`, `outputs/models/sspm_phase3e/CLEARSCOPE_E5_V33B_FULL_PHASE3E_BASE.pkl`, `outputs/models/phase3g_action_heads/CLEARSCOPE_E5_V33B_FULL_PHASE3G_CONDITIONAL_HEAD.pkl`, `outputs/results/tflr_light/CLEARSCOPE_E5_V33B_FULL_PIPELINE/`, `logs/clearscope_e5_v33b_full/`.

## Task 1: Add Failing Full Runner Tests

**Files:**
- Create: `tests/test_clearscope_e5_full_runner.py`

- [ ] **Step 1: Write tests for the missing full runner**

Create `tests/test_clearscope_e5_full_runner.py` with:

```python
import os
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts/run/run_clearscope_e5_v33b_full_phase3e_phase3g.sh"


def _run_runner(extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "CS4M_ROOT": str(REPO_ROOT),
            "DRY_RUN": "1",
            "PYTHON_BIN": "python3",
        },
    )
    env.update(extra_env)
    return subprocess.run(
        ["bash", str(RUNNER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


class ClearScopeE5FullRunnerTests(unittest.TestCase):
    def test_dry_run_all_uses_e5_v33b_full_artifacts(self) -> None:
        result = _run_runner({"STAGE": "all"})

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        combined = result.stdout + result.stderr
        self.assertIn("train_word2vec", combined)
        self.assertIn("--datasets CLEARSCOPE_E5", combined)
        self.assertIn("--dataset CLEARSCOPE_E5", combined)
        self.assertIn("--semantic_mode raw_detail_v33b_e5_android_safe", combined)
        self.assertIn("--max_train_events 0", combined)
        self.assertIn("--max_ref_events 0", combined)
        self.assertIn("--max_test_events 0", combined)
        self.assertIn("--word2vec_epochs 10", combined)
        self.assertIn("--sspm_conditional_max_epochs 80", combined)
        self.assertIn("CLEARSCOPE_E5_V33B_FULL", combined)
        self.assertNotIn("CLEARSCOPE_E3", combined)
        self.assertNotIn("CLEARSCOPE_E5_V33B_BOUNDED", combined)
        self.assertNotIn("CLEARSCOPE_E5_v33b_bounded", combined)

    def test_rejects_e3_dataset_even_in_dry_run(self) -> None:
        result = _run_runner({"DATASET": "CLEARSCOPE_E3", "STAGE": "all"})

        self.assertEqual(result.returncode, 2)
        self.assertIn("restricted to CLEARSCOPE_E5", result.stderr)

    def test_rejects_non_v33b_semantic_mode_even_in_dry_run(self) -> None:
        result = _run_runner(
            {
                "SEMANTIC_MODE": "raw_detail_v33_e5_android_safe",
                "STAGE": "all",
            },
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("raw_detail_v33b_e5_android_safe", result.stderr)

    def test_rejects_bounded_artifact_path_even_in_dry_run(self) -> None:
        result = _run_runner(
            {
                "PRETRAINED_RESIDUAL_EMBEDDER_PATH": (
                    "outputs/models/residual_word2vec/"
                    "CLEARSCOPE_E5_V33B_BOUNDED_LATENT64_word2vec_window3.pkl"
                ),
                "STAGE": "all",
            },
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("must not reference CLEARSCOPE_E5_V33B_BOUNDED", result.stderr)

    def test_shell_syntax_is_valid(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(RUNNER)],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
python3 -m unittest tests.test_clearscope_e5_full_runner -v
```

Expected: FAIL because `scripts/run/run_clearscope_e5_v33b_full_phase3e_phase3g.sh`
does not exist.

## Task 2: Implement Full Runner

**Files:**
- Create: `scripts/run/run_clearscope_e5_v33b_full_phase3e_phase3g.sh`

- [ ] **Step 1: Create the full runner**

Create `scripts/run/run_clearscope_e5_v33b_full_phase3e_phase3g.sh` by copying
the E5 bounded runner structure and making these exact full-run substitutions:

```text
SEMANTIC_LABEL=V33B_FULL
SEMANTIC_SUFFIX=v33b_full
LOG_DIR=logs/clearscope_e5_v33b_full
WORD2VEC_TAG=V33B_FULL_LATENT64
MAX_TRAIN_EVENTS=0
MAX_REF_EVENTS=0
MAX_TEST_EVENTS=0
SSPM_CONDITIONAL_MAX_EPOCHS=80
WORD2VEC_EPOCHS=10
WORD2VEC_WORKERS=4
infer stage name=infer_full
infer out_tag=CLEARSCOPE_E5_V33B_FULL_PIPELINE
```

The runner must reject paths containing any of:

```text
CLEARSCOPE_E3
CLEARSCOPE_E5_V33B_BOUNDED
CLEARSCOPE_E5_v33b_bounded
```

The supported stages must be:

```text
all|train_word2vec|build_phase3e_artifacts|train_phase3e_base|train_phase3g_head|infer_full
```

- [ ] **Step 2: Make the runner executable**

Run:

```bash
chmod +x scripts/run/run_clearscope_e5_v33b_full_phase3e_phase3g.sh
```

- [ ] **Step 3: Run tests to verify GREEN**

Run:

```bash
python3 -m unittest tests.test_clearscope_e5_full_runner -v
```

Expected: PASS.

- [ ] **Step 4: Run shell syntax check**

Run:

```bash
bash -n scripts/run/run_clearscope_e5_v33b_full_phase3e_phase3g.sh
```

Expected: PASS.

- [ ] **Step 5: Commit runner and tests**

Run:

```bash
git add scripts/run/run_clearscope_e5_v33b_full_phase3e_phase3g.sh tests/test_clearscope_e5_full_runner.py
git commit -m "feat: add clearscope e5 v33b full runner"
```

Expected: commit succeeds.

## Task 3: Run Full Pipeline

**Files:**
- Generated only: `outputs/`, `logs/`.

- [ ] **Step 1: Run full pipeline**

Run:

```bash
CLAD_DB_HOST="${CLAD_DB_HOST:-127.0.0.1}" \
CLAD_DB_PORT="${CLAD_DB_PORT:-5433}" \
CLAD_DB_USER="${CLAD_DB_USER:-postgres}" \
CLAD_DB_PASSWORD="${CLAD_DB_PASSWORD:-123456}" \
STAGE=all \
scripts/run/run_clearscope_e5_v33b_full_phase3e_phase3g.sh
```

Expected: all stages exit `0`.

- [ ] **Step 2: Check required full artifacts and outputs exist**

Run:

```bash
test -f outputs/models/residual_word2vec/CLEARSCOPE_E5_V33B_FULL_LATENT64_word2vec_window3.pkl
test -f outputs/models/residual_word2vec/CLEARSCOPE_E5_V33B_FULL_LATENT64_word2vec_window3.json
test -f outputs/cache/phase3e/event_indices/CLEARSCOPE_E5_v33b_full/event_index_meta.json
test -f outputs/cache/phase3e/node_embeddings/CLEARSCOPE_E5_v33b_full_latent64/node_embeddings.npy
test -f outputs/cache/phase3e/action_embeddings/CLEARSCOPE_E5_v33b_full_latent64/action_embeddings.npy
test -f outputs/models/sspm_phase3e/CLEARSCOPE_E5_V33B_FULL_PHASE3E_BASE.pkl
test -f outputs/models/phase3g_action_heads/CLEARSCOPE_E5_V33B_FULL_PHASE3G_CONDITIONAL_HEAD.pkl
test -f outputs/results/tflr_light/CLEARSCOPE_E5_V33B_FULL_PIPELINE/online_event_alerts.csv
test -f outputs/results/tflr_light/CLEARSCOPE_E5_V33B_FULL_PIPELINE/metrics.json
test -f outputs/results/tflr_light/CLEARSCOPE_E5_V33B_FULL_PIPELINE/eval_causal_semantics_slim.json
```

Expected: all checks exit `0`.

- [ ] **Step 3: Check provenance rejects E3 and bounded artifacts**

Run:

```bash
rg -n "CLEARSCOPE_E3|CLEARSCOPE_E5_V33B_BOUNDED|CLEARSCOPE_E5_v33b_bounded" \
  outputs/models/residual_word2vec/CLEARSCOPE_E5_V33B_FULL_LATENT64_word2vec_window3.json \
  outputs/models/residual_word2vec/CLEARSCOPE_E5_V33B_FULL_LATENT64/ \
  outputs/cache/phase3e/event_indices/CLEARSCOPE_E5_v33b_full/ \
  outputs/results/tflr_light/CLEARSCOPE_E5_V33B_FULL_PIPELINE/ \
  logs/clearscope_e5_v33b_full/
```

Expected: no matches, exit code `1`.

Run:

```bash
rg -n "CLEARSCOPE_E5|raw_detail_v33b_e5_android_safe|CLEARSCOPE_E5_V33B_FULL" \
  outputs/models/residual_word2vec/CLEARSCOPE_E5_V33B_FULL_LATENT64_word2vec_window3.json \
  outputs/models/residual_word2vec/CLEARSCOPE_E5_V33B_FULL_LATENT64/word2vec_training_summary.json \
  outputs/cache/phase3e/event_indices/CLEARSCOPE_E5_v33b_full/ \
  outputs/results/tflr_light/CLEARSCOPE_E5_V33B_FULL_PIPELINE/ \
  logs/clearscope_e5_v33b_full/
```

Expected: matches exist.

- [ ] **Step 4: Check alert schema and leakage flags**

Run:

```bash
python3 - <<'PY'
import csv
import json
from pathlib import Path

root = Path("outputs/results/tflr_light/CLEARSCOPE_E5_V33B_FULL_PIPELINE")
with (root / "online_event_alerts.csv").open(newline="", encoding="utf-8") as handle:
    reader = csv.reader(handle)
    header = next(reader)
required_any = [
    ("event", "event_id", "event_index", "event_idx", "idx"),
    ("score", "event_score", "residual_score", "final_score"),
    ("threshold", "event_threshold", "threshold_basis"),
    ("entity", "node", "subject", "object", "src_idx", "dst_idx", "info_src", "info_dst"),
]
missing = [group for group in required_any if not any(name in header for name in group)]
if missing:
    raise SystemExit(f"missing alert header groups: {missing}; header={header}")
eval_payload = json.loads((root / "eval_causal_semantics_slim.json").read_text(encoding="utf-8"))
if not eval_payload.get("ground_truth_used_only_for_evaluation"):
    raise SystemExit("ground_truth_used_only_for_evaluation flag is not true")
if not eval_payload.get("labels_attached_after_streaming"):
    raise SystemExit("labels_attached_after_streaming flag is not true")
config = eval_payload.get("config", {})
if config.get("semantic_mode") != "raw_detail_v33b_e5_android_safe":
    raise SystemExit(f"unexpected semantic_mode: {config.get('semantic_mode')}")
print(
    {
        "dataset": eval_payload.get("dataset"),
        "out_tag": eval_payload.get("out_tag"),
        "semantic_mode": config.get("semantic_mode"),
        "header_count": len(header),
    },
)
PY
```

Expected: exits `0`.

## Task 4: Final Verification

**Files:**
- No source files expected.

- [ ] **Step 1: Run focused tests and syntax checks**

Run:

```bash
python3 -m unittest tests.test_clearscope_e5_full_runner tests.test_clearscope_e5_bounded_runner tests.test_residual_word2vec_corpus_file -v
python3 -m py_compile legacy/tools/train_residual_word2vec_models.py
bash -n scripts/run/run_clearscope_e5_v33b_full_phase3e_phase3g.sh
```

Expected: all checks pass.

- [ ] **Step 2: Report go/no-go**

Report:

```text
E5_v33b_full_pipeline_passed
```

only if every Task 3 and Task 4 check passed.

