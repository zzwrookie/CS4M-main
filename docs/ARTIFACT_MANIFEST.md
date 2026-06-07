# CS4M Artifact Manifest

This repository intentionally does not commit large generated artifacts. Restore them on each
server from backups, object storage, or a local artifact bundle before running non-dry-run
reproduction commands.

## Not Committed

The following are ignored by `.gitignore`:

- `outputs/cache/`
- `outputs/results/`
- `logs/`
- `tmp/`
- `*.memmap`
- `*.pkl`, `*.npy`, `*.npz`
- checkpoint/model files such as `*.pt`, `*.pth`, `*.ckpt`
- `.env`, private keys, passwords, and local credentials

## Required Artifact Groups

CADETS_E3 best reproduction typically needs:

- residual Word2Vec model under `outputs/models/residual_word2vec/`;
- SSPM Phase3E checkpoints under `outputs/models/sspm_phase3e/`;
- Phase3G action/conditional head checkpoints under `outputs/models/phase3g_action_heads*/`;
- Phase3E caches under `outputs/cache/phase3e/`;
- Phase3G action-validation caches under `outputs/cache/phase3g_action_validation/`;
- optional endpoint-suppression caches under `outputs/cache/phase3g_endpoint_suppression*/`;
- historical result directories under `outputs/results/` if comparing backed-up metrics.

THEIA_E3 best reproduction typically needs:

- THEIA residual Word2Vec model;
- THEIA_E3 E4 SSPM Phase3E checkpoint;
- THEIA_E3 E4 Phase3G conditional head reference checkpoint;
- THEIA Phase3E caches;
- THEIA action-validation conditional train memmaps, including:

```text
outputs/cache/phase3g_action_validation/THEIA_E3/conditional_train_memmaps/X_conditional_s4d_complex_node.memmap
outputs/cache/phase3g_action_validation/THEIA_E3/conditional_train_memmaps/Y_conditional_s4d_complex_node.memmap
outputs/cache/phase3g_action_validation/THEIA_E3/conditional_train_memmaps/target_case_s4d_complex_node.memmap
outputs/cache/phase3g_action_validation/THEIA_E3/conditional_train_memmaps/conditional_memmap_s4d_complex_node_meta.json
```

## Portable Location Overrides

Use environment variables rather than editing source:

```bash
export CS4M_ROOT="$(pwd)"
export RESULT_ROOT="${CS4M_ROOT}/outputs/results/tflr_light"
export PHASE3E_CACHE_ROOT="${CS4M_ROOT}/outputs/cache/phase3e"
export ACTION_VALIDATION_CACHE_DIR="${CS4M_ROOT}/outputs/cache/phase3g_action_validation/<DATASET>"
export CONDITIONAL_ENDPOINT_SUPPRESSION_CACHE_DIR="${CS4M_ROOT}/outputs/cache/phase3g_endpoint_suppression/<DATASET>"
```

Database settings must also come from environment variables:

```bash
export CLAD_DB_HOST="127.0.0.1"
export CLAD_DB_PORT="5433"
export CLAD_DB_USER="postgres"
export CLAD_DB_PASSWORD="<set-outside-repo>"
```

Do not commit real credentials or server-specific artifact paths.
