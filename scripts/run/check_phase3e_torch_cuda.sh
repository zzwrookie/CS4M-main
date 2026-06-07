#!/usr/bin/env bash
set -euo pipefail

# Read-only PyTorch CUDA smoke check for Phase3E preflight.

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" - <<'PY'
import torch

print("torch_version", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
print("device_count", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("cuda_available False")

print("device_name", torch.cuda.get_device_name(0))
x = torch.randn(1024, 146, device="cuda")
w = torch.randn(146, 32, device="cuda")
y = x @ w
torch.cuda.synchronize()
print("cuda_matmul_ok", tuple(y.shape))
PY
