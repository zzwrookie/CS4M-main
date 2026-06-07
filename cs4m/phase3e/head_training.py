from __future__ import annotations

import time
from typing import Any

import numpy as np

from cs4m.phase3e.event_index import EVENT_INDEX_DTYPE


def _resolve_torch_device(torch_module: Any, requested: str) -> tuple[str, bool, str]:
    cuda_available = bool(torch_module.cuda.is_available())
    requested_device = str(requested).strip().lower()
    if requested_device == "auto":
        if cuda_available:
            return "cuda", True, str(torch_module.cuda.get_device_name(0))
        return "cpu", False, ""
    if requested_device == "cuda":
        if not cuda_available:
            raise ValueError("SSPM_TORCH_DEVICE=cuda requested but CUDA is unavailable")
        return "cuda", True, str(torch_module.cuda.get_device_name(0))
    if requested_device in ("cpu", "torch_cpu"):
        device_name = str(torch_module.cuda.get_device_name(0)) if cuda_available else ""
        return "cpu", cuda_available, device_name
    raise ValueError(f"unsupported torch device: {requested}")


def _validate_training_inputs(
    *,
    model: Any,
    x_context: np.ndarray,
    event_index: np.ndarray,
    node_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
    epochs: int,
    batch_events: int,
) -> None:
    if not hasattr(model, "set_lowrank_weights"):
        raise TypeError("model must provide set_lowrank_weights")
    if len(getattr(x_context, "shape", ())) != 2:
        raise ValueError("X_context must be a 2-D array or memmap")
    if int(x_context.shape[1]) != int(model.context_dim):
        raise ValueError(f"X_context must have shape (n, {int(model.context_dim)})")
    if len(getattr(event_index, "shape", ())) != 1:
        raise ValueError("event_index must be a one-dimensional array or memmap")
    if np.dtype(event_index.dtype) != EVENT_INDEX_DTYPE:
        raise TypeError(f"event_index must use EVENT_INDEX_DTYPE, got {event_index.dtype}")
    if int(x_context.shape[0]) != int(event_index.shape[0]):
        raise ValueError("X_context and event_index row count mismatch")
    if np.asarray(node_embeddings).ndim != 2:
        raise ValueError("node_embeddings must be a 2-D array")
    if np.asarray(action_embeddings).ndim != 2:
        raise ValueError("action_embeddings must be a 2-D array")
    if int(node_embeddings.shape[1]) != int(model.latent_dim):
        raise ValueError(f"node_embeddings must have shape (n, {int(model.latent_dim)})")
    if int(action_embeddings.shape[1]) != int(model.latent_dim):
        raise ValueError(f"action_embeddings must have shape (n, {int(model.latent_dim)})")
    if int(epochs) <= 0:
        raise ValueError("epochs must be positive")
    if int(batch_events) <= 0:
        raise ValueError("batch_events must be positive")


def _target_batch(
    event_batch: np.ndarray,
    node_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
) -> np.ndarray:
    src_indices = event_batch["src_node_idx"].astype(np.int64, copy=False)
    dst_indices = event_batch["dst_node_idx"].astype(np.int64, copy=False)
    action_indices = event_batch["action_id"].astype(np.int64, copy=False)
    if np.any(src_indices < 0) or np.any(src_indices >= int(node_embeddings.shape[0])):
        raise IndexError("src_node_idx out of bounds")
    if np.any(dst_indices < 0) or np.any(dst_indices >= int(node_embeddings.shape[0])):
        raise IndexError("dst_node_idx out of bounds")
    if np.any(action_indices < 0) or np.any(action_indices >= int(action_embeddings.shape[0])):
        raise IndexError("action_id out of bounds")
    src = np.asarray(node_embeddings[src_indices], dtype=np.float32)
    dst = np.asarray(node_embeddings[dst_indices], dtype=np.float32)
    action = np.asarray(action_embeddings[action_indices], dtype=np.float32)
    return ((src + dst + action) / np.float32(3.0)).astype(np.float32, copy=False)


def _batch_to_float32(array: np.ndarray, start: int, end: int, name: str) -> np.ndarray:
    batch = np.array(array[start:end], dtype=np.float32, copy=True)
    if not bool(np.all(np.isfinite(batch))):
        raise ValueError(f"{name} must contain only finite values")
    return batch


def train_lowrank_head_torch(
    *,
    model: Any,
    x_context: np.ndarray,
    event_index: np.ndarray,
    node_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
    epochs: int,
    batch_events: int,
    lr: float,
    weight_decay: float,
    device: str = "auto",
    patience: int | None = None,
    min_delta: float | None = None,
) -> dict[str, Any]:
    """Train only W1/W2/b from Phase3E X_context and event-index batches."""
    import torch

    _validate_training_inputs(
        model=model,
        x_context=x_context,
        event_index=event_index,
        node_embeddings=node_embeddings,
        action_embeddings=action_embeddings,
        epochs=int(epochs),
        batch_events=int(batch_events),
    )
    torch_device, cuda_available, device_name = _resolve_torch_device(torch, str(device))
    if torch_device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    learning_rate = float(lr)
    if learning_rate <= 0.0:
        raise ValueError("lr must be positive")
    decay = float(weight_decay)
    if decay < 0.0:
        raise ValueError("weight_decay must be non-negative")

    c1 = torch.nn.Parameter(torch.tensor(model.c1, dtype=torch.float32, device=torch_device))
    c2 = torch.nn.Parameter(torch.tensor(model.c2, dtype=torch.float32, device=torch_device))
    bias = torch.nn.Parameter(
        torch.tensor(model.bias, dtype=torch.float32, device=torch_device),
    )
    optimizer = torch.optim.Adam([c1, c2, bias], lr=learning_rate, weight_decay=decay)
    loss_by_epoch: list[float] = []
    best_loss = float("inf")
    plateau_epochs = 0
    effective_patience = (
        int(model.config.early_stop_patience) if patience is None else int(patience)
    )
    effective_min_delta = (
        float(model.config.early_stop_min_delta) if min_delta is None else float(min_delta)
    )
    early_stop_reason = ""
    started = time.perf_counter()
    row_count = int(x_context.shape[0])

    for _epoch in range(int(epochs)):
        epoch_loss = 0.0
        epoch_batches = 0
        for start in range(0, row_count, int(batch_events)):
            end = min(start + int(batch_events), row_count)
            x_np = _batch_to_float32(x_context, start, end, "X_context")
            y_np = _target_batch(event_index[start:end], node_embeddings, action_embeddings)
            if not bool(np.all(np.isfinite(y_np))):
                raise ValueError("target batch must contain only finite values")
            x = torch.as_tensor(x_np, dtype=torch.float32, device=torch_device)
            y = torch.as_tensor(y_np, dtype=torch.float32, device=torch_device)
            pred = x @ c1 @ c2 + bias
            pred_norm = torch.nn.functional.normalize(pred, p=2.0, dim=1, eps=1e-8)
            y_norm = torch.nn.functional.normalize(y, p=2.0, dim=1, eps=1e-8)
            loss_cos = torch.mean(1.0 - torch.sum(pred_norm * y_norm, dim=1))
            loss_mse = torch.mean((pred - y) ** 2) * float(model.config.lambda_mse)
            loss = loss_cos + loss_mse
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([c1, c2, bias], 5.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
            epoch_batches += 1

        mean_loss = float(epoch_loss / max(epoch_batches, 1))
        loss_by_epoch.append(mean_loss)
        if best_loss - mean_loss > effective_min_delta:
            best_loss = mean_loss
            plateau_epochs = 0
        else:
            plateau_epochs += 1
            if effective_patience > 0 and plateau_epochs >= effective_patience:
                early_stop_reason = "loss_plateau"
                break

    if torch_device == "cuda":
        torch.cuda.synchronize()
        peak_gpu_mb = float(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
    else:
        peak_gpu_mb = 0.0

    model.set_lowrank_weights(
        c1.detach().cpu().numpy(),
        c2.detach().cpu().numpy(),
        bias.detach().cpu().numpy(),
    )
    if not loss_by_epoch:
        best_loss = 0.0
    return {
        "sspm_train_backend": "torch" if torch_device == "cuda" else "torch_cpu",
        "torch_device": torch_device,
        "torch_cuda_available": bool(cuda_available),
        "torch_device_name": str(device_name),
        "torch_optimizer": "Adam",
        "torch_lr": learning_rate,
        "torch_weight_decay": decay,
        "torch_batch_events": int(batch_events),
        "torch_epochs_completed": int(len(loss_by_epoch)),
        "torch_train_loss_best": float(best_loss),
        "torch_early_stop_reason": str(early_stop_reason),
        "torch_train_time_sec": float(time.perf_counter() - started),
        "torch_peak_gpu_memory_mb": float(peak_gpu_mb),
        "loss_by_epoch": loss_by_epoch,
        "torch_early_stop_patience": int(effective_patience),
        "torch_early_stop_min_delta": float(effective_min_delta),
    }
