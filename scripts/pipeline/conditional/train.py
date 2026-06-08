"""Phase3G conditional train memmap creation and conditional head training."""

from __future__ import annotations

from scripts.pipeline.config.runtime_config import *


def _phase3g_build_conditional_train_memmap(
    *,
    config: SlimConfig,
    train_index: np.ndarray,
    node_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
    paths: Mapping[str, Path],
    event_meta: Mapping[str, Any],
    input_dim: int,
    output_dim: int,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Build or validate train-only conditional X/Y/target_case memmaps."""
    records = _phase3e_event_index_array(train_index)
    count = int(records.shape[0])
    memmap_paths = _phase3g_conditional_memmap_paths(config)
    memmap_paths["dir"].mkdir(parents=True, exist_ok=True)
    fingerprint = _phase3g_conditional_memmap_fingerprint(
        config=config,
        paths=paths,
        event_meta=event_meta,
        input_dim=input_dim,
    )
    existing = _phase3g_load_conditional_memmap_meta(memmap_paths["meta"])
    if existing and not bool(force_rebuild):
        _phase3g_validate_conditional_memmap(
            meta=existing,
            fingerprint=fingerprint,
            count=count,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        for key in ("x", "y", "target_case"):
            if not memmap_paths[key].exists():
                raise FileNotFoundError(f"conditional memmap sidecar missing: {memmap_paths[key]}")
        return dict(existing)

    started = time.perf_counter()
    rss_peak = _current_rss_mb()
    if str(config.sspm_checkpoint_path).strip():
        model, _ = load_sspm_checkpoint_state_for_conditional(
            config.sspm_checkpoint_path,
            config,
        )
    else:
        model = SSPMLowRankModel(_make_sspm_config(config, _load_process_config(config)))
    model.reset_state()
    type_eye = _phase3g_type_eye()
    x_map = np.memmap(
        memmap_paths["x"],
        dtype=np.float32,
        mode="w+",
        shape=(count, int(input_dim)),
    )
    y_map = np.memmap(
        memmap_paths["y"],
        dtype=np.float32,
        mode="w+",
        shape=(count, int(output_dim)),
    )
    case_map = np.memmap(
        memmap_paths["target_case"],
        dtype=np.int8,
        mode="w+",
        shape=(count,),
    )
    target_case_counts = {EVENT_SEMANTIC_TARGET: 0, BOTH_COLD_ACTION_TARGET: 0}
    chunk_size = max(int(config.sspm_torch_batch_events), 1)
    _stage_log(
        config,
        "phase3g_conditional_memmap_build_start",
        mode="conditional_memmap",
        state_model=str(config.sspm_state_model),
        count=count,
        x_path=str(memmap_paths["x"]),
        y_path=str(memmap_paths["y"]),
    )
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        for offset, row in enumerate(records[start:end], start=start):
            z_state = _phase3e_target_from_index_row(row, node_embeddings, action_embeddings)
            context, src_type_name, dst_type_name, src_has_state, dst_has_state = (
                _phase3g_action_context_from_model(
                    model,
                    row,
                    node_embeddings,
                    type_eye,
                )
            )
            target, target_case = select_conditional_target(
                row,
                node_embeddings,
                action_embeddings,
                src_has_state=src_has_state,
                dst_has_state=dst_has_state,
            )
            x_map[offset] = context
            y_map[offset] = target
            case_map[offset] = np.int8(target_case_id(target_case))
            target_case_counts[target_case] = target_case_counts.get(target_case, 0) + 1
            action = str(ORTHRUS10_ACTION_NAMES[int(row["action_id"])])
            update_residual_score = 0.0
            if _phase3g_update_gate_enabled(config, model):
                fields = _phase3e_fields_from_index_row(row)
                pred_state = model.predict(fields)
                update_residual_score = float(
                    model.residual_score(
                        pred_state,
                        z_state,
                        action=fields.get("raw_action", fields.get("action")),
                    ),
                )
            _phase3f_e2_none_update_from_numeric_row(
                model,
                row,
                z_state,
                action,
                src_type_name,
                dst_type_name,
                residual_score=update_residual_score,
            )
        rss_peak = max(rss_peak, _current_rss_mb())
        if int(config.progress_interval_events) > 0 and end % int(config.progress_interval_events) == 0:
            _stage_log(
                config,
                "phase3g_conditional_memmap_build_progress",
                count=end,
                rss_mb=f"{rss_peak:.3f}",
            )
    x_map.flush()
    y_map.flush()
    case_map.flush()
    elapsed = float(time.perf_counter() - started)
    meta = {
        "schema": "phase3g_conditional_train_memmap_v1",
        "train_data_mode": "conditional_memmap",
        "state_model": str(config.sspm_state_model),
        "num_events": int(count),
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
        "target_case_counts": {key: int(value) for key, value in target_case_counts.items()},
        "x_path": str(memmap_paths["x"]),
        "y_path": str(memmap_paths["y"]),
        "target_case_path": str(memmap_paths["target_case"]),
        "x_size_mb": float(memmap_paths["x"].stat().st_size / 1024.0 / 1024.0),
        "y_size_mb": float(memmap_paths["y"].stat().st_size / 1024.0 / 1024.0),
        "target_case_size_mb": float(memmap_paths["target_case"].stat().st_size / 1024.0 / 1024.0),
        "build_time_sec": elapsed,
        "build_events_per_sec": float(count / max(elapsed, 1e-9)),
        "build_rss_peak_mb": float(rss_peak),
        "fingerprint": fingerprint,
    }
    memmap_paths["meta"].write_text(
        json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    _stage_log(
        config,
        "phase3g_conditional_memmap_build_end",
        count=count,
        events_per_sec=f"{float(meta['build_events_per_sec']):.3f}",
        rss_peak_mb=f"{rss_peak:.3f}",
    )
    return meta


def _phase3g_train_conditional_head_torch_from_memmap(
    *,
    config: SlimConfig,
    head: ConditionalSemanticHead,
    memmap_meta: Mapping[str, Any],
    max_epochs: int,
) -> dict[str, Any]:
    """Train the conditional head by batch-reading prebuilt X/Y memmaps."""
    import torch

    device, cuda_available, device_name = _resolve_phase3g_torch_device(
        torch,
        str(config.sspm_torch_device),
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if head.is_dual_head:
        event_w1 = torch.nn.Parameter(
            torch.tensor(head.event_w1, dtype=torch.float32, device=device),
        )
        event_w2 = torch.nn.Parameter(
            torch.tensor(head.event_w2, dtype=torch.float32, device=device),
        )
        event_bias = torch.nn.Parameter(
            torch.tensor(head.event_bias, dtype=torch.float32, device=device),
        )
        action_w1 = torch.nn.Parameter(
            torch.tensor(head.action_w1, dtype=torch.float32, device=device),
        )
        action_w2 = torch.nn.Parameter(
            torch.tensor(head.action_w2, dtype=torch.float32, device=device),
        )
        action_bias = torch.nn.Parameter(
            torch.tensor(head.action_bias, dtype=torch.float32, device=device),
        )
        train_parameters = [
            event_w1,
            event_w2,
            event_bias,
            action_w1,
            action_w2,
            action_bias,
        ]
    else:
        w1 = torch.nn.Parameter(torch.tensor(head.w1, dtype=torch.float32, device=device))
        w2 = torch.nn.Parameter(torch.tensor(head.w2, dtype=torch.float32, device=device))
        bias = torch.nn.Parameter(torch.tensor(head.bias, dtype=torch.float32, device=device))
        train_parameters = [w1, w2, bias]
    lr = _phase3e_train_lr(config)
    optimizer = torch.optim.Adam(
        train_parameters,
        lr=float(lr),
        weight_decay=float(config.sspm_torch_weight_decay),
    )
    count = int(memmap_meta["num_events"])
    input_dim = int(memmap_meta["input_dim"])
    output_dim = int(memmap_meta["output_dim"])
    x_map = np.memmap(
        str(memmap_meta["x_path"]),
        dtype=np.float32,
        mode="r",
        shape=(count, input_dim),
    )
    y_map = np.memmap(
        str(memmap_meta["y_path"]),
        dtype=np.float32,
        mode="r",
        shape=(count, output_dim),
    )
    case_map = np.memmap(
        str(memmap_meta["target_case_path"]),
        dtype=np.int8,
        mode="r",
        shape=(count,),
    )
    chunk_size = max(int(config.sspm_torch_batch_events), 1)
    best_loss = float("inf")
    loss_by_epoch: list[float] = []
    loss_event_semantic_by_epoch: list[float] = []
    loss_both_cold_action_by_epoch: list[float] = []
    plateau_epochs = 0
    early_stop_reason = ""
    started = time.perf_counter()
    _stage_log(
        config,
        "phase3g_conditional_memmap_train_start",
        train_data_mode="conditional_memmap",
        count=count,
        epochs=int(max_epochs),
        batch_events=chunk_size,
        conditional_head_arch=str(config.sspm_conditional_head_arch),
        conditional_memmap_path=str(memmap_meta["x_path"]),
        conditional_memmap_fingerprint=str(
            dict(memmap_meta.get("fingerprint", {})).get("fingerprint_sha256", ""),
        ),
    )

    def _row_loss(pred: Any, target: Any) -> Any:
        if str(config.conditional_semantic_loss) == "mse":
            return torch.mean((pred - target) ** 2, dim=1)
        pred_norm = torch.nn.functional.normalize(pred, dim=1, eps=1e-8)
        y_norm = torch.nn.functional.normalize(target, dim=1, eps=1e-8)
        return 1.0 - torch.sum(pred_norm * y_norm, dim=1)

    for _epoch in range(max(int(max_epochs), 1)):
        epoch_loss = 0.0
        epoch_batches = 0
        epoch_event_loss_sum = 0.0
        epoch_action_loss_sum = 0.0
        epoch_event_count = 0
        epoch_action_count = 0
        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            x = torch.as_tensor(
                np.asarray(x_map[start:end]).copy(),
                dtype=torch.float32,
                device=device,
            )
            y = torch.as_tensor(
                np.asarray(y_map[start:end]).copy(),
                dtype=torch.float32,
                device=device,
            )
            if head.is_dual_head:
                case_ids = torch.as_tensor(
                    np.asarray(case_map[start:end]).copy(),
                    dtype=torch.int64,
                    device=device,
                )
                batch_losses: list[Any] = []
                event_mask = case_ids == int(EVENT_SEMANTIC_TARGET_ID)
                action_mask = case_ids == int(BOTH_COLD_ACTION_TARGET_ID)
                if bool(torch.any(event_mask).detach().cpu().item()):
                    event_pred = x[event_mask] @ event_w1 @ event_w2 + event_bias
                    event_losses = _row_loss(event_pred, y[event_mask])
                    batch_losses.append(event_losses)
                    event_count = int(event_losses.shape[0])
                    epoch_event_loss_sum += float(
                        torch.sum(event_losses).detach().cpu().item(),
                    )
                    epoch_event_count += event_count
                if bool(torch.any(action_mask).detach().cpu().item()):
                    action_pred = x[action_mask] @ action_w1 @ action_w2 + action_bias
                    action_losses = _row_loss(action_pred, y[action_mask])
                    batch_losses.append(action_losses)
                    action_count = int(action_losses.shape[0])
                    epoch_action_loss_sum += float(
                        torch.sum(action_losses).detach().cpu().item(),
                    )
                    epoch_action_count += action_count
                if not batch_losses:
                    raise ValueError("conditional memmap batch has no supported target cases")
                loss = torch.mean(torch.cat(batch_losses, dim=0))
            else:
                pred = x @ w1 @ w2 + bias
                row_losses = _row_loss(pred, y)
                loss = torch.mean(row_losses)
                cases_np = np.asarray(case_map[start:end])
                event_mask_np = cases_np == int(EVENT_SEMANTIC_TARGET_ID)
                action_mask_np = cases_np == int(BOTH_COLD_ACTION_TARGET_ID)
                row_losses_np = row_losses.detach().cpu().numpy()
                if np.any(event_mask_np):
                    epoch_event_loss_sum += float(np.sum(row_losses_np[event_mask_np]))
                    epoch_event_count += int(np.count_nonzero(event_mask_np))
                if np.any(action_mask_np):
                    epoch_action_loss_sum += float(np.sum(row_losses_np[action_mask_np]))
                    epoch_action_count += int(np.count_nonzero(action_mask_np))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_parameters, 5.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
            epoch_batches += 1
        mean_loss = float(epoch_loss / max(epoch_batches, 1))
        loss_by_epoch.append(mean_loss)
        event_loss = float(epoch_event_loss_sum / max(epoch_event_count, 1))
        action_loss = float(epoch_action_loss_sum / max(epoch_action_count, 1))
        loss_event_semantic_by_epoch.append(event_loss)
        loss_both_cold_action_by_epoch.append(action_loss)
        if best_loss - mean_loss > float(config.sspm_early_stop_min_delta):
            best_loss = mean_loss
            plateau_epochs = 0
        else:
            plateau_epochs += 1
            if int(config.sspm_early_stop_patience) > 0 and plateau_epochs >= int(
                config.sspm_early_stop_patience,
            ):
                early_stop_reason = "loss_plateau"
                break
        _stage_log(
            config,
            "phase3g_conditional_memmap_train_epoch",
            epoch=len(loss_by_epoch),
            loss=f"{mean_loss:.6f}",
            loss_event_semantic=f"{event_loss:.6f}",
            loss_both_cold_action=f"{action_loss:.6f}",
            event_semantic=int(
                dict(memmap_meta.get("target_case_counts", {})).get(EVENT_SEMANTIC_TARGET, 0),
            ),
            both_cold=int(
                dict(memmap_meta.get("target_case_counts", {})).get(BOTH_COLD_ACTION_TARGET, 0),
            ),
        )
    if device == "cuda":
        torch.cuda.synchronize()
        peak_gpu_mb = float(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
    else:
        peak_gpu_mb = 0.0
    if head.is_dual_head:
        head.event_w1 = event_w1.detach().cpu().numpy().astype(np.float32, copy=True)
        head.event_w2 = event_w2.detach().cpu().numpy().astype(np.float32, copy=True)
        head.event_bias = event_bias.detach().cpu().numpy().astype(np.float32, copy=True)
        head.action_w1 = action_w1.detach().cpu().numpy().astype(np.float32, copy=True)
        head.action_w2 = action_w2.detach().cpu().numpy().astype(np.float32, copy=True)
        head.action_bias = action_bias.detach().cpu().numpy().astype(np.float32, copy=True)
        head.w1 = head.event_w1
        head.w2 = head.event_w2
        head.bias = head.event_bias
    else:
        head.w1 = w1.detach().cpu().numpy().astype(np.float32, copy=True)
        head.w2 = w2.detach().cpu().numpy().astype(np.float32, copy=True)
        head.bias = bias.detach().cpu().numpy().astype(np.float32, copy=True)
    del x_map
    del y_map
    del case_map
    final_case_summary = conditional_case_loss_summary(
        np.asarray(
            [
                loss_event_semantic_by_epoch[-1] if loss_event_semantic_by_epoch else 0.0,
                loss_both_cold_action_by_epoch[-1] if loss_both_cold_action_by_epoch else 0.0,
            ],
            dtype=np.float32,
        ),
        np.asarray(
            [EVENT_SEMANTIC_TARGET_ID, BOTH_COLD_ACTION_TARGET_ID],
            dtype=np.int8,
        ),
    )
    return {
        "score_head": str(config.sspm_score_head),
        "conditional_head_arch": str(config.sspm_conditional_head_arch),
        "head_arch": str(config.sspm_conditional_head_arch),
        "node_repr_fusion": str(config.node_repr_fusion),
        "loss_type": str(config.conditional_semantic_loss),
        "target_case_mode": "conditional_cold_action_else_event",
        "target_case_head_mode": str(head.config.target_case_head_mode),
        "target": "conditional_target",
        "target_case_counts": dict(memmap_meta.get("target_case_counts", {})),
        "count_event_semantic": int(
            dict(memmap_meta.get("target_case_counts", {})).get(EVENT_SEMANTIC_TARGET, 0),
        ),
        "count_both_cold_action": int(
            dict(memmap_meta.get("target_case_counts", {})).get(BOTH_COLD_ACTION_TARGET, 0),
        ),
        "input_dim": int(head.config.input_dim),
        "context_dim": int(head.config.input_dim),
        "rank": int(head.config.rank),
        "output_dim": int(head.config.output_dim),
        "train_data_mode": "conditional_memmap",
        "conditional_train_data_mode": "memmap",
        "conditional_memmap_path": str(memmap_meta["x_path"]),
        "conditional_memmap_y_path": str(memmap_meta["y_path"]),
        "conditional_memmap_target_case_path": str(memmap_meta["target_case_path"]),
        "conditional_memmap_fingerprint": dict(memmap_meta.get("fingerprint", {})),
        "conditional_memmap_size_mb": float(
            float(memmap_meta.get("x_size_mb", 0.0))
            + float(memmap_meta.get("y_size_mb", 0.0))
            + float(memmap_meta.get("target_case_size_mb", 0.0)),
        ),
        "torch_device": device,
        "torch_cuda_available": bool(cuda_available),
        "torch_device_name": str(device_name),
        "torch_optimizer": "Adam",
        "torch_lr": float(lr),
        "torch_weight_decay": float(config.sspm_torch_weight_decay),
        "torch_batch_events": int(config.sspm_torch_batch_events),
        "torch_epochs_completed": int(len(loss_by_epoch)),
        "torch_train_loss_best": float(best_loss if loss_by_epoch else 0.0),
        "torch_train_loss_final": float(loss_by_epoch[-1] if loss_by_epoch else 0.0),
        "epochs_completed": int(len(loss_by_epoch)),
        "best_loss": float(best_loss if loss_by_epoch else 0.0),
        "final_loss": float(loss_by_epoch[-1] if loss_by_epoch else 0.0),
        "loss_event_semantic": float(
            loss_event_semantic_by_epoch[-1] if loss_event_semantic_by_epoch else 0.0,
        ),
        "loss_both_cold_action": float(
            loss_both_cold_action_by_epoch[-1] if loss_both_cold_action_by_epoch else 0.0,
        ),
        "loss_event_semantic_by_epoch": loss_event_semantic_by_epoch,
        "loss_both_cold_action_by_epoch": loss_both_cold_action_by_epoch,
        "case_loss_summary": final_case_summary,
        "torch_early_stop_reason": str(early_stop_reason),
        "torch_train_time_sec": float(time.perf_counter() - started),
        "torch_peak_gpu_memory_mb": float(peak_gpu_mb),
        "loss_by_epoch": loss_by_epoch,
        "train_events_actual": int(count),
        "train_events_seen_total": int(count) * int(len(loss_by_epoch)),
    }



def _phase3g_train_conditional_head_torch(
    *,
    config: SlimConfig,
    head: ConditionalSemanticHead,
    train_index: np.ndarray,
    node_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
) -> dict[str, Any]:
    import torch

    device, cuda_available, device_name = _resolve_phase3g_torch_device(
        torch,
        str(config.sspm_torch_device),
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    w1 = torch.nn.Parameter(torch.tensor(head.w1, dtype=torch.float32, device=device))
    w2 = torch.nn.Parameter(torch.tensor(head.w2, dtype=torch.float32, device=device))
    bias = torch.nn.Parameter(torch.tensor(head.bias, dtype=torch.float32, device=device))
    lr = _phase3e_train_lr(config)
    optimizer = torch.optim.Adam(
        [w1, w2, bias],
        lr=float(lr),
        weight_decay=float(config.sspm_torch_weight_decay),
    )
    records = _phase3e_event_index_array(train_index)
    row_count = int(records.shape[0])
    chunk_size = max(int(config.sspm_torch_batch_events), 1)
    best_loss = float("inf")
    loss_by_epoch: list[float] = []
    is_action_embedding_head = str(config.sspm_score_head) == CONDITIONAL_ACTION_EMBEDDING_SCORE_HEAD
    target_case_totals = (
        {"action_embedding_target": 0}
        if is_action_embedding_head
        else {
            EVENT_SEMANTIC_TARGET: 0,
            BOTH_COLD_ACTION_TARGET: 0,
        }
    )
    plateau_epochs = 0
    early_stop_reason = ""
    started = time.perf_counter()
    for _epoch in range(max(int(config.sspm_epochs), 1)):
        model = SSPMLowRankModel(_make_sspm_config(config, _load_process_config(config)))
        model.reset_state()
        type_eye = _phase3g_type_eye()
        epoch_loss = 0.0
        epoch_batches = 0
        epoch_cases = (
            {"action_embedding_target": 0}
            if is_action_embedding_head
            else {
                EVENT_SEMANTIC_TARGET: 0,
                BOTH_COLD_ACTION_TARGET: 0,
            }
        )
        for start in range(0, row_count, chunk_size):
            end = min(start + chunk_size, row_count)
            chunk = records[start:end]
            contexts = np.zeros((int(chunk.shape[0]), int(head.config.input_dim)), dtype=np.float32)
            targets = np.zeros((int(chunk.shape[0]), int(head.config.output_dim)), dtype=np.float32)
            for offset, row in enumerate(chunk):
                z_state = _phase3e_target_from_index_row(row, node_embeddings, action_embeddings)
                context, src_type_name, dst_type_name, src_has_state, dst_has_state = (
                    _phase3g_action_context_from_model(
                        model,
                        row,
                        node_embeddings,
                        type_eye,
                    )
                )
                if is_action_embedding_head:
                    target = select_action_embedding_target(row, action_embeddings)
                    target_case = "action_embedding_target"
                else:
                    target, target_case = select_conditional_target(
                        row,
                        node_embeddings,
                        action_embeddings,
                        src_has_state=src_has_state,
                        dst_has_state=dst_has_state,
                    )
                contexts[offset] = context
                targets[offset] = target
                epoch_cases[target_case] += 1
                action = str(ORTHRUS10_ACTION_NAMES[int(row["action_id"])])
                _phase3f_e2_none_update_from_numeric_row(
                    model,
                    row,
                    z_state,
                    action,
                    src_type_name,
                    dst_type_name,
                    residual_score=0.0,
                )
            x = torch.as_tensor(contexts, dtype=torch.float32, device=device)
            y = torch.as_tensor(targets, dtype=torch.float32, device=device)
            pred = x @ w1 @ w2 + bias
            if str(config.conditional_semantic_loss) == "mse":
                loss = torch.mean((pred - y) ** 2)
            else:
                pred_norm = torch.nn.functional.normalize(pred, dim=1, eps=1e-8)
                y_norm = torch.nn.functional.normalize(y, dim=1, eps=1e-8)
                loss = torch.mean(1.0 - torch.sum(pred_norm * y_norm, dim=1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([w1, w2, bias], 5.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
            epoch_batches += 1
        for key, value in epoch_cases.items():
            target_case_totals[key] = target_case_totals.get(key, 0) + int(value)
        mean_loss = float(epoch_loss / max(epoch_batches, 1))
        loss_by_epoch.append(mean_loss)
        if best_loss - mean_loss > float(config.sspm_early_stop_min_delta):
            best_loss = mean_loss
            plateau_epochs = 0
        else:
            plateau_epochs += 1
            if int(config.sspm_early_stop_patience) > 0 and plateau_epochs >= int(
                config.sspm_early_stop_patience,
            ):
                early_stop_reason = "loss_plateau"
                break
        _stage_log(
            config,
            "phase3g_conditional_train_epoch",
            epoch=len(loss_by_epoch),
            loss=f"{mean_loss:.6f}",
            event_semantic=epoch_cases.get(EVENT_SEMANTIC_TARGET, 0),
            both_cold=epoch_cases.get(BOTH_COLD_ACTION_TARGET, 0),
            action_embedding=epoch_cases.get("action_embedding_target", 0),
        )
    if device == "cuda":
        torch.cuda.synchronize()
        peak_gpu_mb = float(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
    else:
        peak_gpu_mb = 0.0
    head.w1 = w1.detach().cpu().numpy().astype(np.float32, copy=True)
    head.w2 = w2.detach().cpu().numpy().astype(np.float32, copy=True)
    head.bias = bias.detach().cpu().numpy().astype(np.float32, copy=True)
    return {
        "score_head": str(config.sspm_score_head),
        "node_repr_fusion": str(config.node_repr_fusion),
        "loss_type": str(config.conditional_semantic_loss),
        "target_case_mode": (
            "action_embedding_only"
            if is_action_embedding_head
            else "conditional_cold_action_else_event"
        ),
        "target": "e_action" if is_action_embedding_head else "conditional_target",
        "target_case_counts": target_case_totals,
        "input_dim": int(head.config.input_dim),
        "rank": int(head.config.rank),
        "output_dim": int(head.config.output_dim),
        "torch_device": device,
        "torch_cuda_available": bool(cuda_available),
        "torch_device_name": str(device_name),
        "torch_optimizer": "Adam",
        "torch_lr": float(lr),
        "torch_weight_decay": float(config.sspm_torch_weight_decay),
        "torch_batch_events": int(config.sspm_torch_batch_events),
        "torch_epochs_completed": int(len(loss_by_epoch)),
        "torch_train_loss_best": float(best_loss if loss_by_epoch else 0.0),
        "torch_train_loss_final": float(loss_by_epoch[-1] if loss_by_epoch else 0.0),
        "torch_early_stop_reason": str(early_stop_reason),
        "torch_train_time_sec": float(time.perf_counter() - started),
        "torch_peak_gpu_memory_mb": float(peak_gpu_mb),
        "loss_by_epoch": loss_by_epoch,
        "train_events_actual": int(row_count),
    }


def _resolve_phase3g_torch_device(torch_module: Any, requested: str) -> tuple[str, bool, str]:
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
    if requested_device in {"cpu", "torch_cpu"}:
        return "cpu", cuda_available, str(torch_module.cuda.get_device_name(0)) if cuda_available else ""
    raise ValueError(f"unsupported torch device: {requested}")


def run_phase3g_action_train_from_precompute(config: SlimConfig) -> Path:
    """Reject historical action-predict head training from active code."""
    del config
    raise ValueError(
        "historical Phase3G action-predict head training moved to legacy; "
        "current best chains use conditional_action_semantic",
    )


def run_phase3g_conditional_train_from_precompute(config: SlimConfig) -> Path:
    """Train one Phase3G conditional vector head from Phase3E artifacts."""
    _validate_active_event_score_mode(config.event_score_mode)
    if str(config.sspm_train_mode) != "train_conditional_and_save":
        raise ValueError(
            f"{config.sspm_score_head} train requires "
            "SSPM_TRAIN_MODE=train_conditional_and_save",
        )
    started = time.perf_counter()
    output_dir = Path(config.result_root) / config.out_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    paths, event_meta = _phase3g_effective_artifacts(config)
    train_index, train_count = _phase3e_open_split_event_index(
        paths,
        event_meta,
        "train",
        max_events=int(config.max_train_events),
    )
    node_embeddings = _phase3g_open_node_embeddings(config, paths)
    action_embeddings = np.load(paths["action_embeddings"], mmap_mode="r")
    validation_count = int(
        dict(dict(event_meta.get("splits", {})).get("validation", {})).get("num_events", 0),
    )
    input_dim = _phase3g_action_input_dim(node_embeddings)
    if is_cadets_dataset(config.dataset) and input_dim != 136:
        raise ValueError(f"CADETS_E3 conditional input_dim must be 136, got {input_dim}")
    head = ConditionalSemanticHead(
        ConditionalSemanticHeadConfig(
            input_dim=input_dim,
            rank=int(config.rank),
            output_dim=int(config.sspm_target_dim),
            loss_type=str(config.conditional_semantic_loss),
            seed=29,
            node_repr_fusion=str(config.node_repr_fusion),
            conditional_head_arch=str(config.sspm_conditional_head_arch),
        ),
    )
    _stage_log(
        config,
        "phase3g_conditional_train_start",
        score_head=str(config.sspm_score_head),
        conditional_train_data_mode=str(config.sspm_conditional_train_data_mode),
        conditional_head_arch=str(config.sspm_conditional_head_arch),
        count=train_count,
        input_dim=input_dim,
        rank=int(config.rank),
        loss=str(config.conditional_semantic_loss),
    )
    conditional_memmap_meta: dict[str, Any] | None = None
    stream_e3_gamma = (
        str(config.sspm_state_model) == "real_diag_learnable"
        and bool(config.real_diag_train_gamma)
    )
    if (
        str(config.sspm_conditional_train_data_mode) == "memmap"
        and not stream_e3_gamma
        and str(config.sspm_score_head) == "conditional_action_semantic"
    ):
        conditional_memmap_meta = _phase3g_build_conditional_train_memmap(
            config=config,
            train_index=train_index,
            node_embeddings=node_embeddings,
            action_embeddings=action_embeddings,
            paths=paths,
            event_meta=event_meta,
            input_dim=input_dim,
            output_dim=int(config.sspm_target_dim),
        )
        if bool(config.phase3g_build_conditional_memmap_only):
            eval_payload = _phase3g_conditional_memmap_only_payload(
                config=config,
                output_dir=output_dir,
                memmap_meta=conditional_memmap_meta,
                train_count=int(train_count),
                validation_count=int(validation_count),
                elapsed_seconds=float(time.perf_counter() - started),
            )
            return _write_phase3g_conditional_memmap_only_outputs(
                config=config,
                output_dir=output_dir,
                eval_payload=eval_payload,
                train_count=int(train_count),
                validation_count=int(validation_count),
            )
        train_stats = _phase3g_train_conditional_head_torch_from_memmap(
            config=config,
            head=head,
            memmap_meta=conditional_memmap_meta,
            max_epochs=int(config.sspm_conditional_max_epochs),
        )
    elif str(config.sspm_conditional_train_data_mode) == "stream_event" or stream_e3_gamma:
        stream_sentinel = _phase3g_conditional_memmap_sentinel(config)
        if stream_e3_gamma:
            original_epochs = int(config.sspm_epochs)
            config.sspm_epochs = min(
                int(config.sspm_epochs),
                int(config.sspm_conditional_e3_max_epochs),
            )
            _stage_log(
                config,
                "phase3g_conditional_train_stream_event_e3_gamma",
                requested_epochs=original_epochs,
                effective_epochs=int(config.sspm_epochs),
            )
        try:
            train_stats = _phase3g_train_conditional_head_torch(
                config=config,
                head=head,
                train_index=train_index,
                node_embeddings=node_embeddings,
                action_embeddings=action_embeddings,
            )
        finally:
            if stream_e3_gamma:
                config.sspm_epochs = original_epochs
        train_stats["conditional_train_data_mode"] = "stream_event"
        train_stats["train_data_mode"] = "stream_event"
        train_stats["conditional_memmap_fingerprint"] = dict(stream_sentinel)
        train_stats["conditional_memmap_path"] = ""
    else:
        raise ValueError(
            "conditional_action_semantic full training requires "
            "SSPM_CONDITIONAL_TRAIN_DATA_MODE=memmap; stream_event must be explicit",
        )
    checkpoint_path = _phase3g_resolved_head_checkpoint_path(config)
    target_case_mode = (
        "action_embedding_only"
        if str(config.sspm_score_head) == CONDITIONAL_ACTION_EMBEDDING_SCORE_HEAD
        else "conditional_cold_action_else_event"
    )
    target_name = (
        "e_action"
        if str(config.sspm_score_head) == CONDITIONAL_ACTION_EMBEDDING_SCORE_HEAD
        else "conditional_target"
    )
    metadata = {
        "score_head": str(config.sspm_score_head),
        "conditional_head_arch": str(config.sspm_conditional_head_arch),
        "head_arch": str(config.sspm_conditional_head_arch),
        "node_repr_fusion": str(config.node_repr_fusion),
        "loss_type": str(config.conditional_semantic_loss),
        "target_case_mode": target_case_mode,
        "target_case_head_mode": str(head.config.target_case_head_mode),
        "target": target_name,
        "dataset": str(config.dataset),
        "target_mode": str(config.sspm_target_mode),
        "node_word2vec_source": str(config.node_word2vec_source),
        "node_embedding_fingerprint": _phase3g_file_fingerprint(paths["node_embeddings"]),
        "action_embedding_fingerprint": _phase3g_file_fingerprint(paths["action_embeddings"]),
        "event_index_fingerprint": dict(dict(event_meta.get("splits", {})).get("train", {})).get(
            "event_index_fingerprint",
        ),
        "state_model": str(config.sspm_state_model),
        "train_backend": str(config.sspm_train_backend),
        "infer_backend": str(config.sspm_infer_backend),
        "context_dim": int(input_dim),
        "target_logic": "state_exists_event_semantic_else_both_cold_action",
        "conditional_train_data_mode": str(
            train_stats.get("conditional_train_data_mode", config.sspm_conditional_train_data_mode),
        ),
        "conditional_memmap_fingerprint": (
            dict(conditional_memmap_meta.get("fingerprint", {}))
            if conditional_memmap_meta is not None
            else dict(
                train_stats.get(
                    "conditional_memmap_fingerprint",
                    _phase3g_conditional_memmap_sentinel(config),
                ),
            )
        ),
        "train_stats": train_stats,
    }
    target_case_path_for_fp = str(train_stats.get("conditional_memmap_target_case_path", ""))
    metadata["target_case_fingerprint"] = (
        _phase3g_file_fingerprint(target_case_path_for_fp)
        if target_case_path_for_fp
        else {}
    )
    head_fingerprint = head.fingerprint()
    metadata["checkpoint_schema"] = str(head_fingerprint.get("schema", ""))
    saved_path = head.save(checkpoint_path, metadata=metadata)
    _, saved_metadata = ConditionalSemanticHead.load(
        saved_path,
        expected_head_arch=str(config.sspm_conditional_head_arch),
    )
    checkpoint_fingerprint = dict(saved_metadata.get("fingerprint", {}))
    checkpoint_schema = str(
        saved_metadata.get("checkpoint_schema", checkpoint_fingerprint.get("schema", "")),
    )
    _stage_log(
        config,
        "phase3g_conditional_train_end",
        score_head=str(config.sspm_score_head),
        checkpoint=str(saved_path),
    )
    eval_payload = {
        "dataset": str(config.dataset),
        "out_tag": str(config.out_tag),
        "phase3g": {
            "score_head": str(config.sspm_score_head),
            "conditional_head_arch": str(config.sspm_conditional_head_arch),
            "head_arch": str(config.sspm_conditional_head_arch),
            "checkpoint_schema": checkpoint_schema,
            "node_repr_fusion": str(config.node_repr_fusion),
            "conditional_semantic_loss": str(config.conditional_semantic_loss),
            "target": target_name,
            "target_case_mode": target_case_mode,
            "target_case_head_mode": str(head.config.target_case_head_mode),
            "action_head_checkpoint_path": str(saved_path),
            "checkpoint_path": str(saved_path),
            "input_dim": int(input_dim),
            "context_dim": int(input_dim),
            "rank": int(config.rank),
            "output_dim": int(config.sspm_target_dim),
            "fingerprint": checkpoint_fingerprint,
            "train_stats": train_stats,
        },
        "train_events_actual": int(train_count),
        "validation_events_actual": int(validation_count),
        "test_events_actual": 0,
        "runtime": {"elapsed_seconds": float(time.perf_counter() - started)},
        "leakage_contract": {
            "contains_test_scores": False,
            "contains_test_labels": False,
            "ground_truth_used": False,
        },
    }
    eval_path = output_dir / "eval_causal_semantics_slim.json"
    eval_path.write_text(
        json.dumps(eval_payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    dummy_model = SSPMLowRankModel(_make_sspm_config(config, _load_process_config(config)))
    write_effective_config(
        output_dir,
        config,
        dummy_model,
        True,
        train_count=int(train_count),
        validation_count=int(validation_count),
        test_count=0,
    )
    write_metrics_json(
        output_dir,
        config,
        dummy_model,
        eval_payload,
        train_count=int(train_count),
        validation_count=int(validation_count),
        test_count=0,
        embedder_loaded=True,
    )
    return eval_path


# Explicit cross-module imports; replaces the migration namespace bridge.
from scripts.pipeline.checks.preflight import _load_process_config
from scripts.pipeline.features.conditional_context import (
    _phase3g_action_context_from_model,
    _phase3g_action_input_dim,
    _phase3g_file_fingerprint,
    _phase3g_resolved_head_checkpoint_path,
    _phase3g_type_eye,
)
from scripts.pipeline.io.conditional_cache import (
    _phase3g_conditional_memmap_fingerprint,
    _phase3g_conditional_memmap_paths,
    _phase3g_conditional_memmap_sentinel,
    _phase3g_load_conditional_memmap_meta,
    _phase3g_validate_conditional_memmap,
)
from scripts.pipeline.io.event_artifacts import (
    _phase3e_event_index_array,
    _phase3e_open_split_event_index,
    _phase3e_train_lr,
    _phase3g_effective_artifacts,
    _phase3g_open_node_embeddings,
    write_effective_config,
    write_metrics_json,
)
from scripts.pipeline.outputs.alert_output import _current_rss_mb
from scripts.pipeline.outputs.metrics_summary import _validate_active_event_score_mode
from scripts.pipeline.state.online_state_runtime import (
    _make_sspm_config,
    _phase3e_fields_from_index_row,
    _phase3e_target_from_index_row,
    _phase3f_e2_none_update_from_numeric_row,
    _phase3g_update_gate_enabled,
    _stage_log,
    load_sspm_checkpoint_state_for_conditional,
)


def _phase3g_conditional_memmap_only_payload(
    *,
    config: SlimConfig,
    output_dir: Path,
    memmap_meta: Mapping[str, Any],
    train_count: int,
    validation_count: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Build the metadata payload for a train-only conditional memmap run."""
    return {
        "dataset": str(config.dataset),
        "out_tag": str(config.out_tag),
        "phase3g": {
            "score_head": str(config.sspm_score_head),
            "conditional_head_arch": str(config.sspm_conditional_head_arch),
            "head_arch": str(config.sspm_conditional_head_arch),
            "train_data_mode": "conditional_memmap",
            "conditional_train_data_mode": "memmap",
            "conditional_memmap": dict(memmap_meta),
            "checkpoint_path": "",
            "action_head_checkpoint_path": "",
        },
        "outputs": {
            "result_dir": str(output_dir),
            "x_path": str(memmap_meta.get("x_path", "")),
            "y_path": str(memmap_meta.get("y_path", "")),
            "target_case_path": str(memmap_meta.get("target_case_path", "")),
        },
        "train_events_actual": int(train_count),
        "validation_events_actual": int(validation_count),
        "test_events_actual": 0,
        "runtime": {"elapsed_seconds": float(elapsed_seconds)},
        "leakage_contract": {
            "contains_test_scores": False,
            "contains_test_labels": False,
            "ground_truth_used": False,
            "purpose": "train-only conditional memmap cache build",
        },
    }


def _write_phase3g_conditional_memmap_only_outputs(
    *,
    config: SlimConfig,
    output_dir: Path,
    eval_payload: Mapping[str, Any],
    train_count: int,
    validation_count: int,
) -> Path:
    """Write sidecars for a conditional-memmap-only run without training a head."""
    eval_path = output_dir / "eval_causal_semantics_slim.json"
    eval_path.write_text(
        json.dumps(dict(eval_payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    dummy_model = SSPMLowRankModel(_make_sspm_config(config, _load_process_config(config)))
    write_effective_config(
        output_dir,
        config,
        dummy_model,
        True,
        train_count=int(train_count),
        validation_count=int(validation_count),
        test_count=0,
    )
    write_metrics_json(
        output_dir,
        config,
        dummy_model,
        dict(eval_payload),
        train_count=int(train_count),
        validation_count=int(validation_count),
        test_count=0,
        embedder_loaded=True,
    )
    return eval_path
