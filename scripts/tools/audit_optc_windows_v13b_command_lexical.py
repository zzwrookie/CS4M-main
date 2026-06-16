"""Read-only OpTC Windows safe command lexical audit."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs4m.semantics.optc_windows import (
    bucket_command_line_v1,
    bucket_file_basename_v1,
    bucket_file_ext_class_v1,
    bucket_file_family_v1,
    bucket_process_role,
    bucket_safe_command_lex_v1_3b,
    bucket_safe_command_lex_v1_3c,
    process_image_name,
)


def _write_counter(path: Path, fieldnames: list[str], counter: Counter, limit: int = 0) -> None:
    rows = counter.most_common(limit or None)
    total = sum(counter.values())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([*fieldnames, "count", "ratio"])
        for key, count in rows:
            values = key if isinstance(key, tuple) else (key,)
            writer.writerow([*values, int(count), float(count / total) if total else 0.0])


def _load_used_node_ids(path: Path) -> set[int]:
    with path.open("rb") as handle:
        return {int(node_id) for node_id in pickle.load(handle)}


def _lexers_for_mode(mode: str) -> dict[str, object]:
    """Return requested safe lexical bucket functions keyed by mode name."""
    if mode == "v1_3b":
        return {"v1_3b": bucket_safe_command_lex_v1_3b}
    if mode == "v1_3c":
        return {"v1_3c": bucket_safe_command_lex_v1_3c}
    return {
        "v1_3b": bucket_safe_command_lex_v1_3b,
        "v1_3c": bucket_safe_command_lex_v1_3c,
    }


def _write_compatibility_outputs(
    out_dir: Path,
    selected_mode: str,
    safe_lex_by_mode: dict[str, Counter],
    other_absorption_by_mode: dict[str, Counter],
) -> None:
    """Write legacy filenames for consumers expecting the original v1.3b audit output."""
    compat_mode = "v1_3b" if selected_mode == "both" else selected_mode
    _write_counter(
        out_dir / "process_safe_cmd_lex_distribution.csv",
        ["safe_cmd_lex"],
        safe_lex_by_mode[compat_mode],
    )
    _write_counter(
        out_dir / "process_other_cmd_absorption.csv",
        ["safe_cmd_lex"],
        other_absorption_by_mode[compat_mode],
    )


def _write_absorption_comparison(
    path: Path,
    other_absorption_by_mode: dict[str, Counter],
) -> None:
    """Write side-by-side v1.3b/v1.3c other_cmd absorption counts."""
    keys = set()
    for counter in other_absorption_by_mode.values():
        keys.update(counter)
    rows = []
    for key in sorted(keys):
        v13b_count = int(other_absorption_by_mode.get("v1_3b", Counter()).get(key, 0))
        v13c_count = int(other_absorption_by_mode.get("v1_3c", Counter()).get(key, 0))
        rows.append((key, v13b_count, v13c_count, v13c_count - v13b_count))
    rows.sort(key=lambda row: (abs(row[3]), row[1], row[2]), reverse=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["safe_cmd_lex", "v1_3b_count", "v1_3c_count", "delta_v1_3c_minus_v1_3b"])
        writer.writerows(rows)


def audit_command_lexical(args: argparse.Namespace) -> dict[str, object]:
    """Run read-only command lexical audit and write CSV/JSON outputs."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    used_node_ids = _load_used_node_ids(Path(args.node_id_to_idx))
    lexers = _lexers_for_mode(str(args.lexical_mode))

    cmd_bucket = Counter()
    safe_lex_by_mode: dict[str, Counter] = {mode: Counter() for mode in lexers}
    other_absorption_by_mode: dict[str, Counter] = {mode: Counter() for mode in lexers}
    role_counts = Counter()
    unknown_rows: list[dict[str, object]] = []
    file_triplets = Counter()
    process_total = 0
    file_total = 0

    conn = psycopg2.connect(
        host=args.db_host,
        port=int(args.db_port),
        user=args.db_user,
        password=args.db_password or os.environ.get("CLAD_DB_PASSWORD") or os.environ.get("PGPASSWORD"),
        dbname=args.db_name,
    )
    try:
        with conn.cursor(name="optc_v13b_process_audit") as cur:
            cur.itersize = int(args.fetch_size)
            cur.execute("select index_id, path, cmd from subject_node_table")
            for index_id, path, cmd in cur:
                node_id = int(index_id)
                if node_id not in used_node_ids:
                    continue
                image = process_image_name(path or cmd)
                role = bucket_process_role(image)
                bucket = bucket_command_line_v1(cmd)
                lex_by_mode = {mode: str(fn(cmd)) for mode, fn in lexers.items()}
                process_total += 1
                role_counts[role] += 1
                cmd_bucket[bucket] += 1
                for mode, lex in lex_by_mode.items():
                    safe_lex_by_mode[mode][lex] += 1
                    if bucket == "other_cmd":
                        other_absorption_by_mode[mode][lex] += 1
                if bucket == "other_cmd":
                    pass
                if role == "unknown_process":
                    primary_mode = "v1_3c" if "v1_3c" in lex_by_mode else next(iter(lex_by_mode))
                    unknown_rows.append(
                        {
                            "node_id": node_id,
                            "image_name": image,
                            "cmd_bucket": bucket,
                            "safe_cmd_lex": lex_by_mode[primary_mode],
                            "path": str(path or ""),
                            "cmd": str(cmd or ""),
                        },
                    )
        with conn.cursor(name="optc_v13b_file_audit") as cur:
            cur.itersize = int(args.fetch_size)
            cur.execute("select index_id, path from file_node_table")
            for index_id, path in cur:
                node_id = int(index_id)
                if node_id not in used_node_ids:
                    continue
                file_total += 1
                file_triplets[
                    (
                        bucket_file_family_v1(path),
                        bucket_file_ext_class_v1(path),
                        bucket_file_basename_v1(path),
                    )
                ] += 1
    finally:
        conn.close()

    _write_counter(out_dir / "process_role_distribution.csv", ["role"], role_counts)
    _write_counter(out_dir / "process_cmd_bucket_distribution.csv", ["cmd_bucket"], cmd_bucket)
    for mode in sorted(lexers):
        _write_counter(
            out_dir / f"process_safe_cmd_lex_distribution_{mode}.csv",
            ["safe_cmd_lex"],
            safe_lex_by_mode[mode],
        )
        _write_counter(
            out_dir / f"process_other_cmd_absorption_{mode}.csv",
            ["safe_cmd_lex"],
            other_absorption_by_mode[mode],
        )
    _write_compatibility_outputs(
        out_dir,
        str(args.lexical_mode),
        safe_lex_by_mode,
        other_absorption_by_mode,
    )
    if {"v1_3b", "v1_3c"}.issubset(set(lexers)):
        _write_absorption_comparison(
            out_dir / "process_other_cmd_absorption_comparison.csv",
            other_absorption_by_mode,
        )
    _write_counter(
        out_dir / "file_family_ext_basename_distribution.csv",
        ["family", "ext_class", "basename_bucket"],
        file_triplets,
        limit=1000,
    )
    with (out_dir / "unknown_process_nodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["node_id", "image_name", "cmd_bucket", "safe_cmd_lex", "path", "cmd"],
        )
        writer.writeheader()
        writer.writerows(unknown_rows)

    non_empty_cmd = process_total - cmd_bucket.get("empty_cmd", 0)
    primary_mode = "v1_3c" if "v1_3c" in lexers else next(iter(lexers))
    primary_other_absorption = other_absorption_by_mode[primary_mode]
    primary_safe_lex = safe_lex_by_mode[primary_mode]
    mode_summary = {}
    for mode in sorted(lexers):
        other_safe_count = int(other_absorption_by_mode[mode].get("cmdarg|other_safe", 0))
        mode_summary[mode] = {
            "other_cmd_other_safe_count": other_safe_count,
            "top_other_cmd_absorption": other_absorption_by_mode[mode].most_common(50),
            "top_safe_cmd_lex": safe_lex_by_mode[mode].most_common(50),
        }
    comparison_summary = {}
    if {"v1_3b", "v1_3c"}.issubset(set(lexers)):
        v13b_other_safe = int(other_absorption_by_mode["v1_3b"].get("cmdarg|other_safe", 0))
        v13c_other_safe = int(other_absorption_by_mode["v1_3c"].get("cmdarg|other_safe", 0))
        comparison_summary = {
            "v1_3b_other_cmd_other_safe_count": v13b_other_safe,
            "v1_3c_other_cmd_other_safe_count": v13c_other_safe,
            "other_safe_delta_v1_3c_minus_v1_3b": v13c_other_safe - v13b_other_safe,
            "other_safe_reduction_ratio": (
                float((v13b_other_safe - v13c_other_safe) / v13b_other_safe)
                if v13b_other_safe
                else 0.0
            ),
        }
    summary = {
        "db_name": str(args.db_name),
        "lexical_mode": str(args.lexical_mode),
        "node_id_to_idx": str(args.node_id_to_idx),
        "process_total_used_nodes": int(process_total),
        "process_non_empty_cmd_nodes": int(non_empty_cmd),
        "other_cmd_count": int(cmd_bucket.get("other_cmd", 0)),
        "other_cmd_ratio_all_process": (
            float(cmd_bucket.get("other_cmd", 0) / process_total) if process_total else 0.0
        ),
        "other_cmd_ratio_non_empty": (
            float(cmd_bucket.get("other_cmd", 0) / non_empty_cmd) if non_empty_cmd else 0.0
        ),
        "unknown_process_count": int(role_counts.get("unknown_process", 0)),
        "unknown_process_ratio": (
            float(role_counts.get("unknown_process", 0) / process_total) if process_total else 0.0
        ),
        "file_total_used_nodes": int(file_total),
        "file_other_other_ext_other_file_count": int(
            file_triplets.get(("file_other", "other_ext", "other_file"), 0),
        ),
        "file_other_other_ext_other_file_ratio": (
            float(file_triplets.get(("file_other", "other_ext", "other_file"), 0) / file_total)
            if file_total
            else 0.0
        ),
        "mode_summary": mode_summary,
        "comparison_summary": comparison_summary,
        "top_other_cmd_absorption": primary_other_absorption.most_common(50),
        "top_safe_cmd_lex": primary_safe_lex.most_common(50),
        "top_file_triplets": file_triplets.most_common(50),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = [
        "# OPTC Windows Command Lexical Audit",
        "",
        f"- lexical mode: {args.lexical_mode}",
        f"- process used nodes: {process_total}",
        f"- non-empty cmd nodes: {non_empty_cmd}",
        f"- other_cmd: {summary['other_cmd_count']} ({summary['other_cmd_ratio_non_empty']:.4%} of non-empty)",
        f"- unknown_process: {summary['unknown_process_count']} ({summary['unknown_process_ratio']:.4%})",
        f"- file_other|other_ext|other_file: {summary['file_other_other_ext_other_file_count']} ({summary['file_other_other_ext_other_file_ratio']:.4%})",
        "",
        "## Other Cmd Absorption",
    ]
    for mode in sorted(lexers):
        report.append(f"### {mode}")
        for key, count in other_absorption_by_mode[mode].most_common(30):
            report.append(f"- {key}: {count}")
    report.append("")
    report.append("## Safe Command Lexical Distribution")
    for mode in sorted(lexers):
        report.append(f"### {mode}")
        for key, count in safe_lex_by_mode[mode].most_common(30):
            report.append(f"- {key}: {count}")
    report.append("")
    report.append("## File Audit")
    for key, count in file_triplets.most_common(30):
        report.append(f"- {key}: {count}")
    (out_dir / "OPTC_Windows_v13b_Command_Lexical_Audit_Report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    (out_dir / "OPTC_Windows_v13c_Command_Lexical_Audit_Report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-name", default="optc_051")
    parser.add_argument("--db-host", default=os.environ.get("CLAD_DB_HOST", "localhost"))
    parser.add_argument("--db-port", default=os.environ.get("CLAD_DB_PORT", "5433"))
    parser.add_argument("--db-user", default=os.environ.get("CLAD_DB_USER", "postgres"))
    parser.add_argument("--db-password", default=os.environ.get("CLAD_DB_PASSWORD", ""))
    parser.add_argument(
        "--node-id-to-idx",
        default=(
            "outputs/cache/phase3e_optc_windows_v1_3/"
            "OPTC_051_OPTC_WINDOWS_V1_3_DETAIL_full/node/node_id_to_idx.pkl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/diagnostics/optc_windows_v13b_command_lexical_audit",
    )
    parser.add_argument(
        "--lexical-mode",
        choices=("v1_3b", "v1_3c", "both"),
        default="both",
        help="Safe command lexical bucket version to audit.",
    )
    parser.add_argument("--fetch-size", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    """Run CLI entrypoint."""
    summary = audit_command_lexical(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
