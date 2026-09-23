from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


ARTIFACT_ROOT = Path("artifacts/phase6_act_dataset")
TARGET_VOCAB = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ("SPACE", "BACKSPACE")


def _latest_run(target: str) -> Path:
    root = ARTIFACT_ROOT / f"{target.lower()}_natural_v3" / "runs"
    runs = sorted(p for p in root.glob("run_*") if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No Phase 6 v3 runs found under {root}")
    return runs[-1]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _image_shape_ok(value) -> bool:
    shape = tuple(int(x) for x in _as_numpy(value).shape)
    return shape in {(480, 640, 3), (3, 480, 640)}


def _action_vector_from_named(payload: dict, names: list[str]) -> np.ndarray:
    return np.asarray([float(payload[name]) for name in names], dtype=np.float64)


def _finite_action_dict(payload: dict, names: list[str]) -> bool:
    try:
        return all(name in payload and math.isfinite(float(payload[name])) for name in names)
    except (TypeError, ValueError):
        return False


def _check_trace(summary: dict, replay_path: Path) -> tuple[list[str], list[str], dict]:
    fail: list[str] = []
    review: list[str] = []
    trace = _read_jsonl(replay_path)
    names = list(summary.get("motor_names") or [])

    if len(names) != 6:
        fail.append(f"motor_names_count={len(names)} expected=6")
    if len(trace) < 10:
        fail.append(f"replay_trace_too_short={len(trace)}")
        return fail, review, {"trace_samples": len(trace)}

    timestamps = [float(item.get("timestamp", float("nan"))) for item in trace]
    if not all(math.isfinite(v) for v in timestamps):
        fail.append("replay_trace_nonfinite_timestamp")
    elif any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        fail.append("replay_trace_timestamp_not_strictly_increasing")

    if names:
        bad = 0
        for item in trace:
            sent = item.get("sent_action") or {}
            if not _finite_action_dict(sent, names):
                bad += 1
        if bad:
            fail.append(f"replay_trace_bad_sent_actions={bad}")

    intervals_ms = [
        (timestamps[i] - timestamps[i - 1]) * 1000.0
        for i in range(1, len(timestamps))
        if math.isfinite(timestamps[i]) and math.isfinite(timestamps[i - 1])
    ]
    trace_stats = {
        "trace_samples": len(trace),
        "duration_s": (timestamps[-1] - timestamps[0]) if len(timestamps) >= 2 else 0.0,
        "interval_ms_p50": float(np.percentile(intervals_ms, 50)) if intervals_ms else None,
        "interval_ms_p95": float(np.percentile(intervals_ms, 95)) if intervals_ms else None,
        "interval_ms_p99": float(np.percentile(intervals_ms, 99)) if intervals_ms else None,
        "interval_ms_max": max(intervals_ms) if intervals_ms else None,
    }

    perf = summary.get("performance") or {}
    achieved = float(perf.get("achieved_send_hz") or 0.0)
    p99 = (perf.get("inter_send_ms") or {}).get("p99")
    max_ms = (perf.get("inter_send_ms") or {}).get("max")
    sample_misses = int(perf.get("sample_missed_count") or 0)
    top_age = (perf.get("sample_frame_age_ms") or {}).get("top_p95")
    wrist_age = (perf.get("sample_frame_age_ms") or {}).get("wrist_p95")

    if achieved and achieved < 45.0:
        fail.append(f"control_rate_too_low={achieved:.1f}Hz")
    elif achieved and achieved < 52.0:
        review.append(f"control_rate_low={achieved:.1f}Hz")

    if p99 is not None and float(p99) > 40.0:
        review.append(f"inter_send_p99_high={float(p99):.1f}ms")
    if max_ms is not None and float(max_ms) > 100.0:
        review.append(f"inter_send_max_high={float(max_ms):.1f}ms")
    if sample_misses > 0:
        review.append(f"dataset_sample_misses={sample_misses}")
    if top_age is not None and float(top_age) > 100.0:
        review.append(f"top_frame_age_p95_high={float(top_age):.1f}ms")
    if wrist_age is not None and float(wrist_age) > 100.0:
        review.append(f"wrist_frame_age_p95_high={float(wrist_age):.1f}ms")

    clamp = float(summary.get("max_requested_vs_sent_action_delta_deg") or 0.0)
    if clamp > 1.0:
        review.append(f"requested_vs_sent_clamp={clamp:.3f}deg")

    quality = summary.get("quality") or {}
    for warning in quality.get("warnings") or []:
        review.append(f"trajectory:{warning}")

    return fail, review, trace_stats


def _check_audit(summary: dict, audit_path: Path) -> tuple[list[str], list[str], dict]:
    fail: list[str] = []
    review: list[str] = []
    rows = _read_jsonl(audit_path)
    if not rows:
        fail.append("sample_audit_missing_or_empty")
        return fail, review, {"audit_samples": 0}

    top_ids = [int(r["top_frame_id"]) for r in rows]
    wrist_ids = [int(r["wrist_frame_id"]) for r in rows]
    if any(b <= a for a, b in zip(top_ids, top_ids[1:])):
        review.append("top_frame_ids_not_strictly_increasing")
    if any(b <= a for a, b in zip(wrist_ids, wrist_ids[1:])):
        review.append("wrist_frame_ids_not_strictly_increasing")

    top_ages = [float(r.get("top_frame_age_ms", 0.0)) for r in rows]
    wrist_ages = [float(r.get("wrist_frame_age_ms", 0.0)) for r in rows]
    return fail, review, {
        "audit_samples": len(rows),
        "top_frame_age_ms_p95": float(np.percentile(top_ages, 95)),
        "wrist_frame_age_ms_p95": float(np.percentile(wrist_ages, 95)),
    }


def _check_dataset_episode(
    *,
    repo_id: str,
    dataset_root: Path,
    episode_index: int,
    summary: dict,
    target: str,
) -> tuple[list[str], list[str], dict]:
    fail: list[str] = []
    review: list[str] = []
    ds = LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=[episode_index],
        return_uint8=True,
    )
    expected_count = int(summary.get("saved_samples") or 0)
    if len(ds) != expected_count:
        fail.append(f"dataset_frame_count={len(ds)} expected={expected_count}")
    if len(ds) == 0:
        return fail + ["dataset_episode_empty"], review, {"dataset_frames": 0}

    names = list(summary.get("motor_names") or [])
    expected_target_index = TARGET_VOCAB.index(target)
    sample_indices = sorted(set([0, len(ds) // 2, len(ds) - 1]))
    first_action = None
    last_action = None
    for i in sample_indices:
        frame = ds[i]
        for key in ("observation.images.top", "observation.images.wrist", "observation.state", "action"):
            if key not in frame:
                fail.append(f"dataset_missing_feature:{key}")
                continue
        if "observation.images.top" in frame and not _image_shape_ok(frame["observation.images.top"]):
            fail.append(f"top_image_bad_shape={tuple(_as_numpy(frame['observation.images.top']).shape)}")
        if "observation.images.wrist" in frame and not _image_shape_ok(frame["observation.images.wrist"]):
            fail.append(f"wrist_image_bad_shape={tuple(_as_numpy(frame['observation.images.wrist']).shape)}")

        if "observation.state" in frame:
            state = _as_numpy(frame["observation.state"]).reshape(-1).astype(np.float64)
            if state.shape != (34,):
                fail.append(f"state_bad_shape={state.shape}")
            elif not np.isfinite(state).all():
                fail.append("state_nonfinite")
            else:
                one_hot = state[6:]
                if not np.isclose(float(one_hot.sum()), 1.0, atol=1e-5):
                    fail.append("target_one_hot_sum_not_1")
                elif int(np.argmax(one_hot)) != expected_target_index:
                    fail.append(
                        f"target_one_hot_wrong_index={int(np.argmax(one_hot))} expected={expected_target_index}"
                    )

        if "action" in frame:
            action = _as_numpy(frame["action"]).reshape(-1).astype(np.float64)
            if action.shape != (6,):
                fail.append(f"action_bad_shape={action.shape}")
            elif not np.isfinite(action).all():
                fail.append("action_nonfinite")

    first = ds[0]
    last = ds[len(ds) - 1]
    if "action" in first:
        first_action = _as_numpy(first["action"]).reshape(-1).astype(np.float64)
    if "action" in last:
        last_action = _as_numpy(last["action"]).reshape(-1).astype(np.float64)

    if names and first_action is not None and summary.get("first_saved_action"):
        expected = _action_vector_from_named(summary["first_saved_action"], names)
        if not np.allclose(first_action, expected, atol=1e-4):
            fail.append("dataset_first_action_mismatch_audit")
    if names and last_action is not None and summary.get("final_saved_action"):
        expected = _action_vector_from_named(summary["final_saved_action"], names)
        if not np.allclose(last_action, expected, atol=1e-4):
            fail.append("dataset_final_action_mismatch_audit")

    # A duplicate camera stream is almost certainly a wiring/configuration error.
    try:
        top = _as_numpy(first["observation.images.top"])
        wrist = _as_numpy(first["observation.images.wrist"])
        if top.shape == wrist.shape and np.array_equal(top, wrist):
            fail.append("top_and_wrist_images_identical")
    except Exception:
        review.append("camera_duplicate_check_unavailable")

    return fail, review, {"dataset_frames": len(ds), "sampled_indices": sample_indices}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6D generic episode-level QC. Start-state strategy is outside this tool; no Rerun/manual viewing is required for PASS episodes."
    )
    parser.add_argument("--target", default="G")
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    target = str(args.target).strip().upper()
    if target not in TARGET_VOCAB[:26]:
        raise ValueError("Current Phase 6 QC expects a single A-Z target")

    run_dir = args.run_dir or _latest_run(target)
    run_summary_path = run_dir / "run_summary.json"
    if not run_summary_path.exists():
        raise FileNotFoundError(run_summary_path)
    run_summary = _read_json(run_summary_path)
    repo_id = str(run_summary["repo_id"])
    dataset_root = Path(run_summary["dataset_root"])

    summaries = sorted(run_dir.glob("attempt_*/summary.json"))
    results: list[dict] = []
    candidates: list[int] = []

    print("=" * 72)
    print("PHASE 6D — AUTOMATIC EPISODE QC")
    print("=" * 72)
    print("run dir     :", run_dir)
    print("dataset root:", dataset_root)
    print("policy      : PASS -> replay candidate; REVIEW/FAIL -> excluded until recollected or inspected")
    print("Rerun       : NOT REQUIRED; reserve it only for optional debugging")
    print()

    for summary_path in summaries:
        summary = _read_json(summary_path)
        if not summary.get("accepted") or "dataset_episode_index" not in summary:
            continue

        episode_index = int(summary["dataset_episode_index"])
        fail: list[str] = []
        review: list[str] = []

        replay_path = Path(summary.get("replay_trace_jsonl") or summary_path.parent / "replay_trace.jsonl")
        audit_path = Path(summary.get("audit_jsonl") or summary_path.parent / "sample_audit.jsonl")
        f, r, trace_stats = _check_trace(summary, replay_path)
        fail.extend(f); review.extend(r)
        f, r, audit_stats = _check_audit(summary, audit_path)
        fail.extend(f); review.extend(r)
        try:
            f, r, dataset_stats = _check_dataset_episode(
                repo_id=repo_id,
                dataset_root=dataset_root,
                episode_index=episode_index,
                summary=summary,
                target=target,
            )
            fail.extend(f); review.extend(r)
        except Exception as exc:
            fail.append(f"dataset_read_error:{type(exc).__name__}:{exc}")
            dataset_stats = {}

        # Deduplicate while preserving readable order.
        fail = list(dict.fromkeys(fail))
        review = list(dict.fromkeys(review))
        status = "FAIL" if fail else ("REVIEW" if review else "PASS")
        if status == "PASS":
            candidates.append(episode_index)

        item = {
            "episode_number": int(summary["episode_number"]),
            "dataset_episode_index": episode_index,
            "status": status,
            "fail_reasons": fail,
            "review_reasons": review,
            "trace": trace_stats,
            "audit": audit_stats,
            "dataset": dataset_stats,
            "summary_json": str(summary_path),
            "replay_trace_jsonl": str(replay_path),
        }
        results.append(item)
        reasons = fail or review
        suffix = "" if not reasons else " | " + "; ".join(reasons)
        print(
            f"[{status:6}] dataset_ep={episode_index:03d} "
            f"episode={item['episode_number']:02d}{suffix}"
        )

    report = {
        "schema": "phase6.episode_qc.v2",
        "target": target,
        "run_dir": str(run_dir),
        "dataset_root": str(dataset_root),
        "repo_id": repo_id,
        "policy": "Only PASS episodes automatically enter physical replay validation; Rerun is optional debug only.",
        "episodes": results,
        "pass_dataset_episode_indices": candidates,
        "counts": {
            "pass": sum(x["status"] == "PASS" for x in results),
            "review": sum(x["status"] == "REVIEW" for x in results),
            "fail": sum(x["status"] == "FAIL" for x in results),
        },
    }
    report_path = run_dir / "qc_report.json"
    candidates_path = run_dir / "qc_candidates.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    candidates_path.write_text(
        json.dumps(
            {
                "schema": "phase6.replay_candidates.v2",
                "target": target,
                "run_dir": str(run_dir),
                "dataset_episode_indices": candidates,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print("QC counts     :", report["counts"])
    print("replay candidates:", candidates)
    print("report        :", report_path)
    print("candidate list:", candidates_path)


if __name__ == "__main__":
    main()
