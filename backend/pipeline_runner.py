from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple


def _log(log: Callable[[str], None], msg: str) -> None:
    log(msg)


# 与 `main.py` 脚本入口、README「权重路径」约定一致；可用环境变量覆盖（见 README）。
_DEFAULT_TRACKNET_REL = "data/weights/ckpts/TrackNet_best.pt"
_DEFAULT_HITNET_REL = "data/weights/hitnet_output/hitnet_overfit_best.pth"
_DEFAULT_POSE_REL = "yolov8x-pose.pt"


def _resolve_model_weight(project_root: Path, env_var: str, default_relative: str) -> Path:
    """优先读 `SHUTTLEVISION_*`；相对路径相对于项目根目录。"""
    raw = (os.environ.get(env_var) or "").strip()
    if raw:
        p = Path(raw)
        return p.resolve() if p.is_absolute() else (project_root / p).resolve()
    return (project_root / Path(default_relative)).resolve()


def _resolve_tracknet_weights(project_root: Path, weights_tracknet: Path | None, log: Callable[[str], None]) -> Path:
    """
    TrackNet 检查点：显式参数 > 环境变量 > 默认文件名 > 常见备用路径 >
    `data/weights/ckpts` 下唯一或与 tracknet 同名的 .pt。
    """
    if weights_tracknet is not None:
        return Path(weights_tracknet).resolve()

    raw = (os.environ.get("SHUTTLEVISION_TRACKNET") or "").strip()
    if raw:
        p = Path(raw)
        return p.resolve() if p.is_absolute() else (project_root / p).resolve()

    default = (project_root / Path(_DEFAULT_TRACKNET_REL)).resolve()
    if default.is_file():
        return default

    for rel in (
        "data/weights/TrackNet_best.pt",
        "data/weights/tracknet_best.pt",
        "data/weights/ckpts/tracknet_best.pt",
    ):
        cand = (project_root / rel).resolve()
        if cand.is_file():
            _log(log, f"[weights] tracknet: default missing, using {cand}")
            return cand

    ckpts_dir = project_root / "data" / "weights" / "ckpts"
    if ckpts_dir.is_dir():
        pts = sorted(
            [p for p in ckpts_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pt"],
            key=lambda x: x.name.lower(),
        )
        preferred = sorted(
            [p for p in pts if "tracknet" in p.name.lower()],
            key=lambda x: x.name.lower(),
        )
        pick: Path | None = None
        if len(preferred) >= 1:
            pick = preferred[0]
        elif len(pts) == 1:
            pick = pts[0]
        if pick is not None:
            _log(log, f"[weights] tracknet: default missing, using ckpts/{pick.name}")
            return pick.resolve()

    return default


def _resolve_hitnet_weights(project_root: Path, weights_hitnet: Path | None, log: Callable[[str], None]) -> Path:
    """
    HitNet 权重：显式参数 > 环境变量 > 默认文件名 > 常见备用路径 >
    `data/weights/hitnet_output` 或 `data/weights` 下唯一或与 hitnet 同名的 .pth。
    """
    if weights_hitnet is not None:
        return Path(weights_hitnet).resolve()

    raw = (os.environ.get("SHUTTLEVISION_HITNET") or "").strip()
    if raw:
        p = Path(raw)
        return p.resolve() if p.is_absolute() else (project_root / p).resolve()

    default = (project_root / Path(_DEFAULT_HITNET_REL)).resolve()
    if default.is_file():
        return default

    for rel in (
        "data/weights/hitnet_overfit_best.pth",
        "data/weights/hitnet_output/hitnet_overfit_best.pth",
        "data/weights/hitNet_output/hitnet_overfit_best.pth",
    ):
        cand = (project_root / rel).resolve()
        if cand.is_file():
            _log(log, f"[weights] hitnet: default missing, using {cand}")
            return cand

    for search_dir_rel in ("data/weights/hitnet_output", "data/weights"):
        search_dir = (project_root / search_dir_rel).resolve()
        if not search_dir.is_dir():
            continue
        pths = sorted(
            [p for p in search_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pth"],
            key=lambda x: x.name.lower(),
        )
        preferred = sorted(
            [p for p in pths if "hitnet" in p.name.lower()],
            key=lambda x: x.name.lower(),
        )
        pick: Path | None = None
        if len(preferred) >= 1:
            pick = preferred[0]
        elif len(pths) == 1:
            pick = pths[0]
        if pick is not None:
            _log(log, f"[weights] hitnet: default missing, using {search_dir_rel}/{pick.name}")
            return pick.resolve()

    return default


def _raise_if_weights_missing(
    *,
    log: Callable[[str], None],
    weights: list[tuple[str, Path, str]],
) -> None:
    """在跑 TrackNet 前检查文件存在，避免 torch.load 报晦涩的 Errno 2。"""
    missing: list[tuple[str, Path, str]] = []
    for label, path, env_var in weights:
        if not path.is_file():
            missing.append((label, path, env_var))
    if not missing:
        return
    lines = ["缺少模型权重文件（请放置权重或设置环境变量）："]
    for label, path, env_var in missing:
        lines.append(f"  [{label}] {path}")
        lines.append(f"           环境变量: {env_var}")
    msg = "\n".join(lines)
    _log(log, msg)
    raise FileNotFoundError(msg)


def _split_report_png_to_panels(report_png: Path, out_dir: Path, log: Callable[[str], None]) -> dict[str, Path]:
    """
    将 `visualize_results` 生成的 3x2 总图拆分为 6 张子图，便于 Web 下拉菜单逐项预览。
    面板顺序（从左到右、从上到下）：
      traj3d, height_time, side_zy,
      top_xy, side_zx, overlay_panel
    """
    if not report_png.is_file():
        return {}
    try:
        from PIL import Image  # type: ignore
    except Exception as e:
        _log(log, f"[save] split report skipped: Pillow unavailable ({e})")
        return {}

    try:
        with Image.open(report_png) as im:
            w, h = im.size
            if w < 60 or h < 40:
                _log(log, f"[save] split report skipped: invalid size {w}x{h}")
                return {}

            names = [
                "traj3d",
                "height_time",
                "side_zy",
                "top_xy",
                "side_zx",
                "overlay_panel",
            ]
            panel_paths: dict[str, Path] = {}

            for r in range(2):
                for c in range(3):
                    idx = r * 3 + c
                    name = names[idx]
                    left = int((c * w) / 3)
                    right = int(((c + 1) * w) / 3)
                    top = int((r * h) / 2)
                    bottom = int(((r + 1) * h) / 2)
                    crop = im.crop((left, top, right, bottom))
                    out = out_dir / f"output_3d_{name}.png"
                    crop.save(out)
                    panel_paths[name] = out
            _log(log, "[save] split report png into 6 panels")
            return panel_paths
    except Exception as e:
        _log(log, f"[save] split report failed: {e}")
        return {}


def _transcode_mp4_for_web(src_mp4: Path, dst_mp4: Path, log: Callable[[str], None]) -> Path | None:
    """
    将 OpenCV 产出的 MP4 转为浏览器更稳定可播的 H.264/yuv420p（含 faststart）。
    失败时不抛异常，仅返回 None，避免影响主流程。
    """
    if not src_mp4.is_file():
        return None
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        _log(log, "[save] ffmpeg not found, skip web transcode")
        return None

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src_mp4),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        str(dst_mp4),
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600,
            check=False,
        )
    except Exception as e:
        _log(log, f"[save] ffmpeg transcode exception: {e}")
        return None

    if proc.returncode == 0 and dst_mp4.is_file() and dst_mp4.stat().st_size > 0:
        _log(log, f"[save] web mp4 -> {dst_mp4}")
        return dst_mp4

    err_tail = (proc.stderr or "").strip().splitlines()[-1:] or ["unknown ffmpeg error"]
    _log(log, f"[save] ffmpeg transcode failed: {err_tail[0]}")
    return None


def run_pipeline(
    *,
    project_root: Path,
    video_path: Path,
    points_2d: List[Tuple[int, int]],
    out_dir: Path,
    perf_mode: str = "standard",
    weights_tracknet: Path | None = None,
    weights_hitnet: Path | None = None,
    weights_pose: Path | None = None,
    log: Callable[[str], None],
    set_progress: Callable[[float, str], None],
) -> dict:
    """
    Web-friendly wrapper around existing logic in `main.py`.

    - Uses provided 6 calibration points (no OpenCV GUI).
    - Writes artifacts into out_dir.
    - Returns artifact paths and basic metrics.
    """
    # macOS + worker thread: matplotlib GUI backend raises; ensure non-interactive before main pulls pyplot.
    os.environ.setdefault("MPLBACKEND", "Agg")

    # Ensure import from project root
    sys.path.insert(0, str(project_root))

    # Lazy import heavy modules only inside job thread
    from main import (  # type: ignore
        CameraCalibrator,
        DataLoader,
        HitInferenceRunner,
        HitNetConfig,
        PhysicsModel,
        TrajectoryReconstructor,
        VideoRenderer,
        build_players_export,
        predict_trajectory,
        save_player_movement_heatmap,
        visualize_results,
    )
    from render_player_pose_2d import extract_poses_by_frame_from_track  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = project_root / "data" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    weights_tracknet = _resolve_tracknet_weights(project_root, weights_tracknet, log)

    weights_hitnet = _resolve_hitnet_weights(project_root, weights_hitnet, log)

    pose_arg_explicit = weights_pose is not None
    pose_env_set = bool((os.environ.get("SHUTTLEVISION_POSE") or "").strip())
    if weights_pose is None:
        weights_pose = _resolve_model_weight(project_root, "SHUTTLEVISION_POSE", _DEFAULT_POSE_REL)
    else:
        weights_pose = Path(weights_pose)
    if (
        not weights_pose.is_file()
        and not pose_arg_explicit
        and not pose_env_set
    ):
        _log(
            log,
            "[weights] pose: default weight file absent, using yolov8x-pose.pt (Ultralytics hub / cache)",
        )
        weights_pose = Path(_DEFAULT_POSE_REL)

    _log(
        log,
        f"[weights] tracknet={weights_tracknet} hitnet={weights_hitnet} pose={weights_pose}",
    )
    _raise_if_weights_missing(
        log=log,
        weights=[
            ("TrackNet", weights_tracknet, "SHUTTLEVISION_TRACKNET"),
            ("HitNet", weights_hitnet, "SHUTTLEVISION_HITNET"),
        ],
    )

    traj_csv = tmp_dir / f"tmp_{out_dir.name}.csv"
    out_csv = out_dir / "output_reconstructed_3d.csv"
    out_video = out_dir / "output_3d.mp4"
    out_video_web = out_dir / "output_3d_web.mp4"
    out_png = out_dir / "output_3d_result.png"
    panel_pngs: dict[str, Path] = {}
    out_players_csv = out_dir / "output_players_3d.csv"
    out_players_stats = out_dir / "output_players_stats.csv"
    out_players_heatmap = out_dir / "output_players_heatmap.png"
    out_reproj_json = out_dir / "output_reproj_metrics.json"

    # perf tuning
    batch_size = 4 if perf_mode == "fast" else 8 if perf_mode == "standard" else 16

    try:
        set_progress(0.05, "calibration")
        _log(log, "[step] calibration: computing P from 6 points")
        calib = CameraCalibrator(str(video_path))
        calib.points_2d_original = [(int(x), int(y)) for x, y in points_2d]
        calib._apply_fixed_calibration()  # noqa: SLF001 (existing private method)
        court_corners = [[x[0], x[1]] for i, x in enumerate(calib.points_2d_original) if i < 4]
        _log(log, f"[calibration] reproj_error={calib.reproj_error:.2f}px camera=({calib.camera_pos[0]:.2f},{calib.camera_pos[1]:.2f},{calib.camera_pos[2]:.2f})")

        set_progress(0.18, "tracknet_2d")
        _log(log, f"[step] tracknet: generating 2D trajectory csv -> {traj_csv}")
        predict_trajectory(
            video_file=str(video_path),
            tracknet_file=str(weights_tracknet),
            inpaintnet_file=None,
            batch_size=batch_size,
            eval_mode="nonoverlap",
            large_video=True,
            output_video=False,
            save_dir=str(tmp_dir),
            out_csv_file=str(traj_csv),
            return_dict=False,
        )
        _log(log, "[tracknet] done")

        set_progress(0.28, "pose_track")
        _log(log, "[step] yolo pose + bytetrack (shared for hitnet + optional overlay)")
        poses_by_frame = {}
        track_draw_by_frame = {}
        try:
            poses_by_frame, track_draw_by_frame = extract_poses_by_frame_from_track(
                str(video_path),
                str(weights_pose),
                conf_thr=0.25,
                device=None,
                tracker="bytetrack.yaml",
            )
        except Exception as e:
            _log(log, f"[pose_track] failed, hitnet falls back to per-frame yolo: {e}")
        pose_cache = poses_by_frame if len(poses_by_frame) > 0 else None

        set_progress(0.32, "hitnet_init")
        _log(log, "[step] hitnet: init model")
        hit_config = HitNetConfig(
            court_corners=court_corners,
            video_path=str(video_path),
            weights_path=str(weights_hitnet),
            POSE_model=str(weights_pose),
            fps=30,
        )
        inference_runner = HitInferenceRunner(hit_config, poses_by_frame=pose_cache)
        _log(log, "[hitnet] model loaded")

        set_progress(0.42, "hitnet_infer")
        _log(log, "[step] hitnet: segment shots")
        loader = DataLoader(str(traj_csv), inference_runner=inference_runner)
        shots = loader.get_shots()
        _log(log, f"[hitnet] shots={len(shots)}")
        inference_runner.release()

        set_progress(0.55, "reconstruct_3d")
        _log(log, "[step] reconstruct: 3D trajectory optimization")
        physics = PhysicsModel()
        reconstructor = TrajectoryReconstructor(calib, physics, fps=calib.fps, scale_y=1.0)

        trajectories_3d = []
        success_count = 0
        for i, shot in enumerate(shots):
            traj = reconstructor.reconstruct(shot)
            trajectories_3d.append(traj)
            if traj is not None:
                success_count += 1
            # coarse progress update
            if len(shots) > 0:
                set_progress(0.55 + 0.25 * ((i + 1) / len(shots)), "reconstruct_3d")

        set_progress(0.82, "save_csv")
        _log(log, "[step] save: csv/mp4/png")

        # System reprojection quality: project the optimised 3D trajectory back
        # through the calibrated camera and compare against TrackNet 2D observations.
        try:
            from pipeline import _compute_per_shot_reproj_metrics  # type: ignore
            reproj_metrics = _compute_per_shot_reproj_metrics(shots, trajectories_3d, calib)
            with open(out_reproj_json, "w", encoding="utf-8") as f:
                json.dump(reproj_metrics, f, ensure_ascii=False, indent=2)
            _log(
                log,
                f"[reproj] calib={reproj_metrics['calib_reproj_error_px']:.2f}px"
                f" | shuttle mean={reproj_metrics['overall_mean_px']:.2f}px"
                f" median={reproj_metrics['overall_median_px']:.2f}px"
                f" rms={reproj_metrics['overall_rms_px']:.2f}px"
                f" n_obs={reproj_metrics['overall_n_obs']}"
                f" -> {out_reproj_json}",
            )
        except Exception as e:
            _log(log, f"[reproj] skipped: {e}")

        # Reuse main.py's saving logic by duplicating the essential lines here.
        # main.py already stored per-shot predictions inside `shot` during reconstruct().
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore

        output_data = []
        for shot, traj in zip(shots, trajectories_3d):
            if traj is None:
                continue
            for j in range(len(traj)):
                frame_num = shot.start_frame + j
                pred_x = pred_y = actual_x = actual_y = error_val = method_val = conf_val = None
                if hasattr(shot, "frame_predictions") and frame_num in shot.frame_predictions:
                    pred = shot.frame_predictions[frame_num]
                    pred_x, pred_y = float(pred["pos"][0]), float(pred["pos"][1])
                    method_val = pred.get("method")
                    conf_val = float(pred.get("confidence")) if pred.get("confidence") is not None else None
                if j == len(traj) - 1:
                    if getattr(shot, "actual_landing", None) is not None:
                        actual_x = float(shot.actual_landing[0])
                        actual_y = float(shot.actual_landing[1])
                    error_val = float(shot.prediction_error) if getattr(shot, "prediction_error", None) is not None else None

                pos = traj[j]
                output_data.append(
                    {
                        "rally_id": int(shot.rally_id),
                        "shot_number": int(shot.shot_number),
                        "frame": int(frame_num),
                        "x": float(pos[0]),
                        "y": float(pos[1]),
                        "z": float(pos[2]),
                        "hitter": str(shot.hitter),
                        "is_last_in_rally": 1 if bool(shot.is_last_in_rally) else 0,
                        "predicted_landing_x": pred_x,
                        "predicted_landing_y": pred_y,
                        "prediction_method": method_val,
                        "prediction_confidence": conf_val,
                        "actual_landing_x": actual_x,
                        "actual_landing_y": actual_y,
                        "prediction_error": error_val,
                    }
                )

        if output_data:
            pd.DataFrame(output_data).to_csv(out_csv, index=False)
            _log(log, f"[save] csv rows={len(output_data)} -> {out_csv}")

        if success_count > 0:
            renderer = VideoRenderer(str(video_path), str(out_video), calib)
            renderer.render(
                shots,
                trajectories_3d,
                track_draw_by_frame=track_draw_by_frame or None,
                pose_render_conf=0.25,
                pose_draw_skeleton=True,
            )
            visualize_results(str(video_path), shots, trajectories_3d, calib, str(out_png))
            panel_pngs = _split_report_png_to_panels(out_png, out_dir, log)
            _transcode_mp4_for_web(out_video, out_video_web, log)
            _log(log, f"[save] mp4 -> {out_video}")
            _log(log, f"[save] png -> {out_png}")
        else:
            _log(log, "[save] no successful trajectories, skip mp4/png")

        # Optional: player movement metrics + heatmap
        near_dist = far_dist = None
        near_avg = far_avg = None
        if poses_by_frame and getattr(calib, "P", None) is not None:
            try:
                (
                    players_df,
                    total_near_m,
                    total_far_m,
                    avg_near_seg,
                    avg_far_seg,
                    avg_near_span,
                    avg_far_span,
                    max_near_seg,
                    max_far_seg,
                ) = build_players_export(poses_by_frame, calib, fps=float(calib.fps))
                players_df.to_csv(out_players_csv, index=False)
                pd.DataFrame(
                    [
                        {
                            "total_path_near_m": total_near_m,
                            "total_path_far_m": total_far_m,
                            "avg_speed_segment_near_mps": avg_near_seg,
                            "avg_speed_segment_far_mps": avg_far_seg,
                            "max_speed_segment_near_mps": max_near_seg,
                            "max_speed_segment_far_mps": max_far_seg,
                            "avg_speed_span_near_mps": avg_near_span,
                            "avg_speed_span_far_mps": avg_far_span,
                            "fps": float(calib.fps),
                        }
                    ]
                ).to_csv(out_players_stats, index=False)
                save_player_movement_heatmap(
                    players_df,
                    str(out_players_heatmap),
                    avg_near_segment_mps=avg_near_seg,
                    avg_far_segment_mps=avg_far_seg,
                    avg_near_span_mps=avg_near_span,
                    avg_far_span_mps=avg_far_span,
                    max_near_segment_mps=max_near_seg,
                    max_far_segment_mps=max_far_seg,
                    total_near_m=total_near_m,
                    total_far_m=total_far_m,
                    rally_label="回合 1",
                )
                near_dist = float(total_near_m)
                far_dist = float(total_far_m)
                near_avg = float(avg_near_seg) if np.isfinite(avg_near_seg) else None
                far_avg = float(avg_far_seg) if np.isfinite(avg_far_seg) else None
                _log(log, f"[save] player heatmap -> {out_players_heatmap}")
            except Exception as e:
                _log(log, f"[players] export skipped: {e}")

        set_progress(1.0, "done")
        video_for_web = out_video_web if out_video_web.exists() else out_video
        return {
            "reproj_error_px": float(getattr(calib, "reproj_error", 0.0)),
            "video_fps": float(getattr(calib, "fps", 30.0)),
            "shots": int(len(shots)),
            "success": int(success_count),
            "near_player_distance_m": near_dist,
            "far_player_distance_m": far_dist,
            "near_player_avg_speed_mps": near_avg,
            "far_player_avg_speed_mps": far_avg,
            "artifacts": {
                "csv": str(out_csv) if out_csv.exists() else "",
                "mp4": str(video_for_web) if video_for_web.exists() else "",
                "png": str(out_png) if out_png.exists() else "",
                "heatmap": str(out_players_heatmap) if out_players_heatmap.exists() else "",
                "reproj_metrics": str(out_reproj_json) if out_reproj_json.exists() else "",
                "traj3d": str(panel_pngs["traj3d"]) if panel_pngs.get("traj3d", Path()).exists() else (str(out_png) if out_png.exists() else ""),
                "height_time": str(panel_pngs["height_time"]) if panel_pngs.get("height_time", Path()).exists() else (str(out_png) if out_png.exists() else ""),
                "side_zy": str(panel_pngs["side_zy"]) if panel_pngs.get("side_zy", Path()).exists() else (str(out_png) if out_png.exists() else ""),
                "top_xy": str(panel_pngs["top_xy"]) if panel_pngs.get("top_xy", Path()).exists() else (str(out_png) if out_png.exists() else ""),
                "side_zx": str(panel_pngs["side_zx"]) if panel_pngs.get("side_zx", Path()).exists() else (str(out_png) if out_png.exists() else ""),
                "overlay_panel": str(panel_pngs["overlay_panel"]) if panel_pngs.get("overlay_panel", Path()).exists() else "",
                "overlay_ball": str(video_for_web) if video_for_web.exists() else "",
            },
        }
    except Exception as e:
        _log(log, f"[error] {e}")
        tb = traceback.format_exc()
        _log(log, tb)
        raise

