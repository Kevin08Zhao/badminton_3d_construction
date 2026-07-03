"""
可编程调用的完整分析流水线（对应用户手册：标定 → TrackNet → Pose → HitNet → 3D → 导出）。
供 FastAPI 或其它脚本调用；交互式标定请仍使用 main.main()。
"""

from __future__ import annotations

import json
import os
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LogFn = Callable[[str], None]


def _compute_per_shot_reproj_metrics(
    shots: List[Any],
    trajectories_3d: List[Optional[np.ndarray]],
    calib: Any,
) -> Dict[str, Any]:
    """
    Compute per-shot reprojection RMS/mean (pixels) by projecting the optimised
    3D trajectory back through the calibrated camera and comparing against the
    visible TrackNet 2D observations.

    Output schema:
      {
        "calib_reproj_error_px": float,
        "n_shots": int,
        "n_reconstructed": int,
        "overall_rms_px": float, "overall_mean_px": float,
        "overall_median_px": float, "overall_p95_px": float,
        "overall_max_px": float, "overall_n_obs": int,
        "per_shot": [
          {"rally_id": int, "shot_number": int, "n_visible_obs": int,
           "reproj_rms_px": float, "reproj_mean_px": float,
           "reproj_median_px": float, "reproj_max_px": float}, ...
        ],
      }
    """
    per_shot: List[Dict[str, Any]] = []
    all_errs: List[float] = []
    n_reconstructed = 0
    for shot, traj_3d in zip(shots, trajectories_3d):
        if traj_3d is None or len(traj_3d) == 0:
            continue
        try:
            visible_mask = np.asarray(shot.is_visible, dtype=bool)
            if visible_mask.sum() == 0:
                continue
            visible_traj_2d = np.asarray(shot.traj_2d, dtype=float)[visible_mask]
            visible_frames = np.asarray(shot.frames, dtype=int)[visible_mask]
            rel_idx = visible_frames - int(shot.start_frame)
            valid = (rel_idx >= 0) & (rel_idx < len(traj_3d))
            if not np.any(valid):
                continue
            traj_subset = traj_3d[rel_idx[valid]]
            obs_subset = visible_traj_2d[valid]
            proj = calib.project(traj_subset)
            errs = np.linalg.norm(proj - obs_subset, axis=1)
            per_shot.append({
                "rally_id": int(getattr(shot, "rally_id", -1)),
                "shot_number": int(getattr(shot, "shot_number", -1)),
                "n_visible_obs": int(errs.size),
                "reproj_rms_px": float(np.sqrt(np.mean(errs ** 2))),
                "reproj_mean_px": float(np.mean(errs)),
                "reproj_median_px": float(np.median(errs)),
                "reproj_max_px": float(np.max(errs)),
            })
            all_errs.extend(errs.tolist())
            n_reconstructed += 1
        except Exception:
            continue

    if all_errs:
        arr = np.asarray(all_errs, dtype=float)
        overall = {
            "overall_rms_px": float(np.sqrt(np.mean(arr ** 2))),
            "overall_mean_px": float(np.mean(arr)),
            "overall_median_px": float(np.median(arr)),
            "overall_p95_px": float(np.percentile(arr, 95)),
            "overall_max_px": float(np.max(arr)),
            "overall_n_obs": int(arr.size),
        }
    else:
        overall = {
            "overall_rms_px": float("nan"),
            "overall_mean_px": float("nan"),
            "overall_median_px": float("nan"),
            "overall_p95_px": float("nan"),
            "overall_max_px": float("nan"),
            "overall_n_obs": 0,
        }

    return {
        "calib_reproj_error_px": float(getattr(calib, "reproj_error", float("nan"))),
        "n_shots": int(len(shots)),
        "n_reconstructed": int(n_reconstructed),
        "per_shot": per_shot,
        **overall,
    }


def run_full_pipeline(
    video_path: str,
    output_dir: str,
    calibration_points: List[Tuple[float, float]],
    *,
    hitnet_weights: str,
    tracknet_weights: str,
    pose_model: str,
    log: LogFn = print,
) -> Dict[str, Any]:
    """
    calibration_points: 6 个像素点 (x,y)，顺序与手册一致：
    远端左、远端右、近端左、近端右、左网柱、右网柱。
    返回结果字典（输出文件路径、指标等）；异常时抛出。
    """
    # 延迟导入，避免循环依赖
    from main import (
        CameraCalibrator,
        DataLoader,
        HitInferenceRunner,
        HitNetConfig,
        PhysicsModel,
        TrajectoryReconstructor,
        VideoRenderer,
        build_players_export,
        instantaneous_velocity_from_trajectory,
        save_player_movement_heatmap,
        visualize_results,
    )
    from render_player_pose_2d import extract_poses_by_frame_from_track
    from trackNetV3.prediction import predict_trajectory

    os.makedirs(output_dir, exist_ok=True)
    traj_csv = os.path.join(output_dir, "trajectory_2d.csv")
    output_csv = os.path.join(output_dir, "output_reconstructed_3d.csv")
    output_video = os.path.join(output_dir, "output_3d.mp4")
    output_chart = os.path.join(output_dir, "output_3d_result.png")
    output_players_csv = os.path.join(output_dir, "output_players_3d.csv")
    output_players_heatmap = os.path.join(output_dir, "output_players_heatmap.png")
    output_players_stats = os.path.join(output_dir, "output_players_stats.csv")
    output_reproj_json = os.path.join(output_dir, "output_reproj_metrics.json")

    log("=== ShuttleVision 流水线开始 ===")
    calib = CameraCalibrator(video_path)
    try:
        calib.set_calibration_from_pixels(calibration_points)
    finally:
        if calib.cap is not None:
            calib.cap.release()

    log(f"标定完成: 重投影误差 {calib.reproj_error:.2f} px")
    court_corners = [[x[0], x[1]] for i, x in enumerate(calib.points_2d_original) if i < 4]

    log("[TrackNet] 生成 2D 轨迹…")
    predict_trajectory(
        video_file=video_path,
        tracknet_file=tracknet_weights,
        inpaintnet_file=None,
        batch_size=4,
        eval_mode="nonoverlap",
        large_video=True,
        output_video=False,
        save_dir=output_dir,
        out_csv_file=traj_csv,
        return_dict=False,
    )

    poses_by_frame = {}
    track_draw_by_frame = {}
    log("[Pose+ByteTrack] 球员姿态…")
    try:
        poses_by_frame, track_draw_by_frame = extract_poses_by_frame_from_track(
            video_path,
            pose_model,
            conf_thr=0.25,
            device=None,
            tracker="bytetrack.yaml",
        )
    except Exception as e:
        log(f"  警告: pose 跟踪失败 — {e}")

    log("[HitNet] 击球检测…")
    hit_config = HitNetConfig(
        court_corners=court_corners,
        video_path=video_path,
        weights_path=hitnet_weights,
        POSE_model=pose_model,
        fps=int(round(calib.fps)),
    )
    inference_runner = HitInferenceRunner(hit_config, poses_by_frame=poses_by_frame or None)
    try:
        loader = DataLoader(traj_csv, inference_runner=inference_runner)
        shots = loader.get_shots()
        if len(shots) == 0:
            raise RuntimeError("未分割到任何有效 shots")
    finally:
        if hasattr(inference_runner, "release"):
            inference_runner.release()

    physics = PhysicsModel()
    reconstructor = TrajectoryReconstructor(calib, physics, fps=calib.fps, scale_y=1.0)
    trajectories_3d: List[Optional[np.ndarray]] = []
    success_count = 0

    log("[3D] 重建…")
    for i, shot in enumerate(shots):
        traj = reconstructor.reconstruct(shot)
        trajectories_3d.append(traj)
        if traj is None:
            continue
        success_count += 1
        shot.frame_predictions = {}
        for frame_idx in range(shot.start_frame, shot.end_frame + 1):
            current_idx = frame_idx - shot.start_frame
            if current_idx >= len(traj):
                break
            landing_pos, score_result, score_reason, confidence = reconstructor.predict_landing_realtime(
                shot, current_idx
            )
            valid_points = sum(1 for j in range(current_idx + 1) if j < len(traj))
            if valid_points < 5:
                method = "early_phase"
            elif shot.is_last_in_rally and current_idx > len(traj) * 0.8:
                method = "late_phase"
            else:
                method = "middle_ransac"
            shot.frame_predictions[frame_idx] = {
                "pos": landing_pos,
                "method": method,
                "result": score_result,
                "reason": score_reason,
                "confidence": confidence,
            }
        final_frame = shot.end_frame
        if final_frame in shot.frame_predictions:
            fp = shot.frame_predictions[final_frame]
            shot.predicted_landing = fp["pos"]
            shot.prediction_method = fp["method"]
            shot.score_result = fp["result"]
            shot.score_reason = fp["reason"]
            actual_landing = reconstructor.get_landing_point_from_trajectory(traj)
            shot.actual_landing = actual_landing
            if actual_landing is not None and shot.predicted_landing is not None:
                shot.prediction_error = float(
                    np.linalg.norm(shot.predicted_landing - actual_landing)
                )

    log(f"重建统计: {success_count}/{len(shots)} 成功")

    output_data = []
    for shot, traj in zip(shots, trajectories_3d):
        if traj is None:
            continue
        vel = instantaneous_velocity_from_trajectory(traj, calib.fps)
        for j in range(len(traj)):
            frame_num = shot.start_frame + j
            pred_x = pred_y = None
            actual_x = actual_y = None
            error_val = None
            method_val = None
            conf_val = None
            if frame_num in shot.frame_predictions:
                pred = shot.frame_predictions[frame_num]
                pred_x, pred_y = pred["pos"][0], pred["pos"][1]
                method_val = pred["method"]
                conf_val = pred["confidence"]
            if j == len(traj) - 1:
                if shot.actual_landing is not None:
                    actual_x = shot.actual_landing[0]
                    actual_y = shot.actual_landing[1]
                error_val = shot.prediction_error
            pos = traj[j]
            vvec = vel[j]
            spd = float(np.linalg.norm(vvec))
            output_data.append(
                {
                    "rally_id": shot.rally_id,
                    "shot_number": shot.shot_number,
                    "frame": frame_num,
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "z": float(pos[2]),
                    "vx": float(vvec[0]),
                    "vy": float(vvec[1]),
                    "vz": float(vvec[2]),
                    "speed_mps": spd,
                    "hitter": shot.hitter,
                    "is_last_in_rally": 1 if shot.is_last_in_rally else 0,
                    "predicted_landing_x": pred_x,
                    "predicted_landing_y": pred_y,
                    "prediction_method": method_val,
                    "prediction_confidence": conf_val,
                    "actual_landing_x": actual_x,
                    "actual_landing_y": actual_y,
                    "prediction_error": error_val,
                    "score_result": shot.score_result if j == len(traj) - 1 else None,
                    "score_reason": shot.score_reason if j == len(traj) - 1 else None,
                }
            )

    if output_data:
        pd.DataFrame(output_data).to_csv(output_csv, index=False)

    reproj_metrics = _compute_per_shot_reproj_metrics(shots, trajectories_3d, calib)
    try:
        with open(output_reproj_json, "w", encoding="utf-8") as f:
            json.dump(reproj_metrics, f, ensure_ascii=False, indent=2)
        log(
            f"[reproj] calib={reproj_metrics['calib_reproj_error_px']:.2f}px"
            f" | per-shot RMS overall mean={reproj_metrics['overall_mean_px']:.2f}px"
            f" median={reproj_metrics['overall_median_px']:.2f}px"
            f" n_obs={reproj_metrics['overall_n_obs']}"
            f" -> {output_reproj_json}"
        )
    except Exception as e:
        log(f"  警告: reproj metrics 写入失败 — {e}")

    metrics: Dict[str, Any] = {
        "hits_shots": len(shots),
        "rebuild_success": success_count,
        "reproj_error_px": float(calib.reproj_error),
        "shuttle_reproj_mean_px": reproj_metrics["overall_mean_px"],
        "shuttle_reproj_median_px": reproj_metrics["overall_median_px"],
        "shuttle_reproj_rms_px": reproj_metrics["overall_rms_px"],
        "shuttle_reproj_p95_px": reproj_metrics["overall_p95_px"],
        "shuttle_reproj_n_obs": reproj_metrics["overall_n_obs"],
        "total_path_near_m": None,
        "total_path_far_m": None,
        "avg_speed_segment_near_mps": None,
        "avg_speed_segment_far_mps": None,
    }

    if poses_by_frame and calib.P is not None:
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
            players_df.to_csv(output_players_csv, index=False)
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
            ).to_csv(output_players_stats, index=False)
            save_player_movement_heatmap(
                players_df,
                output_players_heatmap,
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
            metrics.update(
                {
                    "total_path_near_m": float(total_near_m),
                    "total_path_far_m": float(total_far_m),
                    "avg_speed_segment_near_mps": float(avg_near_seg)
                    if np.isfinite(avg_near_seg)
                    else None,
                    "avg_speed_segment_far_mps": float(avg_far_seg)
                    if np.isfinite(avg_far_seg)
                    else None,
                }
            )
        except Exception as e:
            log(f"  警告: 球员导出失败 — {e}\n{traceback.format_exc()}")

    if success_count > 0:
        try:
            renderer = VideoRenderer(video_path, output_video, calib)
            renderer.render(
                shots,
                trajectories_3d,
                track_draw_by_frame=track_draw_by_frame or None,
                pose_render_conf=0.25,
                pose_draw_skeleton=True,
            )
            visualize_results(video_path, shots, trajectories_3d, calib, output_chart)
        except Exception as e:
            log(f"  警告: 渲染失败 — {e}\n{traceback.format_exc()}")

    log("=== ShuttleVision 流水线结束 ===")
    return {
        "output_dir": output_dir,
        "files": {
            "trajectory_2d": traj_csv,
            "reconstructed_3d": output_csv,
            "video": output_video,
            "chart": output_chart,
            "players_csv": output_players_csv,
            "players_heatmap": output_players_heatmap,
            "players_stats": output_players_stats,
            "reproj_metrics": output_reproj_json,
        },
        "metrics": metrics,
    }
