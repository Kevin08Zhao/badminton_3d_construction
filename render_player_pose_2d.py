#!/usr/bin/env python3
"""
独立程序：用 YOLOv8-pose 对视频中的人体关键点进行追踪（track），
绘制 COCO-17 关键点与骨架、左右脚踝连线及踝间中点，输出 2D 标注视频。

用法示例：
  python render_player_pose_2d.py --video data/video/test0.mp4 --out result/pose2d_out.mp4
  python render_player_pose_2d.py --video data/video/test0.mp4 --model yolov8m-pose.pt --no-skeleton
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as e:
    print("需要安装 ultralytics：pip install ultralytics", file=sys.stderr)
    raise SystemExit(1) from e


# COCO 17 关键点骨架连线（与 Ultralytics pose 一致）
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# 调色板：按 track_id 取色（BGR）
def _color_for_id(tid: int) -> tuple[int, int, int]:
    rng = np.random.RandomState(17 + tid * 997)
    return tuple(int(x) for x in rng.randint(40, 255, size=3))


def _valid_player_mask(
    xy: np.ndarray,
    conf: Optional[np.ndarray],
    min_ankle_conf: float,
) -> np.ndarray:
    """Return boolean mask (N,) — True for detections with reliable ankle keypoints.

    A detection is considered valid when:
      1. Both ankle xy are non-zero, AND
      2. If *conf* is provided, the mean confidence of the two ankle keypoints >= *min_ankle_conf*.

    This filters out non-player detections (umpire, line judge, …) whose ankle keypoints
    are hallucinated with very low confidence.
    """
    n = xy.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    ankles_nonzero = np.all(xy[:, [15, 16]] != 0, axis=(1, 2))  # (N,)
    if conf is not None and min_ankle_conf > 0:
        ankle_conf_mean = conf[:, [15, 16]].mean(axis=1)  # (N,)
        return ankles_nonzero & (ankle_conf_mean >= min_ankle_conf)
    return ankles_nonzero


MIN_ANKLE_CONF = 0.3


def _assign_near_far_idx(
    xy: np.ndarray,
    conf: Optional[np.ndarray] = None,
    min_ankle_conf: float = MIN_ANKLE_CONF,
) -> tuple[int | None, int | None]:
    """xy: (N, 17, 2)，与 main.py 一致：踝部平均像素 Y 大者为近端。

    Only detections that pass ``_valid_player_mask`` are considered so that
    non-player persons (umpire, line judge …) are ignored.
    """
    n = xy.shape[0]
    if n < 2:
        return None, None
    mask = _valid_player_mask(xy, conf, min_ankle_conf)
    valid_idx = np.where(mask)[0]
    if len(valid_idx) < 2:
        return None, None
    valid_xy = xy[valid_idx]
    ankle_y = valid_xy[:, [15, 16], 1].mean(axis=1)
    near_local = int(np.argmax(ankle_y))
    far_local = int(np.argmin(ankle_y))
    return int(valid_idx[near_local]), int(valid_idx[far_local])


def pack_near_far_pose_arrays(
    xy: np.ndarray,
    frame_height: int,
    conf: Optional[np.ndarray] = None,
    min_ankle_conf: float = MIN_ANKLE_CONF,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将一帧内 YOLO 检出的多人关键点整理为 (pose_near, pose_far)，每者形状 (17, 2)，与 main.py / HitNet 约定一致。

    Only detections that pass ankle-confidence filtering are considered (see
    ``_valid_player_mask``).  This avoids assigning non-player persons (umpire,
    line judge …) as the near/far player.

    - ≥2 valid: 踝部平均 Y 最大者为 near，最小者为 far。
    - 1 valid: 踝部平均 Y > 帧高一半 → near，否则 far（另一半置零）。
    - 0 valid: 两者全零。
    """
    pose_near = np.zeros((17, 2), dtype=np.float32)
    pose_far = np.zeros((17, 2), dtype=np.float32)
    n = xy.shape[0]
    if n == 0:
        return pose_near, pose_far

    mask = _valid_player_mask(xy, conf, min_ankle_conf)
    valid_idx = np.where(mask)[0]

    if len(valid_idx) >= 2:
        valid_xy = xy[valid_idx]
        ankle_y = valid_xy[:, [15, 16], 1].mean(axis=1)
        near_local = int(np.argmax(ankle_y))
        far_local = int(np.argmin(ankle_y))
        pose_near = valid_xy[near_local].astype(np.float32)
        pose_far = valid_xy[far_local].astype(np.float32)
        return pose_near, pose_far

    if len(valid_idx) == 1:
        i = valid_idx[0]
        ay = float(xy[i, [15, 16], 1].mean())
        if ay > frame_height / 2.0:
            pose_near = xy[i].astype(np.float32)
        else:
            pose_far = xy[i].astype(np.float32)
        return pose_near, pose_far

    return pose_near, pose_far


# 单帧 track 结果：xy (N,17,2)，conf (N,17) 或 None，track id (N,) 或 None
TrackDrawPack = Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]


def draw_track_results_on_frame(
    frame: np.ndarray,
    xy: np.ndarray,
    conf: Optional[np.ndarray],
    ids: Optional[np.ndarray],
    *,
    conf_thr: float = 0.25,
    draw_skeleton: bool = True,
) -> None:
    """
    与 CLI 渲染循环一致：对每个 track 调用 draw_single_person（全身骨架 + 踝线 + 中点）。
    """
    if xy is None or len(xy) == 0:
        return
    for j in range(xy.shape[0]):
        tid = int(ids[j]) if ids is not None and j < len(ids) else j
        color = _color_for_id(tid)
        kc = conf[j] if conf is not None else None
        draw_single_person(frame, xy[j], kc, color, conf_thr, draw_skeleton)


def extract_poses_by_frame_from_track(
    video_path: str,
    model_path: str,
    *,
    conf_thr: float = 0.25,
    device: str | None = None,
    tracker: str = "bytetrack.yaml",
    progress_every: int = 50,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], Dict[int, Optional[TrackDrawPack]]]:
    """
    YOLO pose + ByteTrack 扫视频。
    返回 (poses_by_frame, draw_by_frame)：前者为 packed near/far（unproject/CSV）；后者为每帧原始多人
    关键点+置信度+track id，供 draw_track_results_on_frame 叠加。main.py 中可与 HitNet 共用同一次调用结果。
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    model = YOLO(model_path)
    kwargs: dict = {"conf": conf_thr, "verbose": False}
    if device:
        kwargs["device"] = device

    results_iter = model.track(
        source=video_path,
        stream=True,
        persist=True,
        tracker=tracker,
        **kwargs,
    )

    poses_by_frame: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    draw_by_frame: Dict[int, Optional[TrackDrawPack]] = {}
    frame_idx = 0

    for result in results_iter:
        draw_pack: Optional[TrackDrawPack] = None
        if result.orig_img is None:
            pose_near = np.zeros((17, 2), dtype=np.float32)
            pose_far = np.zeros((17, 2), dtype=np.float32)
        else:
            h = result.orig_img.shape[0]
            if result.keypoints is None or len(result.keypoints) == 0:
                pose_near = np.zeros((17, 2), dtype=np.float32)
                pose_far = np.zeros((17, 2), dtype=np.float32)
            else:
                xy = result.keypoints.xy.cpu().numpy()
                kconf = None
                if result.keypoints.conf is not None:
                    kconf = result.keypoints.conf.cpu().numpy()
                pose_near, pose_far = pack_near_far_pose_arrays(xy, h, conf=kconf)
                tid_arr = None
                if result.boxes is not None and result.boxes.id is not None:
                    tid_arr = result.boxes.id.cpu().numpy().astype(np.int64)
                draw_pack = (xy, kconf, tid_arr)

        poses_by_frame[frame_idx] = (pose_near, pose_far)
        draw_by_frame[frame_idx] = draw_pack
        frame_idx += 1
        if progress_every and frame_idx % progress_every == 0:
            print(f"  [track 姿态] 已处理 {frame_idx} 帧...")

    print(f"  [track 姿态] 完成，共 {frame_idx} 帧")
    return poses_by_frame, draw_by_frame


def draw_single_person(
    frame: np.ndarray,
    kpt_xy: np.ndarray,
    kpt_conf: np.ndarray | None,
    color: tuple[int, int, int],
    conf_thr: float,
    draw_skeleton: bool,
    ankle_line_thickness: int = 3,
    ankle_mid_radius: int = 8,
) -> None:
    """kpt_xy (17,2), kpt_conf (17,) optional."""
    h, w = frame.shape[:2]

    def ok(i: int) -> bool:
        if kpt_xy[i, 0] <= 0 and kpt_xy[i, 1] <= 0:
            return False
        if kpt_conf is not None and float(kpt_conf[i]) < conf_thr:
            return False
        x, y = int(kpt_xy[i, 0]), int(kpt_xy[i, 1])
        return 0 <= x < w and 0 <= y < h

    if draw_skeleton:
        for a, b in SKELETON_EDGES:
            if ok(a) and ok(b):
                pa = (int(kpt_xy[a, 0]), int(kpt_xy[a, 1]))
                pb = (int(kpt_xy[b, 0]), int(kpt_xy[b, 1]))
                cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)

        for i in range(17):
            if ok(i):
                p = (int(kpt_xy[i, 0]), int(kpt_xy[i, 1]))
                cv2.circle(frame, p, 4, color, -1, cv2.LINE_AA)
                cv2.circle(frame, p, 4, (255, 255, 255), 1, cv2.LINE_AA)

    # 踝 15、16 连线 + 中点（始终绘制，若两点有效）
    if ok(15) and ok(16):
        L = (int(kpt_xy[15, 0]), int(kpt_xy[15, 1]))
        R = (int(kpt_xy[16, 0]), int(kpt_xy[16, 1]))
        cv2.line(frame, L, R, color, ankle_line_thickness, cv2.LINE_AA)
        m = ((L[0] + R[0]) // 2, (L[1] + R[1]) // 2)
        cv2.circle(frame, m, ankle_mid_radius, color, 2, cv2.LINE_AA)
        cv2.circle(frame, m, 3, (255, 255, 255), -1, cv2.LINE_AA)


def run(
    video_path: str,
    out_path: str,
    model_path: str,
    device: str | None,
    conf_thr: float,
    draw_skeleton: bool,
    tracker: str,
) -> None:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    kwargs = {"conf": conf_thr, "verbose": False}
    if device:
        kwargs["device"] = device

    print(f"输入: {video_path}  ({width}x{height} @ {fps:.2f} fps, ~{nframes} 帧)")
    print(f"模型: {model_path}  tracker: {tracker}")
    print(f"输出: {out_path}")

    fi = 0
    # stream=True 逐帧处理，避免一次性占满内存
    results_iter = model.track(
        source=video_path,
        stream=True,
        persist=True,
        tracker=tracker,
        **kwargs,
    )

    for result in results_iter:
        frame = result.orig_img
        if frame is None:
            continue
        out = frame.copy()

        if result.keypoints is None or len(result.keypoints) == 0:
            writer.write(out)
            fi += 1
            if fi % 50 == 0:
                print(f"  已处理 {fi} 帧...")
            continue

        xy = result.keypoints.xy.cpu().numpy()
        conf = None
        if result.keypoints.conf is not None:
            conf = result.keypoints.conf.cpu().numpy()

        ids = None
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)

        near_i, far_i = _assign_near_far_idx(xy, conf)
        n = xy.shape[0]

        for j in range(n):
            tid = int(ids[j]) if ids is not None and j < len(ids) else j
            color = _color_for_id(tid)
            kc = conf[j] if conf is not None else None
            draw_single_person(out, xy[j], kc, color, conf_thr, draw_skeleton)

            # 标签：track id + 近/远端（仅当本帧恰好两人）
            parts = [f"id{tid}"]
            if near_i is not None and far_i is not None:
                if j == near_i:
                    parts.append("NEAR")
                elif j == far_i:
                    parts.append("FAR")
            label = " ".join(parts)
            # 用髋部中点或第一个有效点挂文字
            root = xy[j, 11:13].mean(axis=0)
            if root[0] > 0 or root[1] > 0:
                tx, ty = int(root[0]), int(max(20, root[1] - 10))
                cv2.putText(
                    out,
                    label,
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    out,
                    label,
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        writer.write(out)
        fi += 1
        if fi % 50 == 0:
            print(f"  已处理 {fi} 帧...")

    writer.release()
    print(f"完成，共写入 {fi} 帧 -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="YOLO pose 追踪 + 2D 骨架/踝线/中点渲染视频")
    p.add_argument("--video", "-v", required=True, help="输入视频路径")
    p.add_argument("--out", "-o", default="", help="输出 mp4（默认：与输入同目录，文件名加 _pose2d）")
    p.add_argument("--model", "-m", default="yolov8x-pose.pt", help="Ultralytics pose 权重路径或名称")
    p.add_argument("--device", default="", help="cuda:0 / cpu / mps，留空则自动")
    p.add_argument("--conf", type=float, default=0.25, help="关键点置信度阈值")
    p.add_argument("--no-skeleton", action="store_true", help="只画踝连线与中点，不画全身骨架")
    p.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Ultralytics 追踪配置（如 bytetrack.yaml）；可改为 botsort.yaml",
    )
    args = p.parse_args()

    video_path = args.video
    out_path = args.out
    if not out_path:
        base, ext = os.path.splitext(video_path)
        out_path = f"{base}_pose2d{ext or '.mp4'}"

    device = args.device.strip() or None
    try:
        run(
            video_path=video_path,
            out_path=out_path,
            model_path=args.model,
            device=device,
            conf_thr=args.conf,
            draw_skeleton=not args.no_skeleton,
            tracker=args.tracker,
        )
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
