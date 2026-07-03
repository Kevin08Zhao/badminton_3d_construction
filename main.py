import os
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import minimize, least_squares
from scipy.integrate import odeint
from scipy.ndimage import gaussian_filter
import matplotlib

# 非主线程（如 FastAPI 后台任务）在 macOS 上不能用 GUI 后端；默认 Agg 仅保存文件。
# 需要交互窗口时在运行前设置环境变量：MPLBACKEND=MacOSX
matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import warnings
import torch
import torch.nn as nn
from ultralytics import YOLO
from trackNetV3.prediction import predict_trajectory
from render_player_pose_2d import draw_track_results_on_frame, extract_poses_by_frame_from_track

# 过滤掉数值计算过程中可能出现的警告信息（如除零警告、无效值警告等），保持输出整洁
warnings.filterwarnings('ignore')

# ==================== 1. 推理模型定义（击球检测） ====================
"""
【模块说明】
本模块定义了基于深度学习的击球检测网络（HitNet）。
功能：从视频帧序列中检测羽毛球何时被击打（hit），以及由哪位球员击打（near/far）。
输入：连续12帧的特征向量（包含场地角点、球员姿态、羽毛球位置）
输出：击球类别（无击球/near球员击球/far球员击球）、是否为最后一击
"""

class HitNetConfig:
    """
    HitNet模型配置类
    存储模型推理所需的配置参数，包括场地角点、视频路径、模型权重、帧率等

    属性说明：
    - hitnet_weights: 训练好的PyTorch模型权重文件路径(.pth)
    - video_path: 输入视频文件路径
    - court_corners: 场地四个角点在图像中的像素坐标，用于特征归一化
    - fps: 视频帧率，影响时间特征计算和最小击球间隔转换
    - seq_len: 输入序列长度，模型要求固定12帧的时序输入
    - stride: 滑动窗口步长，决定推理时的采样密度（2表示每隔2帧做一次预测）
    - conf_threshold: 置信度过滤阈值（0表示保留所有预测，后续通过后处理筛选）
    - min_hit_interval: 两次击球的物理时间间隔下限，用于去除重复检测
    - device: 计算设备（优先使用GPU加速，否则回退到CPU）
    """

    def __init__(self, court_corners, video_path, weights_path, POSE_model, fps=30):
        self.hitnet_weights = weights_path  # 模型权重文件路径
        self.video_path = video_path  # 输入视频路径
        self.court_corners = court_corners  # 场地四个角点坐标（用于特征归一化）
        self.fps = fps  # 视频帧率
        self.seq_len = 12  # 输入序列长度（12帧），与模型训练时的输入维度一致
        self.stride = 2  # 滑动窗口步长，控制推理密度（步长越大计算越快但可能漏检）
        self.conf_threshold = 0.0  # 置信度阈值（0表示不过滤，保留所有候选供后续NMS处理）
        self.min_hit_interval = 0.1  # 两次击球最小间隔（秒），用于去重（0.1秒=100毫秒）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动检测GPU
        self.POSE_model = POSE_model


class HitNetOverfit(nn.Module):
    """
    击球检测神经网络架构
    基于GRU（门控循环单元）的序列模型，用于从时序特征中识别击球事件

    网络结构解析：
    1. Embedding层：将输入特征（78维）映射到128维，通过两层全连接+ReLU激活提取高阶特征
    2. GRU层：3层GRU网络深度建模时序依赖关系，捕捉击球动作的时间上下文
    3. 分类头：
       - hit_classifier：三分类输出（0=无击球，1=near击球，2=far击球）
       - last_classifier：二分类输出（是否为回合最后一击，用于区分回合结束）

    输入维度说明：
    - 场地角点：6个点×2坐标=12维（实际使用4个角点+2个网柱，共6个点）
    - 近端球员姿态：17个关键点×2坐标=34维（COCO格式人体姿态）
    - 远端球员姿态：34维
    - 羽毛球位置：2维（x,y像素坐标）
    - 总计：12+34+34+2=82维（代码中实际使用78维，可能有所简化）
    """

    def __init__(self, input_dim=78, hidden_dim=128, num_layers=3):
        super().__init__()
        # 特征嵌入层：将原始输入映射到高维空间，增强表达能力
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),  # 第一层：78→64，非线性激活
            nn.Linear(64, 128), nn.ReLU()  # 第二层：64→128，得到高维特征表示
        )
        # GRU循环神经网络：捕捉时序依赖关系
        # input_size=128对应embedding输出，hidden_size=128，num_layers=3表示堆叠3层GRU
        # batch_first=True表示输入格式为[batch, seq_len, features]
        # dropout=0表示训练时不使用dropout（overfit版本可能 intentionally 不过拟合）
        self.gru = nn.GRU(128, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=0)
        # 击球分类器：输出3个类别的logits（未归一化的概率分数）
        self.hit_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),  # 降维到64维
            nn.Linear(64, 3)  # 输出3类分类结果
        )
        # 最后一击分类器：输出是否为回合结束的置信度（单个数值，经sigmoid后转为概率）
        self.last_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),  # 降维到32维
            nn.Linear(32, 1)  # 输出1维 logits
        )

    def forward(self, x):
        # x形状: [batch_size, seq_len, input_dim]，seq_len固定为12帧
        x = self.embedding(x)  # [batch, seq_len, 128]，逐帧特征变换
        out, _ = self.gru(x)  # out: [batch, seq_len, hidden_dim]，提取时序特征
        last_hidden = out[:, -1, :]  # 取最后一帧的隐藏状态作为序列整体表示
        # 分别通过两个分类头得到击球类别和最后一击判断
        return self.hit_classifier(last_hidden), self.last_classifier(last_hidden)


class InferFeatureExtractor:
    """
    推理阶段特征提取器
    从视频帧中提取78维特征向量，包括：
    - 场地角点特征（12维，6个点×2坐标，归一化到1920）
    - 近端球员姿态（34维，17个关键点×2坐标）
    - 远端球员姿态（34维）
    - 羽毛球位置（2维）
    总计：12+34+34+2=82维（实际代码使用78维，可能有特定筛选）

    姿态检测：默认 YOLOv8-pose 逐帧推理；若传入 poses_by_frame（通常来自 extract_poses_by_frame_from_track
    的 ByteTrack 结果），则按帧号复用缓存，避免整段视频第二次跑 pose，仅在缺帧时惰性加载 YOLO 回退。
    羽毛球检测：基于HSV颜色空间（白色）的轮廓检测，利用羽毛球颜色特性进行简单分割
    """

    def __init__(
        self,
        court_corners,
        POSE_model,
        poses_by_frame: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
    ):
        # 场地角点归一化（除以1920进行标准化，假设视频分辨率为1920×1080）
        # flatten将6×2的坐标展平为12维向量，astype确保32位浮点数精度
        self.court = np.array(court_corners).flatten().astype(np.float32) / 1920.0
        self.poses_by_frame = poses_by_frame
        self._pose_model_path = POSE_model
        # 有预计算姿态时推迟加载 YOLO，整段命中缓存则完全不加载，省显存与时间
        self.pose_model = None if poses_by_frame is not None else YOLO(POSE_model)

    def _lazy_pose_model(self):
        if self.pose_model is None:
            self.pose_model = YOLO(self._pose_model_path)
        return self.pose_model

    def detect_shuttle(self, frame):
        """
        检测羽毛球在图像中的位置
        方法：HSV颜色阈值分割检测白色物体，选择最接近画面中心的轮廓

        HSV色彩空间说明：
        - H（色调）：白色在0-180范围内分布，故设为0-180
        - S（饱和度）：白色饱和度低，设为0-30（排除鲜艳颜色）
        - V（亮度）：白色亮度高，设为200-255（确保检测到亮白色物体）

        几何筛选逻辑：
        - 遍历所有白色轮廓，计算其重心（moment）
        - 选择距离画面中心最近的白点（假设羽毛球通常在画面中央附近飞行）
        - 过滤太小的噪声（m00>10确保足够面积）

        返回：羽毛球中心点坐标（像素，float32格式）
        """
        # 转换到HSV颜色空间，分离亮度和色度信息，便于白色物体检测
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 白色物体的HSV范围：低饱和度(0-30)，高亮度(200-255)
        # 色调范围设为0-180涵盖所有可能的白色色调（实际白色色调任意，饱和度才是关键）
        mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
        # 查找轮廓：RETR_EXTERNAL只检测外层轮廓，CHAIN_APPROX_SIMPLE压缩水平/垂直/对角线段
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            h, w = frame.shape[:2]
            center = np.array([w / 2, h / 2])  # 画面几何中心
            best, best_dist = center, float('inf')
            # 遍历轮廓，选择最接近画面中心的（基于羽毛球飞行轨迹通常在画面中央的假设）
            for c in contours:
                M = cv2.moments(c)  # 计算图像矩，用于求取重心
                if M["m00"] > 10:  # m00是轮廓面积，过滤太小的噪声点
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])  # 重心坐标
                    dist = np.sqrt((cx - center[0]) ** 2 + (cy - center[1]) ** 2)  # 欧氏距离
                    if dist < best_dist:
                        best_dist, best = dist, np.array([cx, cy])
            return best.astype(np.float32)
        # 未检测到则返回画面中心（作为默认值，避免后续计算错误）
        return np.array([frame.shape[1] / 2, frame.shape[0] / 2], dtype=np.float32)

    def extract(self, frame, frame_index: Optional[int] = None):
        """
        从单帧图像提取完整特征向量

        处理流程：
        1. 姿态：若提供 poses_by_frame 且 frame_index 命中，直接用 ByteTrack 与路程一致的打包结果；
           否则 YOLO 逐帧检测（与旧版一致）。
        2. 远近端：与 render_player_pose_2d.pack_near_far_pose_arrays 一致（≥2 人踝部 Y 极值）。
        3. 羽毛球检测：HSV白色检测
        4. 特征拼接：场地（12）+近端姿态（34）+远端姿态（34）+羽毛球（2）

        参数：
            frame_index: 视频中的绝对帧号（与 cv2 CAP_PROP_POS_FRAMES 一致），供姿态缓存查找。

        返回：
            feat: 78维特征向量（numpy float32数组）
            poses: 两个球员的17个关键点坐标（用于后续可视化）
        """
        pose_near = np.zeros((17, 2), dtype=np.float32)
        pose_far = np.zeros((17, 2), dtype=np.float32)
        used_cache = False
        if self.poses_by_frame is not None and frame_index is not None:
            cached = self.poses_by_frame.get(frame_index)
            if cached is not None:
                pose_near = np.asarray(cached[0], dtype=np.float32, order="C")
                pose_far = np.asarray(cached[1], dtype=np.float32, order="C")
                used_cache = True
        if not used_cache:
            results = self._lazy_pose_model()(frame, verbose=False)[0]
            if len(results.keypoints) >= 2:
                kpts = results.keypoints.xy.cpu().numpy()
                ankle_y = kpts[:, [15, 16], 1].mean(axis=1)
                near_idx, far_idx = ankle_y.argmax(), ankle_y.argmin()
                pose_near, pose_far = kpts[near_idx], kpts[far_idx]

        # 检测羽毛球位置（白色物体）
        shuttle_pos = self.detect_shuttle(frame)

        # 拼接所有特征（场地+近端姿态+远端姿态+羽毛球位置）
        # 所有空间坐标都除以1920进行归一化，使特征值范围集中在0-1附近，有利于神经网络训练/推理
        feat = np.concatenate([
            self.court,  # 12维，已在外部初始化时归一化
            pose_near.flatten() / 1920.0,  # 34维展平后归一化
            pose_far.flatten() / 1920.0,  # 34维
            shuttle_pos / 1920.0  # 2维
        ])
        return feat.astype(np.float32), (pose_near, pose_far)


def loose_optimize(predictions, min_gap=3):
    """
    击球预测后处理：非极大值抑制（NMS简化版）

    作用：去除过于密集的重复检测，保证两次击球间隔至少min_gap帧
    原理：击球是瞬时事件，同一击球不应在相邻帧被多次检测

    处理逻辑：
    1. 只保留置信度>0的非"无击球"预测（类别0为无击球，1为near，2为far）
    2. 按置信度排序，优先保留高置信度候选
    3. 按时间顺序遍历，只保留与上一个选中帧间隔>=min_gap的帧

    参数：
        predictions: 原始预测列表，每个元素包含frame, conf_0/1/2, is_last_prob
        min_gap: 最小帧间隔（如3帧@30fps=100ms，与min_hit_interval一致）

    返回：过滤后的击球列表，包含帧号、击球方、置信度、最后一击概率
    """
    candidates = []
    for pred in predictions:
        confs = [pred['conf_0'], pred['conf_1'], pred['conf_2']]
        best_cls = np.argmax(confs)  # 找出置信度最高的类别
        # 类别0表示无击球，1表示near，2表示far；仅保留有击球且置信度>0的
        if best_cls != 0 and confs[best_cls] > 0:
            candidates.append({
                'frame': pred['frame'],
                'hitter': 'near' if best_cls == 1 else 'far',
                'conf': confs[best_cls],
                'is_last_prob': pred['is_last_prob']
            })
    if not candidates:
        return []

    candidates.sort(key=lambda x: x['frame'])  # 按时间顺序排序
    filtered = [candidates[0]]  # 保留第一个候选
    for c in candidates[1:]:
        # 时间间隔过滤，避免重复检测同一击球（如挥拍动作持续多帧）
        if c['frame'] - filtered[-1]['frame'] >= min_gap:
            filtered.append(c)
    return filtered


def _pose_ankles_valid(pose: np.ndarray) -> bool:
    """COCO 15/16 为左右踝；全零表示未检测到。"""
    if pose is None or pose.shape[0] <= 16:
        return False
    return bool(np.any(pose[15]) and np.any(pose[16]))


def instantaneous_velocity_from_trajectory(traj: np.ndarray, fps: float) -> np.ndarray:
    """对均匀采样（间隔 1/fps 秒）的 3D 轨迹做数值微分，返回 [N,3] m/s。"""
    n = len(traj)
    v = np.zeros((n, 3), dtype=np.float64)
    if n == 0:
        return v
    dt = 1.0 / float(fps)
    if n == 1:
        return v
    v[0] = (traj[1] - traj[0]) / dt
    v[-1] = (traj[-1] - traj[-2]) / dt
    for j in range(1, n - 1):
        v[j] = (traj[j + 1] - traj[j - 1]) / (2.0 * dt)
    return v


def player_ankle_midpoint_on_ground(
    calib: 'CameraCalibrator',
    kpts_xy: np.ndarray,
) -> Optional[np.ndarray]:
    """
    球员地面位置：左脚踝(15)、右脚踝(16) 的 2D 像素分别反投到地面平面 Z=0，
    得到两点的 3D 坐标后再取算术平均作为中点；写入 CSV 时 Z 强制为 0（脚点高度默认 0）。
    不可与「先算像素中点再反投一次」混用。
    """
    if not _pose_ankles_valid(kpts_xy):
        return None
    try:
        u15, v15 = float(kpts_xy[15][0]), float(kpts_xy[15][1])
        u16, v16 = float(kpts_xy[16][0]), float(kpts_xy[16][1])
        p_l = calib.unproject_to_ground((u15, v15), z=0)
        p_r = calib.unproject_to_ground((u16, v16), z=0)
        mid = (np.asarray(p_l, dtype=np.float64) + np.asarray(p_r, dtype=np.float64)) / 2.0
        mid[2] = 0.0
        return mid
    except Exception:
        return None


def _medfilt1d(arr: np.ndarray, kernel: int = 5) -> np.ndarray:
    """1-D median filter that only touches finite entries; NaN stays NaN."""
    from scipy.signal import medfilt
    out = arr.copy()
    mask = np.isfinite(arr)
    if mask.sum() < kernel:
        return out
    out[mask] = medfilt(arr[mask], kernel_size=kernel)
    return out


def _smooth_player_positions(
    x: np.ndarray, y: np.ndarray, valid: np.ndarray,
    med_k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply moving-median smoothing to a player's XY ground positions.

    Only operates on valid entries; invalid entries remain NaN.
    Returns smoothed copies of (x, y).
    """
    sx, sy = x.copy(), y.copy()
    vm = valid.astype(bool)
    if vm.sum() < med_k:
        return sx, sy
    sx[vm] = _medfilt1d(x[vm], med_k)
    sy[vm] = _medfilt1d(y[vm], med_k)
    return sx, sy


_PLAYER_MAX_SPEED_MPS = 8.0
_NOISE_FLOOR_NEAR_M = 0.015
_NOISE_FLOOR_FAR_M = 0.045
_NET_Y = 6.7


def build_players_export(
    poses_by_frame: Dict[int, Tuple[np.ndarray, np.ndarray]],
    calib: 'CameraCalibrator',
    fps: float,
):
    """
    由逐帧 (pose_near, pose_far) 生成踝点像素、双踝中点 3D（地面）CSV，
    累计近/远端有效连续帧间路程（米），并计算两类平均标量速度（m/s）。

    改进：
    1. 对 3D 地面坐标先做 moving-median 平滑（window=5），消除关键点抖动；
    2. 逐帧位移低于透视感知噪声阈值时视为静止（远端阈值更高）；
    3. 单帧速度超过生理极限 (8 m/s) 时截断位移，防止异常帧污染统计。
    """
    sorted_frames = sorted(poses_by_frame.keys())

    # ---------- Pass 1: collect raw 3D midpoints ----------
    n_frames = len(sorted_frames)
    frame_arr = np.array(sorted_frames, dtype=np.int64)
    near_x_raw = np.full(n_frames, np.nan, dtype=np.float64)
    near_y_raw = np.full(n_frames, np.nan, dtype=np.float64)
    far_x_raw = np.full(n_frames, np.nan, dtype=np.float64)
    far_y_raw = np.full(n_frames, np.nan, dtype=np.float64)
    near_valid_arr = np.zeros(n_frames, dtype=np.int32)
    far_valid_arr = np.zeros(n_frames, dtype=np.int32)

    raw_row_data: List[dict] = []
    for i, f in enumerate(sorted_frames):
        pn, pf = poses_by_frame[f]
        nv = _pose_ankles_valid(pn)
        fv = _pose_ankles_valid(pf)

        near_l_u = near_l_v = near_r_u = near_r_v = float('nan')
        near_mx = near_my = near_mz = float('nan')
        if nv:
            near_l_u, near_l_v = float(pn[15][0]), float(pn[15][1])
            near_r_u, near_r_v = float(pn[16][0]), float(pn[16][1])
            mid_arr = player_ankle_midpoint_on_ground(calib, pn)
            if mid_arr is None:
                nv = False
            else:
                near_mx, near_my, near_mz = float(mid_arr[0]), float(mid_arr[1]), float(mid_arr[2])

        far_l_u = far_l_v = far_r_u = far_r_v = float('nan')
        far_mx = far_my = far_mz = float('nan')
        if fv:
            far_l_u, far_l_v = float(pf[15][0]), float(pf[15][1])
            far_r_u, far_r_v = float(pf[16][0]), float(pf[16][1])
            mid_arr = player_ankle_midpoint_on_ground(calib, pf)
            if mid_arr is None:
                fv = False
            else:
                far_mx, far_my, far_mz = float(mid_arr[0]), float(mid_arr[1]), float(mid_arr[2])

        if nv:
            near_x_raw[i] = near_mx
            near_y_raw[i] = near_my
            near_valid_arr[i] = 1
        if fv:
            far_x_raw[i] = far_mx
            far_y_raw[i] = far_my
            far_valid_arr[i] = 1

        raw_row_data.append({
            'frame': f,
            'near_ankle_l_u': near_l_u, 'near_ankle_l_v': near_l_v,
            'near_ankle_r_u': near_r_u, 'near_ankle_r_v': near_r_v,
            'near_valid': int(nv),
            'far_ankle_l_u': far_l_u, 'far_ankle_l_v': far_l_v,
            'far_ankle_r_u': far_r_u, 'far_ankle_r_v': far_r_v,
            'far_valid': int(fv),
        })

    # ---------- Pass 2: smooth XY positions ----------
    near_x_sm, near_y_sm = _smooth_player_positions(
        near_x_raw, near_y_raw, near_valid_arr, med_k=5,
    )
    far_x_sm, far_y_sm = _smooth_player_positions(
        far_x_raw, far_y_raw, far_valid_arr, med_k=5,
    )

    # ---------- Pass 3: accumulate path with noise/speed guards ----------
    dt = 1.0 / float(fps)
    max_seg = _PLAYER_MAX_SPEED_MPS * dt

    total_near_m = 0.0
    total_far_m = 0.0
    near_speeds: List[float] = []
    far_speeds: List[float] = []
    last_near_i: Optional[int] = None
    last_far_i: Optional[int] = None

    for i in range(n_frames):
        nv = near_valid_arr[i]
        fv = far_valid_arr[i]

        if nv and last_near_i is not None and frame_arr[i] - frame_arr[last_near_i] == 1:
            seg = float(np.hypot(near_x_sm[i] - near_x_sm[last_near_i],
                                 near_y_sm[i] - near_y_sm[last_near_i]))
            if seg < _NOISE_FLOOR_NEAR_M:
                seg = 0.0
            seg = min(seg, max_seg)
            total_near_m += seg
            near_speeds.append(seg * float(fps))
        if nv:
            last_near_i = i
        else:
            last_near_i = None

        if fv and last_far_i is not None and frame_arr[i] - frame_arr[last_far_i] == 1:
            seg = float(np.hypot(far_x_sm[i] - far_x_sm[last_far_i],
                                 far_y_sm[i] - far_y_sm[last_far_i]))
            avg_y = (far_y_sm[i] + far_y_sm[last_far_i]) / 2.0
            perspective_ratio = max(0.0, 1.0 - avg_y / (2.0 * _NET_Y))
            noise_floor = _NOISE_FLOOR_NEAR_M + (_NOISE_FLOOR_FAR_M - _NOISE_FLOOR_NEAR_M) * perspective_ratio
            if seg < noise_floor:
                seg = 0.0
            seg = min(seg, max_seg)
            total_far_m += seg
            far_speeds.append(seg * float(fps))
        if fv:
            last_far_i = i
        else:
            last_far_i = None

    # ---------- Build output DataFrame (uses smoothed midpoints) ----------
    for i in range(n_frames):
        rd = raw_row_data[i]
        rd['near_mid_x'] = float(near_x_sm[i]) if near_valid_arr[i] else float('nan')
        rd['near_mid_y'] = float(near_y_sm[i]) if near_valid_arr[i] else float('nan')
        rd['near_mid_z'] = 0.0 if near_valid_arr[i] else float('nan')
        rd['far_mid_x'] = float(far_x_sm[i]) if far_valid_arr[i] else float('nan')
        rd['far_mid_y'] = float(far_y_sm[i]) if far_valid_arr[i] else float('nan')
        rd['far_mid_z'] = 0.0 if far_valid_arr[i] else float('nan')

    df = pd.DataFrame(raw_row_data)

    def _avg_speed_over_span(side: str, total_path: float) -> float:
        sub = df[df[f'{side}_valid'] == 1]
        if len(sub) < 2:
            return float('nan')
        span_sec = (int(sub['frame'].max()) - int(sub['frame'].min())) / float(fps)
        if span_sec <= 1e-9:
            return float('nan')
        return float(total_path / span_sec)

    avg_near_segment = float(np.mean(near_speeds)) if near_speeds else float('nan')
    avg_far_segment = float(np.mean(far_speeds)) if far_speeds else float('nan')
    avg_near_span = _avg_speed_over_span('near', total_near_m)
    avg_far_span = _avg_speed_over_span('far', total_far_m)
    max_near_segment = float(np.max(near_speeds)) if near_speeds else float('nan')
    max_far_segment = float(np.max(far_speeds)) if far_speeds else float('nan')

    return (
        df,
        total_near_m,
        total_far_m,
        avg_near_segment,
        avg_far_segment,
        avg_near_span,
        avg_far_span,
        max_near_segment,
        max_far_segment,
    )


def _draw_badminton_court_topdown(
    ax,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    net_y: float = 6.7,
    short_from_net: float = 1.98,
    line_color: str = 'white',
    lw: float = 1.2,
) -> None:
    """俯视图白线球场（黑底）：外框、球网、双打发球区短线。"""
    # 外边框
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=line_color, linewidth=lw, linestyle='-')
    # 球网
    ax.axhline(net_y, color=line_color, linewidth=lw * 1.1, linestyle='--', alpha=0.95)
    y_ss_near = net_y + short_from_net
    y_ss_far = net_y - short_from_net
    if y0 <= y_ss_far <= y1:
        ax.plot([x0, x1], [y_ss_far, y_ss_far], color=line_color, linewidth=lw * 0.85, linestyle='-', alpha=0.9)
    if y0 <= y_ss_near <= y1:
        ax.plot([x0, x1], [y_ss_near, y_ss_near], color=line_color, linewidth=lw * 0.85, linestyle='-', alpha=0.9)
    # 中线（双打半场分界）
    mx = (x0 + x1) / 2.0
    ax.plot([mx, mx], [y0, y_ss_far], color=line_color, linewidth=lw * 0.75, linestyle='-', alpha=0.75)
    ax.plot([mx, mx], [y_ss_near, y1], color=line_color, linewidth=lw * 0.75, linestyle='-', alpha=0.75)


def save_player_movement_heatmap(
    players_df: pd.DataFrame,
    out_path: str,
    *,
    court_x: Tuple[float, float] = (0.0, 6.1),
    court_y: Tuple[float, float] = (0.0, 13.4),
    bins: Tuple[int, int] = (61, 134),
    smooth_sigma: float = 1.0,
    avg_near_segment_mps: float = float('nan'),
    avg_far_segment_mps: float = float('nan'),
    avg_near_span_mps: float = float('nan'),
    avg_far_span_mps: float = float('nan'),
    max_near_segment_mps: float = float('nan'),
    max_far_segment_mps: float = float('nan'),
    total_near_m: float = float('nan'),
    total_far_m: float = float('nan'),
    rally_label: str = '回合 1',
) -> None:
    """
    基于双踝中点 3D 地面坐标 (X,Y)：左栏为整场俯视图热力（远端半场红、近端半场蓝），
    右栏为同坐标散点；黑底白线球场，与范例一致。
    """
    if players_df is None or len(players_df) == 0:
        return

    net_y = 6.7

    def _series_xy(side: str):
        v = players_df[f'{side}_valid'].to_numpy() == 1
        x = players_df.loc[v, f'{side}_mid_x'].to_numpy(dtype=np.float64)
        y = players_df.loc[v, f'{side}_mid_y'].to_numpy(dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        return x[m], y[m]

    def _half_filter(side: str, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """远端（far）仅保留 Y<球网；近端（near）仅保留 Y>球网，用于半场热力。"""
        if side == 'far':
            m = ys < net_y
        else:
            m = ys > net_y
        return xs[m], ys[m]

    x0, x1 = court_x
    y0, y1 = court_y
    nx, ny = bins

    xf, yf = _series_xy('far')
    xn, yn = _series_xy('near')
    xf_h, yf_h = _half_filter('far', xf, yf)
    xn_h, yn_h = _half_filter('near', xn, yn)

    # 2D 密度（仅中点），半场约束后分别平滑，再合成 RGB：红=远、蓝=近
    Hf = np.zeros((nx, ny), dtype=np.float64)
    Hn = np.zeros((nx, ny), dtype=np.float64)
    if len(xf_h) >= 1:
        Hf, _, _ = np.histogram2d(xf_h, yf_h, bins=bins, range=[[x0, x1], [y0, y1]])
    if len(xn_h) >= 1:
        Hn, _, _ = np.histogram2d(xn_h, yn_h, bins=bins, range=[[x0, x1], [y0, y1]])

    if smooth_sigma > 0:
        Hf = gaussian_filter(Hf, sigma=smooth_sigma)
        Hn = gaussian_filter(Hn, sigma=smooth_sigma)

    def _norm01(h: np.ndarray) -> np.ndarray:
        m = float(np.max(h))
        if m <= 1e-12:
            return np.zeros_like(h)
        g = np.log1p(h) / np.log1p(m)
        return np.clip(g, 0.0, 1.0)

    R = _norm01(Hf)
    B = _norm01(Hn)
    # imshow: 行对应 Y，origin=upper 且 y 轴倒置后，图像上方为高 Y 还是低 Y 需与 set_ylim 一致
    rgb = np.stack([R.T, np.zeros_like(R.T), B.T], axis=-1)
    rgb = np.clip(rgb * 1.15, 0.0, 1.0)

    def _style_court_ax(ax) -> None:
        ax.set_facecolor('black')
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('场地宽度（米）', color='white', fontsize=11)
        ax.set_ylabel('场地长度（米）', color='white', fontsize=11)
        ax.tick_params(colors='white', which='both')
        ax.grid(True, color='gray', alpha=0.35, linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_color('white')
        _draw_badminton_court_topdown(ax, x0, x1, y0, y1, net_y=net_y)

    fig, axes = plt.subplots(1, 2, figsize=(14, 9), facecolor='black')
    fig.patch.set_facecolor('black')

    # --- 左：合成热力 ---
    ax0 = axes[0]
    ax0.imshow(
        rgb,
        origin='upper',
        extent=[x0, x1, y1, y0],
        aspect='auto',
        interpolation='bicubic',
    )
    _style_court_ax(ax0)
    ax0.set_title('球员位置热力图', color='white', fontsize=13, pad=8)

    # --- 右：散点（远=红圆，近=蓝三角）---
    ax1 = axes[1]
    ax1.set_facecolor('black')
    ax1.set_xlim(x0, x1)
    ax1.set_ylim(y1, y0)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel('场地宽度（米）', color='white', fontsize=11)
    ax1.set_ylabel('场地长度（米）', color='white', fontsize=11)
    ax1.tick_params(colors='white', which='both')
    ax1.grid(True, color='gray', alpha=0.35, linewidth=0.6)
    for spine in ax1.spines.values():
        spine.set_color('white')
    _draw_badminton_court_topdown(ax1, x0, x1, y0, y1, net_y=net_y)

    if len(xf):
        ax1.scatter(
            xf, yf, c='#ff4444', s=18, alpha=0.65, marker='o',
            edgecolors='none', label=f'上场球员（远端）{rally_label}',
        )
    if len(xn):
        ax1.scatter(
            xn, yn, c='#4488ff', s=20, alpha=0.65, marker='^',
            edgecolors='none', label=f'下场球员（近端）{rally_label}',
        )
    ax1.legend(loc='upper right', fontsize=9, facecolor='0.15', edgecolor='white', labelcolor='white')
    ax1.set_title('球员位置散点图', color='white', fontsize=13, pad=8)

    def _fmps(v: float) -> str:
        return f'{v:.2f}' if v == v else '—'

    def _fmd(v: float) -> str:
        return f'{v:.2f}' if v == v else '—'

    stats_txt = (
        f'{rally_label} 统计\n'
        f'上场球员（远端）: 平均速度 {_fmps(avg_far_segment_mps)} 米/秒  |  '
        f'最大速度 {_fmps(max_far_segment_mps)} 米/秒  |  '
        f'移动距离 {_fmd(total_far_m)} 米\n'
        f'下场球员（近端）: 平均速度 {_fmps(avg_near_segment_mps)} 米/秒  |  '
        f'最大速度 {_fmps(max_near_segment_mps)} 米/秒  |  '
        f'移动距离 {_fmd(total_near_m)} 米'
    )
    fig.text(
        0.5, 0.02, stats_txt, ha='center', va='bottom', fontsize=10, color='white',
        bbox=dict(facecolor='black', alpha=0.72, edgecolor='white', boxstyle='round,pad=0.4'),
    )
    fig.suptitle('球员踝间中点 · 3D 地面投影分布', color='white', fontsize=14, y=0.985)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close(fig)


class HitInferenceRunner:
    """
    击球检测推理运行器
    整合特征提取、模型推理、后处理全流程
    处理视频中的每个回合（rally），输出击球点信息

    主要组件：
    - cfg: 配置参数
    - model: 加载权重的HitNet模型
    - extractor: 特征提取器（YOLO+HSV）
    - cap: OpenCV视频捕获对象
    """

    def __init__(
        self,
        config: HitNetConfig,
        poses_by_frame: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
    ):
        self.cfg = config
        # 加载训练好的模型权重到指定设备（GPU/CPU）
        self.model = HitNetOverfit().to(config.device)
        self.model.load_state_dict(torch.load(config.hitnet_weights, map_location=config.device))
        self.model.eval()  # 推理模式（关闭Dropout、BatchNorm使用运行统计量）
        self.extractor = InferFeatureExtractor(
            config.court_corners, config.POSE_model, poses_by_frame=poses_by_frame
        )
        self.cap = cv2.VideoCapture(config.video_path)  # 打开视频文件
        self.fps = config.fps

    def process_rally(self, start_frame, end_frame, rally_id):
        """
        处理单个回合（从一次发球到球落地）

        完整流程：
        1. 提取该回合所有帧的特征序列（逐帧读取视频，提取78维特征）
        2. 使用滑动窗口（12帧）进行模型推理，步长为stride（2帧）
           - 填充边界（镜像填充），使序列两端也能被检测到（避免边缘漏检）
        3. 后处理得到击球帧（loose_optimize去重）
        4. 提取击球时的球员位置信息（通过脚踝关键点计算地面位置）

        特殊处理：
        - 如果后处理后无结果，取置信度最高的5个候选作为备用（确保不遗漏关键击球）
        - 强制将每个回合的最后一帧标记为is_last_in_rally=1（确保回合结束检测）

        参数：
            start_frame: 回合起始帧号
            end_frame: 回合结束帧号
            rally_id: 回合唯一标识

        返回：包含击球详细信息的字典列表（每字典代表一次击球）
        """
        print(f"处理回合 {rally_id}: 帧 {start_frame}-{end_frame}")
        features, poses_cache = [], {}

        # 读取视频帧并提取特征
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # 跳转到起始帧
        for f in range(start_frame, end_frame + 1):
            ret, frame = self.cap.read()
            if not ret:
                break
            feat, poses = self.extractor.extract(frame, f)  # 提取78维特征+姿态（姿态可走 ByteTrack 缓存）
            features.append(feat)
            poses_cache[f] = poses  # 缓存姿态用于击球帧 hitter/receiver 像素（仍用 HitNet 同款 extract）

        if len(features) < self.cfg.seq_len:
            return []  # 帧数不足，无法组成一个输入序列

        # 填充边界（镜像填充），解决序列两端无法取满12帧窗口的问题
        # pad_len=6，在开头复制6份第一帧特征，确保第0帧也能被检测到
        pad_len = self.cfg.seq_len // 2
        features = np.array(features)
        features = np.concatenate([np.repeat(features[0:1], pad_len, axis=0), features], axis=0)

        predictions = []
        with torch.no_grad():  # 禁用梯度计算，节省内存，加速推理
            # 滑动窗口推理，步长为stride（2帧），平衡精度与速度
            for i in range(0, len(features) - self.cfg.seq_len + 1, self.cfg.stride):
                # 取12帧窗口，转为tensor并移至GPU
                window = torch.FloatTensor(features[i:i + self.cfg.seq_len]).unsqueeze(0).to(self.cfg.device)
                hit_logits, last_logits = self.model(window)  # 前向传播
                hit_probs = torch.softmax(hit_logits, dim=1).cpu().numpy()[0]  # 转为概率分布
                # 映射回原始帧号（减去padding，加上窗口中心偏移）
                actual_frame = start_frame + i - pad_len + (self.cfg.seq_len // 2)
                if start_frame <= actual_frame <= end_frame:
                    predictions.append({
                        'frame': actual_frame,
                        'conf_0': hit_probs[0],  # 无击球概率
                        'conf_1': hit_probs[1],  # near击球概率
                        'conf_2': hit_probs[2],  # far击球概率
                        'is_last_prob': torch.sigmoid(last_logits).cpu().numpy()[0][0]  # 最后一击概率
                    })

        # 后处理：去除重复检测，保证最小时间间隔
        optimized = loose_optimize(predictions, min_gap=int(self.cfg.min_hit_interval * self.fps))

        # 防御性策略：如果后处理后为空，取置信度最高的5个候选作为备用
        if not optimized:
            cands = [{
                'frame': p['frame'],
                'hitter': 'near' if np.argmax([p['conf_0'], p['conf_1'], p['conf_2']]) == 1 else 'far',
                'conf': max([p['conf_0'], p['conf_1'], p['conf_2']]),
                'is_last_prob': p['is_last_prob']
            } for p in predictions if np.argmax([p['conf_0'], p['conf_1'], p['conf_2']]) != 0]
            cands.sort(key=lambda x: x['conf'], reverse=True)
            optimized = cands[:5]

        print(f"  检测到 {len(optimized)} 次击球")

        # 构建输出记录，计算击球时球员的具体位置（脚踝中心）
        rows = []
        for i, hit in enumerate(optimized, 1):
            f = hit['frame']
            hitter = hit['hitter']
            pn, pf = poses_cache[f]  # 获取该帧的两个球员姿态
            # 使用脚踝关键点（15=左脚踝, 16=右脚踝）的中心作为球员地面位置
            near_pos = (pn[15] + pn[16]) / 2
            far_pos = (pf[15] + pf[16]) / 2
            # 根据击球方确定hitter和receiver的坐标（hitter为击球方，receiver为接球方）
            if hitter == 'near':
                h_u, h_v = int(near_pos[0]), int(near_pos[1])
                r_u, r_v = int(far_pos[0]), int(far_pos[1])
            else:
                h_u, h_v = int(far_pos[0]), int(far_pos[1])
                r_u, r_v = int(near_pos[0]), int(near_pos[1])
            # 判断是否为最后一击（概率>0.3或序列末尾）
            is_last = 1 if (hit['is_last_prob'] > 0.3 or i == len(optimized)) else 0
            rows.append({
                'rally_id': rally_id, 'hit_number': i, 'frame': f, 'hitter': hitter,
                'hitter_u': h_u, 'hitter_v': h_v, 'receiver_u': r_u, 'receiver_v': r_v,
                'is_last_in_rally': is_last
            })
        return rows

    def run(self, rally_segments: List[Tuple[int, int]]) -> pd.DataFrame:
        """
        处理所有回合，整合为DataFrame

        参数：
            rally_segments: 每个回合的起止帧列表 [(start1,end1), (start2,end2), ...]

        返回：
            DataFrame包含列：['rally_id', 'hit_number', 'frame', 'hitter',
                           'hitter_u', 'hitter_v', 'receiver_u', 'receiver_v', 'is_last_in_rally']
        """
        all_rows = []
        for rid, (s, e) in enumerate(rally_segments, 1):
            all_rows.extend(self.process_rally(s, e, rid))

        if not all_rows:
            return pd.DataFrame(columns=['rally_id', 'hit_number', 'frame', 'hitter',
                                         'hitter_u', 'hitter_v', 'receiver_u',
                                         'receiver_v', 'is_last_in_rally'])
        df = pd.DataFrame(all_rows)
        # 强制将每个回合的最后一击标记为is_last_in_rally=1（确保回合结束标志正确）
        for rid in df['rally_id'].unique():
            mask = df['rally_id'] == rid
            if mask.any():
                df.loc[df[mask].index[-1], 'is_last_in_rally'] = 1
        return df

    def release(self):
        """释放视频资源（关闭OpenCV视频捕获）"""
        self.cap.release()


# ==================== 2. 数据结构与加载 ====================

@dataclass
class Shot:
    """
    单次击球数据结构（dataclass自动生成__init__等方法）
    存储从一次击球到下一次击球（或回合结束）的完整信息

    核心概念：
    - 一个Shot代表一次完整的击球飞行过程（从球拍击球到落地或被回击）
    - traj_2d: 从CSV读取的图像坐标（像素），通过TrackNet等检测得到
    - traj_3d: 通过相机标定和物理重建得到的世界坐标（米）
    - predicted_landing: 飞行过程中实时预测的最终落地点（动态更新）
    - actual_landing: 根据完整轨迹计算的真实落地点（轨迹Z最低点）

    属性详解：
    - rally_id: 所属回合编号（一场球包含多个回合）
    - shot_number: 回合内的击球序号（第几拍）
    - start_frame/end_frame: 该Shot在视频中的起止帧号
    - hitter: 击球方（'near'近端或'far'远端）
    - hitter_pos_2d/receiver_pos_2d: 击球时双方球员的像素坐标（通过姿态检测脚踝）
    - traj_2d: 羽毛球在图像中的轨迹点 [N,2]（部分点可能因遮挡不可见）
    - frames: 轨迹点对应的帧号 [N]
    - is_visible: 标记该轨迹点是否被TrackNet检测到（1=可见，0=不可见/插值）
    - is_last_in_rally: 是否为回合最后一击（决定后续是否有回击）
    - predicted_landing: 预测落地点XY坐标（米），在预测过程中实时更新
    - landing_confidence: 预测置信度（0-1）
    - flight_time: 飞行时间（秒）
    - actual_landing: 实际落地点（从重建的3D轨迹最低点计算）
    - traj_3d: 重建的3D轨迹 [N,3]（单位：米，世界坐标系）
    - score_result: 得分判断结果（'score'=界内得分，'lose'=失误/界外）
    - score_reason: 得分/失误原因描述
    - prediction_error: 预测落点与实际落点的欧氏距离误差（米）
    - prediction_method: 使用的预测方法（early_phase/middle_ransac/late_phase等）
    - frame_predictions: 逐帧预测记录，用于可视化时回溯历史预测状态
    """
    rally_id: int
    shot_number: int
    start_frame: int
    end_frame: int
    hitter: str  # 'near'或'far'
    hitter_pos_2d: Tuple[float, float]
    receiver_pos_2d: Tuple[float, float]
    traj_2d: np.ndarray  # 2D轨迹 [N,2]（像素）
    frames: np.ndarray  # 对应帧号 [N]
    is_visible: np.ndarray  # 是否可见标记 [N]
    is_last_in_rally: bool
    # 以下为可选的预测/重建结果，初始为None
    predicted_landing: Optional[np.ndarray] = None
    landing_confidence: Optional[float] = None
    flight_time: Optional[float] = None
    actual_landing: Optional[np.ndarray] = None
    traj_3d: Optional[np.ndarray] = None
    score_result: Optional[str] = None  # 'score'或'lose'
    score_reason: Optional[str] = None
    prediction_error: Optional[float] = None
    prediction_method: Optional[str] = None
    frame_predictions: Dict[int, Dict] = field(default_factory=dict)  # 每帧的预测状态缓存


class DataLoader:
    """
    数据加载器
    整合击球检测（HitNet推理）和轨迹数据（CSV）加载
    将原始数据分割为多个Shot对象

    工作流程：
    1. 加载轨迹CSV（TrackNet输出的羽毛球2D坐标）
    2. 获取击球点信息（通过HitNet推理或外部提供）
    3. 根据击球点分割轨迹，生成连续的Shot序列
    4. 标准化列名，处理数据格式
    """

    def __init__(self, traj_csv: str, inference_runner: Optional[HitInferenceRunner] = None,
                 hits_df: Optional[pd.DataFrame] = None):
        print("加载轨迹数据...")
        self.traj_df = pd.read_csv(traj_csv)  # 读取TrackNet轨迹文件

        # 如果没有提供击球数据框，则运行推理模型检测击球点
        if inference_runner is not None:
            cap = cv2.VideoCapture(inference_runner.cfg.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            rally_segments = [(0, total_frames)]  # 简化：将整个视频视为一个回合（实际应使用回合分割算法）
            print("运行击球检测推理...")
            self.hits_df = inference_runner.run(rally_segments)
            inference_runner.release()
            if len(self.hits_df) == 0:
                raise ValueError("推理未检测到任何击球")
        elif hits_df is not None:
            self.hits_df = hits_df
        else:
            raise ValueError("必须提供 inference_runner 或 hits_df")

        # 标准化列名（处理不同来源的命名差异，如Frame/frame, X/x等）
        col_mapping = {'Frame': 'frame', 'frame': 'frame', 'Visibility': 'visibility',
                       'visibility': 'visibility', 'X': 'x', 'Y': 'y', 'x': 'x', 'y': 'y'}
        self.traj_df.rename(columns={k: v for k, v in col_mapping.items()
                                     if k in self.traj_df.columns}, inplace=True)
        if 'visibility' not in self.traj_df.columns:
            self.traj_df['visibility'] = 1  # 默认所有点可见（若无visibility列）

        # 标准化击球数据列名（统一小写，去除空格）
        self.hits_df.columns = [col.lower().strip() for col in self.hits_df.columns]

        # 验证必要列存在（确保后续处理不会KeyError）
        required_cols = ['rally_id', 'hit_number', 'frame', 'hitter',
                         'hitter_u', 'hitter_v', 'receiver_u', 'receiver_v']
        missing = [col for col in required_cols if col not in self.hits_df.columns]
        if missing:
            raise ValueError(f"击球数据缺少必要列: {missing}")

        if 'is_last_in_rally' not in self.hits_df.columns:
            self.hits_df['is_last_in_rally'] = 0

        # 按回合和击球序号排序，确保时间顺序正确
        self.hits_df = self.hits_df.sort_values(['rally_id', 'hit_number']).reset_index(drop=True)
        print(f"轨迹：{len(self.traj_df)}帧，可见{self.traj_df['visibility'].sum()}帧")
        print(f"击球点：{len(self.hits_df)}个，来自 {self.hits_df['rally_id'].nunique()} 个回合")
        self.max_frame = int(self.traj_df['frame'].max())  # 视频最大帧号

    def get_shots(self) -> List[Shot]:
        """
        根据击球点分割轨迹数据，生成Shot对象列表

        分割逻辑详解：
        - 当前击球帧到下一击球帧（同一回合内）：这是一次完整的击球飞行
        - 如果是回合最后一击（is_last_in_rally=1）：
          - 结束帧为下一回合第一帧的前一帧（回合间有间隔）
          - 或是视频结尾（如果是最后一个回合）
        - 如果轨迹点数<2（太短无法重建），跳过该Shot

        数据处理：
        - 提取该时间段内的轨迹点（包括可见和不可见）
        - 记录击球时双方球员位置（2D像素坐标）
        - 标准化击球方标识（统一为near/far小写）

        返回：Shot对象列表，每个代表一次完整的击球飞行过程
        """
        shots = []
        for i in range(len(self.hits_df)):
            current_hit = self.hits_df.iloc[i]

            # 确定当前Shot的结束帧（三种情况）
            if current_hit['is_last_in_rally'] == 1:
                # 如果是回合最后一击，查找下一个回合的第一帧作为结束边界
                next_rally = self.hits_df[self.hits_df['rally_id'] > current_hit['rally_id']]
                end_frame = int(next_rally.iloc[0]['frame']) - 1 if len(next_rally) > 0 else self.max_frame
            else:
                if i < len(self.hits_df) - 1:
                    next_hit = self.hits_df.iloc[i + 1]
                    # 如果下一击在同一回合，则到下一击前；否则到下一回合前
                    end_frame = int(next_hit['frame']) if next_hit['rally_id'] == current_hit['rally_id'] else int(
                        next_hit['frame']) - 1
                else:
                    end_frame = self.max_frame

            start_frame = int(current_hit['frame'])

            # 提取该时间段内的轨迹点（从traj_df中筛选）
            mask = (self.traj_df['frame'] >= start_frame) & (self.traj_df['frame'] <= end_frame)
            segment = self.traj_df[mask].copy()

            if len(segment) < 2:
                continue  # 轨迹太短（不足2点），无法重建物理轨迹，跳过

            visible_mask = (segment['visibility'] == 1)  # 布尔数组，标记哪些帧TrackNet成功检测到球

            # 标准化击球方标识（处理大小写、空格等不一致情况）
            hitter = str(current_hit['hitter']).lower().strip()
            if hitter not in ['near', 'far']:
                hitter = 'far'  # 默认值

            # 创建Shot对象，封装该次击球的所有信息
            shot = Shot(
                rally_id=int(current_hit['rally_id']),
                shot_number=int(current_hit['hit_number']),
                start_frame=start_frame,
                end_frame=end_frame,
                hitter=hitter,
                hitter_pos_2d=(float(current_hit['hitter_u']), float(current_hit['hitter_v'])),
                receiver_pos_2d=(float(current_hit['receiver_u']), float(current_hit['receiver_v'])),
                traj_2d=segment[['x', 'y']].values,  # 提取xy坐标为numpy数组
                frames=segment['frame'].values,  # 帧号数组
                is_visible=visible_mask.values,  # 可见性布尔数组
                is_last_in_rally=(current_hit['is_last_in_rally'] == 1)
            )
            shots.append(shot)
        print(f"成功分割 {len(shots)} 个shots")
        return shots


# ==================== 3. 相机标定（固定X方向版本） ====================
class CameraCalibrator:
    """
    相机标定器
    通过6个对应点（4个场地角点+2个网柱）计算相机投影矩阵P（3×4）
    实现2D图像坐标与3D世界坐标的相互转换

    坐标系定义（右手坐标系）：
    - X轴：场地宽度方向（0到6.1米，单打边线到边线）
    - Y轴：场地长度方向（0到13.4米，底线到底线，网在6.7米处）
    - Z轴：高度方向（0为地面，网柱高度1.55米）
    - 原点(0,0,0)：左远角（画面左上角的场地角）

    固定映射规则（关键约束）：
    - 画面左侧（点1,3）对应X=0.0（场地左边界）
    - 画面右侧（点2,4）对应X=6.1（场地右边界）
    - 确保X坐标与画面水平方向一致，防止左右镜像错误

    算法核心：DLT（直接线性变换）求解投影矩阵P，满足 λ[u,v,1]^T = P[X,Y,Z,1]^T
    """

    def __init__(self, video_path: str, max_display_size=(1280, 720)):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.max_display_size = max_display_size  # 显示窗口最大尺寸，适配屏幕
        self.scale = 1.0  # 显示缩放比例（根据视频分辨率自动计算）
        self.P = None  # 投影矩阵（3×4），计算完成后设置
        self.points_2d_original = []  # 原始分辨率下的2D点（用户点击的像素坐标）
        self.original_size = (1920, 1080)  # 原始视频分辨率
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0  # 视频帧率，默认30fps

        # 场地3D坐标（单位：米），对应6个标定点
        self.court_3d = np.array([
            [0, 0, 0],  # 0: 左远角（左上，X=0,Y=0）
            [6.1, 0, 0],  # 1: 右远角（右上，X=6.1,Y=0）
            [0, 13.4, 0],  # 2: 左近角（左下，X=0,Y=13.4）
            [6.1, 13.4, 0],  # 3: 右近角（右下，X=6.1,Y=13.4）
            [0, 6.7, 1.55],  # 4: 左网柱（X=0,Y=6.7,Z=1.55）
            [6.1, 6.7, 1.55]  # 5: 右网柱（X=6.1,Y=6.7,Z=1.55）
        ])

    def collect_calibration_points(self):
        """
        交互式采集标定点
        用户通过OpenCV GUI按顺序点击6个点，程序自动映射到对应3D坐标

        操作流程：
        1. 读取第一帧视频，计算缩放比例以适应屏幕
        2. 创建交互窗口，设置鼠标回调函数
        3. 用户按顺序点击6个点（角点绿色，网柱蓝色）
        4. 支持'r'键重置，'q'键完成（必须点满6个点）
        5. 将点击的显示坐标转换回原始分辨率坐标
        6. 调用_apply_fixed_calibration计算投影矩阵

        点击顺序：
        1-2: 远端底线左右（左→右）
        3-4: 近端底线左右（左→右）
        5-6: 左右网柱（左→右）
        """
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("无法读取视频")
        self.original_size = (frame.shape[1], frame.shape[0])
        print(f"视频尺寸: {self.original_size[0]}x{self.original_size[1]}, FPS: {self.fps:.2f}")

        # 计算显示缩放比例，确保标定窗口不超出屏幕
        scale_w = self.max_display_size[0] / frame.shape[1]
        scale_h = self.max_display_size[1] / frame.shape[0]
        self.scale = min(scale_w, scale_h, 1.0)

        # 缩放图像以适应显示（保持交互流畅性）
        display_frame = cv2.resize(frame, (
            int(frame.shape[1] * self.scale), int(frame.shape[0] * self.scale))) if self.scale < 1.0 else frame.copy()

        self.current_point = 0  # 当前等待点击的点序号
        self.point_names = ['1:Left far corner (top left)->X=0.0', '2:Right far corner (top right)->X=6.1',
                            '3:Near left corner (bottom left)->X=0.0', '4:Right corner (bottom right)->X=6.1',
                            '5:Left net pillar (center left)->Z=1.55', '6:Right net pillar (center Right)->Z=1.55']
        self.temp_frame = display_frame.copy()
        self.points_2d_display = []  # 显示分辨率下的坐标（用于绘制）

        # 设置OpenCV鼠标回调函数，捕获点击事件
        cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Calibration', display_frame.shape[1], display_frame.shape[0])
        cv2.setMouseCallback('Calibration', self._mouse_callback)

        print("\n请点击（严格按顺序，固定映射）：")
        print("规则：点1,3(画面左侧) -> X=0.0 | 点2,4(画面右侧) -> X=6.1")
        print("按'r'重置，按'q'完成")

        while True:
            display = self.temp_frame.copy()
            # 绘制已点击的点（角点绿色，网柱蓝色）
            for i, pt in enumerate(self.points_2d_display):
                color = (0, 255, 0) if i < 4 else (255, 0, 0)  # 角点绿色，网柱蓝色
                cv2.circle(display, pt, 6, color, -1)
                cv2.putText(display, str(i + 1), (pt[0] + 10, pt[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if self.current_point < 6:
                cv2.putText(display, f"click {self.point_names[self.current_point]}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow('Calibration', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') and len(self.points_2d_display) == 6:
                break
            elif key == ord('r'):
                self.points_2d_display, self.current_point = [], 0
                self.temp_frame = display_frame.copy()
                print("已重置")

        cv2.destroyAllWindows()
        # 将显示坐标转换回原始分辨率坐标（用于精确计算）
        self.points_2d_original = [(int(pt[0] / self.scale), int(pt[1] / self.scale))
                                   for pt in self.points_2d_display]
        self._apply_fixed_calibration()

    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标点击回调函数，记录点击位置并转换到原始坐标"""
        if event == cv2.EVENT_LBUTTONDOWN and self.current_point < 6:
            self.points_2d_display.append((x, y))
            color = (0, 255, 0) if self.current_point < 4 else (255, 0, 0)
            cv2.circle(self.temp_frame, (x, y), 6, color, -1)
            # 转换到原始坐标并打印映射关系（帮助用户确认）
            orig_x, orig_y = int(x / self.scale), int(y / self.scale)
            x_val = 0.0 if self.current_point in [0, 2] else (6.1 if self.current_point in [1, 3] else 'N/A')
            print(f"{self.point_names[self.current_point]}: 画面({orig_x}, {orig_y}) -> 球场X={x_val}")
            self.current_point += 1

    def _compute_p(self, pts_2d, pts_3d):
        """
        通过DLT（直接线性变换）计算投影矩阵P

        数学原理：
        投影方程：λ[u,v,1]^T = P[X,Y,Z,1]^T，其中P是3×4矩阵
        展开得到：
        u = (p11*X + p12*Y + p13*Z + p14) / (p31*X + p32*Y + p33*Z + p34)
        v = (p21*X + p22*Y + p23*Z + p24) / (p31*X + p32*Y + p33*Z + p34)

        整理为线性方程（消去λ）：
        - 方程1: -p11*X - p12*Y - p13*Z - p14 + u*p31*X + u*p32*Y + u*p33*Z + u*p34 = 0
        - 方程2: -p21*X - p22*Y - p23*Z - p24 + v*p31*X + v*p32*Y + v*p33*Z + v*p34 = 0

        每对2D-3D对应点提供2个方程，6个点共12个方程，求解12维向量p（P的展开）
        使用SVD求解齐次线性方程组Ax=0，最小奇异值对应的右奇异向量为解

        参数：
            pts_2d: 6个2D点 [(u,v),...]
            pts_3d: 6个3D点 [(X,Y,Z),...]

        返回：
            P: 3×4投影矩阵
            mean_error: 重投影误差（像素）
            C: 相机中心在世界坐标系中的位置（P的零空间）
        """
        A = []
        for i in range(6):
            X, Y, Z = pts_3d[i]
            u, v = pts_2d[i]
            # 每对点构建两个方程（对应u和v）
            A.append([-X, -Y, -Z, -1, 0, 0, 0, 0, u * X, u * Y, u * Z, u])
            A.append([0, 0, 0, 0, -X, -Y, -Z, -1, v * X, v * Y, v * Z, v])
        A = np.array(A)
        # SVD分解，最小奇异值对应的右奇异向量为解（齐次方程的最小二乘解）
        _, _, Vt = np.linalg.svd(A)
        P = Vt[-1].reshape(3, 4)
        # 归一化，使P[2,3]=1（齐次坐标的尺度不变性）
        P = P / P[2, 3] if abs(P[2, 3]) > 1e-6 else P / np.linalg.norm(P)

        # 计算重投影误差验证精度（将3D点投影回2D，计算与原始2D点的距离）
        errors = []
        for i in range(6):
            X = np.array([*pts_3d[i], 1])
            proj = P @ X
            proj = proj[:2] / proj[2] if abs(proj[2]) > 1e-6 else proj[:2]  # 齐次坐标归一化
            error = np.linalg.norm(proj - np.array(pts_2d[i]))
            errors.append(error)

        # 计算相机中心（P的零空间，满足PC=0）
        # 分解P=[M|p4]，则C=-M^(-1)*p4
        M, p4 = P[:, :3], P[:, 3]
        try:
            C = -np.linalg.pinv(M) @ p4
        except:
            C = np.array([3.05, 6.7, 5.0])  # 默认值：球场中心上方5米
        return P, np.mean(errors), C

    def _apply_fixed_calibration(self):
        """
        应用固定映射规则计算投影矩阵
        强制：画面左侧对应X=0，右侧对应X=6.1

        这一步确保标定结果符合羽毛球场地标准尺寸，防止左右翻转错误
        """
        p = self.points_2d_original
        print("\n=== 固定坐标系映射 ===")
        print(f"  点1(左上) {p[0]} -> X=0.0")
        print(f"  点2(右上) {p[1]} -> X=6.1")
        print(f"  点3(左下) {p[2]} -> X=0.0")
        print(f"  点4(右下) {p[3]} -> X=6.1")

        pts_2d = [p[0], p[1], p[2], p[3], p[4], p[5]]
        self.P, error, C = self._compute_p(pts_2d, self.court_3d)
        self.reproj_error, self.camera_pos = error, C
        print(f"\n标定完成: 重投影误差={error:.2f}像素")
        print(f"相机位置: ({C[0]:.2f}, {C[1]:.2f}, {C[2]:.2f})")
        self._verify_mapping()

    def _verify_mapping(self):
        """验证X方向映射是否正确（防止左右反转）"""
        # 反投影左侧两个点到地面（Z=0），检查X值是否接近0
        left_x = [self.unproject_to_ground(self.points_2d_original[i], 0)[0] for i in [0, 2]]
        right_x = [self.unproject_to_ground(self.points_2d_original[i], 0)[0] for i in [1, 3]]
        print(f"\n验证: 左侧点平均X={np.mean(left_x):.2f}, 右侧点平均X={np.mean(right_x):.2f}")
        if np.mean(left_x) > np.mean(right_x):
            print("⚠️ 警告：X方向可能反转")

    def project(self, X_3d: np.ndarray) -> np.ndarray:
        """
        3D到2D投影（将世界坐标映射到图像坐标）

        数学：uv_homo = P * XYZ1_homo，然后除以第三分量齐次化
        支持批量投影（输入[N,3]，输出[N,2]）或单点投影（输入[3]，输出[2]）

        应用：将重建的3D轨迹投影回图像，与实际检测的2D轨迹对比验证精度
        """
        if X_3d.ndim == 1:
            X_3d = np.array([*X_3d, 1])  # 转为齐次坐标
            proj = self.P @ X_3d
            return proj[:2] / proj[2]  # 齐次归一化
        else:
            ones = np.ones((X_3d.shape[0], 1))
            X_h = np.hstack([X_3d, ones])  # [N,3] -> [N,4]
            proj = (self.P @ X_h.T).T  # [N,4] @ [4,3] -> [N,3]
            return proj[:, :2] / proj[:, 2:3]  # 批量齐次归一化

    def unproject_to_ground(self, uv: Tuple[float, float], z: float = 0) -> np.ndarray:
        """
        2D到3D反投影（射线与平面交点）

        原理：从相机中心穿过图像点(u,v)形成一条3D射线，求该射线与平面Z=z的交点

        步骤：
        1. 计算相机中心C（P的零空间，满足PC=0）
        2. 通过P的伪逆找到射线上的另一点（P^+ * [u,v,1]给出射线上的点）
        3. 求射线与平面Z=z的交点：C + t*direction，其中t=(z-C_z)/direction_z

        应用：将图像中的2D点（如球员位置）转换到世界坐标系的地面高度（Z=0）

        参数：
            uv: 图像坐标(u,v)
            z: 目标高度（默认为0，即地面）

        返回：3D世界坐标[X,Y,Z]
        """
        M, p4 = self.P[:, :3], self.P[:, 3]
        try:
            C = -np.linalg.pinv(M) @ p4  # 相机中心
        except:
            C = np.array([3.05, 6.7, 5.0])  # 默认值

        # P的伪逆给出射线上的点（P * P^+ = I，但这里利用几何意义）
        P_inv = np.linalg.pinv(self.P)
        point_on_ray = P_inv @ np.array([uv[0], uv[1], 1])
        point_on_ray = point_on_ray[:3] / point_on_ray[3]  # 齐次归一化

        # 射线方向（从相机中心指向图像点对应的3D方向）
        direction = point_on_ray - C
        if np.linalg.norm(direction) < 1e-6:
            return np.array([3.05, 6.7, z])  # 异常情况，返回球场中心
        direction = direction / np.linalg.norm(direction)

        # 求与Z=z平面的交点：C[2] + t*direction[2] = z
        if abs(direction[2]) < 1e-6:  # 射线平行于地面，不会相交（理论上不应发生）
            return np.array([point_on_ray[0], point_on_ray[1], z])
        t = (z - C[2]) / direction[2]
        return C + t * direction


# ==================== 4. 智能落点预测（修复版 - 永不返回None） ====================
class SmartPredictor:
    """
    智能落点预测器 - 羽毛球/网球类运动轨迹预测系统

    核心功能：
    1. 根据轨迹不同阶段（早期/中期/晚期）采用差异化预测策略
    2. 物理抛物线模拟（重力加速度 9.81 m/s²）
    3. 防御性编程设计：永不返回 None，具备多重 fallback 机制
    4. 异常值检测与修正（防止数值发散如 45.8m 等离谱预测）

    场地坐标系定义：
    - X轴：场地宽度方向（0 ~ 6.1m），0=左边界，6.1=右边界
    - Y轴：场地长度方向（0 ~ 13.4m），0=近端底线，13.4=远端底线
    - Z轴：高度方向（垂直向上为正），0=地面，>0=空中
    - 网的位置：Y=6.7m（场地中点），高度假设为 0（仅判断 XY 投影是否过网）
    """

    def __init__(self, fps=30.0):
        """
        初始化预测器

        参数:
            fps: 视频帧率，用于计算时间步长 dt=1/fps
                 默认 30fps 对应 dt≈33.3ms，这是体育视频常用帧率
        """
        self.fps = fps
        self.dt = 1.0 / fps  # 单帧时间间隔（秒），物理模拟的时间步长

        # === 场地几何参数（标准羽毛球场地，单位：米）===
        self.COURT_LENGTH = 13.4  # 场地长度（Y轴）：单双打共用，13.4米
        self.COURT_WIDTH = 6.1  # 场地宽度（X轴）：双打外沿，6.1米
        self.NET_Y = 6.7  # 网的位置（Y轴中点）：13.4/2 = 6.7米
        self.GRAVITY = 9.81  # 重力加速度（m/s²），垂直向下
        self.LAND_THRESHOLD = 0.2  # 落地判定高度阈值：低于 0.2m 视为已落地
        # 考虑羽毛球自身半径和地面不平整因素

        # === 羽毛球物理极限参数（防御性设计：防止数值发散）===
        # 基于世界纪录和生物力学极限，超过即视为检测噪声或数值错误
        self.MAX_VELOCITY = 40.0  # 最大初速度（m/s）：杀球世界纪录约 35-40 m/s
        # 若计算速度超过此值，判定为跟踪噪声
        self.MAX_FLIGHT_DIST = 20.0  # 最大合理飞行距离（米）：
        # 高远球从一端底线到另一端底线约 13.4m
        # 考虑对角线和高弧线，上限设为 20m
        self.MAX_FLIGHT_TIME = 3.5  # 最大飞行时间（秒）：
        # 挑球（高弧线慢速球）可达 3 秒
        # 原为 2秒过于严格，3.5秒更合理

    def classify_shot_phase(self, trajectory_3d, current_idx, is_last=False):
        """
        轨迹阶段分类器 - 决定使用何种预测策略

        阶段划分逻辑：
        - too_few: 有效点数<2，无法计算速度，只能基于方向猜测
        - early:   有效点数2-4，刚击球后，数据不足易发散
        - late:    最后一帧且球仍高飞（>1.5m），需限制外推距离
        - middle:  点数≥5，数据充足，RANSAC拟合可靠（黄金预测期）

        参数:
            trajectory_3d: 3D轨迹点列表，每个元素是 [x, y, z] numpy 数组或 None
            current_idx: 当前处理的轨迹索引（相对于击球帧的偏移）
            is_last: 是否为该 shot 的最后一帧（球即将落地或已出界）

        返回:
            字符串标识当前阶段: "too_few", "early", "middle", "late"
        """
        # 统计从起始帧到当前帧的有效轨迹点数量（排除 None 值）
        valid_points = sum(1 for i in range(current_idx + 1)
                           if i < len(trajectory_3d) and trajectory_3d[i] is not None)

        if valid_points < 2:
            return "too_few"  # 无法计算速度，最少需要2点确定方向
        elif valid_points < 5:
            return "early"  # 早期阶段：刚击球后 0-133ms（@30fps），数据不足，数值易发散
        elif is_last and self._is_incomplete_trajectory(trajectory_3d, current_idx):
            return "late"  # 晚期阶段：球还在高空，但跟踪即将结束（可能是出界或遮挡）
        else:
            return "middle"  # 中期阶段：数据充足，物理拟合最可靠

    def _is_incomplete_trajectory(self, trajectory_3d, current_idx):
        """
        判断轨迹是否不完整（用于晚期阶段判定）

        判定条件：
        1. 当前高度 > 1.5米（球还在显著空中，未进入下落阶段）
        2. 正在下降（Z坐标递减）

        这意味着球还有较长飞行距离，但跟踪数据即将中断，
        需要特别保守的预测策略（限制外推距离）。

        参数:
            trajectory_3d: 3D轨迹列表
            current_idx: 当前索引

        返回:
            bool: 是否为不完整的未完成轨迹
        """
        if current_idx < 1:
            return False  # 至少需要2个点判断趋势

        # 获取当前点和前一点
        last_pos, prev_pos = trajectory_3d[current_idx], trajectory_3d[current_idx - 1]
        if last_pos is None or prev_pos is None:
            return False

        # 条件：高度仍高（>1.5m）且正在下降（当前Z < 前一帧Z）
        # 注：羽毛球发球/挑球时可能上升，杀球时急速下降
        return (last_pos[2] > 1.5) and ((last_pos[2] - prev_pos[2]) < 0)

    def _estimate_max_range(self, v0_horizontal, z0):
        """
        估算羽毛球在空气阻力下的最大水平射程（经验公式）

        物理背景：
        羽毛球受空气阻力影响极大（阻力与速度平方成正比），
        其轨迹不是标准抛物线，水平速度衰减极快。
        真空中的抛物线公式会严重高估射程（可能 2-3 倍）。

        简化模型假设：
        - 飞行时间 = 上升时间 + 下降时间，受阻力影响比自由落体慢 20%
        - 水平射程 = 初速度 × 时间 × 衰减因子 0.6（考虑阻力减速）

        参数:
            v0_horizontal: 水平初速度大小（m/s），标量
            z0: 初始高度（m），用于估算下落时间

        返回:
            估算的最大飞行距离（米），上限强制限制为 25米（绝对物理极限）
        """
        if v0_horizontal < 1.0:
            return 5.0  # 速度过小时，假设最小射程5米（球网附近轻柔击球）

        # 估算飞行时间（简化物理：考虑阻力，比自由落体慢）
        # t_est = sqrt(2*h/g)*1.2 + 0.3 中：
        # - sqrt(2*h/g) 是理想自由落体时间
        # - *1.2 考虑空气阻力使下落变慢（轻物体如羽毛球、乒乓球）
        # - +0.3 补偿上升段时间（杀球几乎无上升，挑球有显著上升）
        t_est = np.sqrt(2 * z0 / self.GRAVITY) * 1.2 + 0.3

        # 水平射程估算：初速度 × 时间 × 衰减因子
        # 羽毛球阻力极大，水平速度在飞行中迅速衰减，因子取 0.6（经验值）
        max_range = v0_horizontal * t_est * 0.6

        # 防御性上限：即使 40m/s 杀球也不可能飞超过 25米（实际约 13-15米）
        return min(max_range, 25.0)

    def _directional_fallback(self, trajectory_3d, current_idx, hitter):
        """
        方向性保守预测（关键防御机制）

        使用场景：
        当速度估算不可靠（噪声/发散）或数据不足时，
        不返回基于物理外推的具体坐标（避免 45.8m 这种离谱数值），
        而是基于击球方向的保守估计（固定 3-4米距离）。

        策略：
        - 近端击球（hitter='near'）：预测向远端飞行 4米，限制不超过 11米（底线前）
        - 远端击球（hitter='far'）：预测向近端飞行 4米，限制不低于 2.4米（底线前）
        - X坐标：保持当前位置，稍微向中心 3.05米偏移（羽毛球通常往场内中间打）

        参数:
            trajectory_3d: 3D轨迹列表
            current_idx: 当前索引
            hitter: 击球方，'near'（近端，Y较小）或 'far'（远端，Y较大）

        返回:
            landing_pos: 强制限制在合理范围内的 2D 坐标 (x, y)
            method: "directional_fallback" 或 "unstable_center"（如果连当前位置都找不到）
        """
        # 回溯查找最后一个有效轨迹点（当前帧或之前）
        last_pos = None
        for i in range(current_idx, -1, -1):
            if i < len(trajectory_3d) and trajectory_3d[i] is not None:
                last_pos = trajectory_3d[i]
                break

        # 极端fallback：如果连轨迹点都找不到，返回球场中心（理论上不应发生）
        if last_pos is None:
            return np.array([3.05, 6.7]), "unstable_center"  # 3.05=半场宽，6.7=网位置

        # 基于击球方的方向性保守估计（3-4米飞行距离，远低于高远球实际距离）
        if hitter == 'near':
            # 近端向远端打，预测落在远端半场（Y > 6.7）
            # 限制 11.0米：13.4米总长减去约 2.4米缓冲区（双打后发球线附近）
            target_y = min(last_pos[1] + 4.0, 11.0)
        else:
            # 远端向近端打，预测落在近端半场（Y < 6.7）
            # 限制 2.4米：近端底线前（双打后发球线附近）
            target_y = max(last_pos[1] - 4.0, 2.4)

        # X坐标保守处理：限制在合理范围内（0.5米缓冲区到5.6米，避开边界）
        # 羽毛球极少贴着边线打，通常往中间区域
        target_x = np.clip(last_pos[0], 1.0, 5.1)

        return np.array([target_x, target_y]), "directional_fallback"

    def predict(self, trajectory_3d, current_idx, hitter, is_last=False, debug=False):
        """
        主预测函数 - 强制始终返回有效预测（永不返回 None）

        分层防御架构（共4层保险）：
        1. 阶段选择：根据 classify_shot_phase 选择 early/middle/late 策略
        2. 策略内保险：每个具体 predict_* 方法内部有异常处理
        3. 跨阶段fallback：如果主策略失败，降级到更简单策略
        4. 终极保险：_emergency_prediction 确保数值有效

        异常处理流程：
        middle失败 → fallback到early → 再失败fallback到directional
        所有失败 → emergency_prediction → 数值检查 → 范围钳制

        参数:
            trajectory_3d: 3D轨迹列表 [N×3]，可能包含 None 占位符
            current_idx: 当前处理的帧索引（相对于shot起始）
            hitter: 击球方 'near' 或 'far'，用于方向性判断和过网判定
            is_last: 是否为该shot最后一帧（影响阶段判定和显示逻辑）
            debug: 是否打印调试信息（预留接口）

        返回:
            landing_pos: numpy 数组 [x, y]，保证非 None 且数值有限
            method: 使用的方法名称（字符串，用于可视化颜色编码）
            confidence: 置信度 0.0-1.0（基于数据量和稳定性）
                       - too_few/早期: 0.2-0.4（低置信，黄/橙色）
                       - middle: 0.85（高置信，绿色）
                       - late/fallback: 0.3-0.75（中等置信）
        """
        # 阶段判定：决定使用何种预测策略
        phase = self.classify_shot_phase(trajectory_3d, current_idx, is_last)

        # 统计有效点数用于基础置信度计算
        valid_count = sum(1 for i in range(current_idx + 1)
                          if i < len(trajectory_3d) and trajectory_3d[i] is not None)

        # 基础置信度：数据越多越可信，上限 1.0（10个点以上满置信）
        base_confidence = min(valid_count / 10.0, 1.0)

        result = None
        method = "unknown"
        confidence = base_confidence

        # === 阶段 1: 根据阶段选择主要预测方法 ===

        if phase == "too_few":
            # 点数不足（<2）：无法计算速度，使用方向性保守估计
            result, method = self._directional_fallback(trajectory_3d, current_idx, hitter)
            confidence = 0.2  # 低置信度（数据极少）

        elif phase == "early":
            # 早期阶段（2-4个点）：进行物理外推，但有严格速度检查
            result, method = self.predict_early_phase(trajectory_3d, current_idx, hitter)
            confidence = 0.4  # 中等偏低置信（数据仍少）

            # 早期阶段内部保险：如果速度异常或计算失败，fallback到方向性
            if result is None:
                result, method = self._directional_fallback(trajectory_3d, current_idx, hitter)
                method = "early_fallback_dir"  # 标记为早期fallback
                confidence = 0.25

        elif phase == "late":
            # 晚期阶段（最后一帧且球仍高）：限制外推距离（最大2米）
            result = self.predict_late_phase(trajectory_3d, current_idx, hitter)
            method = "late_phase"
            confidence = 0.75  # 晚期通常数据充足，但外推受限

            if result is None:
                result, method = self._directional_fallback(trajectory_3d, current_idx, hitter)
                method = "late_fallback"
                confidence = 0.3

        else:  # phase == "middle" - 最可靠的阶段
            # 中期阶段（≥5点）：使用 RANSAC 抛物线拟合（最精确）
            result = self.predict_middle_phase(trajectory_3d, current_idx)

            if result is None:
                # 中层保险：middle失败则尝试early（使用更少点的物理模拟）
                result, method = self.predict_early_phase(trajectory_3d, current_idx, hitter)
                method = "middle_fallback_early"
                confidence = 0.5

                if result is None:
                    # 再失败则使用方向性fallback
                    result, method = self._directional_fallback(trajectory_3d, current_idx, hitter)
                    method = "middle_fallback"
                    confidence = 0.25
            else:
                # middle成功：最佳情况
                method = "middle_ransac"
                confidence = 0.85

        # === 阶段 2: 终极保险（理论上不会触发，但确保永不返回None）===
        if result is None:
            result = self._emergency_prediction(trajectory_3d, current_idx, hitter)
            method = "emergency"
            confidence = 0.1  # 极低置信度（红色警告）

        # === 阶段 3: 数值合理性检查（防止 NaN/Inf）===
        if not np.isfinite(result).all():
            # 出现非有限数值（除以零、溢出等），强制fallback
            result = self._directional_fallback(trajectory_3d, current_idx, hitter)[0]
            method = "emergency_invalid"
            confidence = 0.1

        # === 阶段 4: 物理范围检查（关键修复：防止 45.8m 显示）===
        # 获取当前球位置用于距离计算
        last_pos = trajectory_3d[current_idx] if current_idx < len(trajectory_3d) else None
        if last_pos is not None:
            # 计算预测落点与当前位置的距离
            dist_moved = np.linalg.norm(result - last_pos[:2])

            # 如果距离超过物理极限的 1.5 倍（>30米），强制钳制
            if dist_moved > self.MAX_FLIGHT_DIST * 1.5:
                # 计算方向向量并归一化
                direction = (result - last_pos[:2]) / dist_moved
                # 强制拉回最大合理距离（20米）
                result = last_pos[:2] + direction * self.MAX_FLIGHT_DIST
                method += "_clamped"  # 标记被钳制过
                confidence *= 0.5  # 钳制后降低置信度

        # 最终置信度保底 0.05（避免显示 0%）
        return result, method, max(confidence, 0.05)

    def predict_early_phase(self, trajectory_3d, current_idx, hitter):
        """
        早期阶段预测 - 基于物理模拟的速度外推（修复版）

        核心修复（解决 45.8m 发散问题）：
        1. 速度合理性检查：水平速度>40m/s 或总速度>48m/s 视为噪声，
           不返回具体坐标而是 fallback 到方向性预测
        2. 射程估算检查：基于 _estimate_max_range 限制最大飞行距离，
           防止无限外推

        物理模型：
        - 使用最近 3 个点的加权平均速度（近期点权重 0.7，前一点 0.3）
        - 垂直方向速度钳制：羽毛球不可能以 >8m/s 持续上升（杀球几乎瞬间下落）
        - 显式欧拉积分模拟：每步 dt=1/30s，更新 vz -= g*dt，直到 z<=0.2m

        参数:
            trajectory_3d: 3D轨迹
            current_idx: 当前索引
            hitter: 击球方（用于速度异常时的 fallback）

        返回:
            (landing_pos, method) 或 (None, reason) 让外层 fallback
            - landing_pos: 成功时返回 2D 坐标
            - method: "early_phase"（正常）或 "early_limited"（被射程限制截断）
            - None: 数据不足或速度异常，需外层 fallback
        """
        # 收集最近最多 3 个有效点用于速度计算
        points = []
        for i in range(max(0, current_idx - 2), current_idx + 1):
            if i < len(trajectory_3d) and trajectory_3d[i] is not None:
                points.append(trajectory_3d[i])

        if len(points) < 2:
            return None, "early_insufficient"  # 不足2点无法算速度

        dt = self.dt  # 单帧时间间隔

        if len(points) >= 3:
            # 加权平均速度：降低噪声敏感度
            # v1: 前一区间速度，v2: 最近区间速度
            v1 = (points[1] - points[0]) / dt
            v2 = (points[2] - points[1]) / dt
            pos = points[2]  # 当前位置（最新点）
            vel = v2 * 0.7 + v1 * 0.3  # 近期速度权重更高
        else:
            # 只有2点：简单差分
            pos = points[-1]
            vel = (points[-1] - points[-2]) / dt

        # === 关键修复 1：速度合理性检查（防止数值发散）===
        v_horizontal = np.linalg.norm(vel[:2])  # 水平速度大小（XY平面）
        v_total = np.linalg.norm(vel)  # 总速度（3D空间）

        # 物理极限检查：羽毛球杀球世界纪录约 35-40m/s，超过即视为跟踪噪声
        if v_horizontal > self.MAX_VELOCITY or v_total > self.MAX_VELOCITY * 1.2:
            # 速度异常（可能是检测噪声导致跳变），返回方向性预测而非离谱外推
            return self._directional_fallback(trajectory_3d, current_idx, hitter)

        # === 关键修复 2：估算最大射程，防止无限外推 ===
        max_range = self._estimate_max_range(v_horizontal, pos[2])

        # 初始状态
        x, y, z = pos[0], pos[1], pos[2]
        vx, vy, vz = vel[0], vel[1], vel[2]

        # 垂直速度钳制：羽毛球上升速度通常 <8m/s（受空气阻力限制）
        if vz > 8.0:
            vz = 8.0

        # 显式欧拉积分物理模拟（最多模拟 3.5 秒，约 105 帧 @30fps）
        max_steps = int(self.MAX_FLIGHT_TIME / self.dt)
        for step in range(max_steps):
            # 更新垂直速度（重力加速度）
            vz -= self.GRAVITY * self.dt

            # 更新位置（水平匀速，垂直加速）
            x += vx * self.dt
            y += vy * self.dt
            z += vz * self.dt

            # 射程检查：如果已经飞出合理范围，强制提前落地（防止 45.8m）
            current_dist = np.linalg.norm(np.array([x, y]) - pos[:2])
            if current_dist > max_range:
                return np.array([x, y]), "early_limited"  # 标记为受限截断

            # 落地检查：高度低于阈值
            if z <= self.LAND_THRESHOLD:
                return np.array([x, y]), "early_phase"

        # 超时未落地（理论上不应发生，但作为保险）
        return np.array([x, y]), "early_timeout"

    def predict_middle_phase(self, trajectory_3d, current_idx):
        """
        中期阶段预测 - 抛物线拟合（RANSAC/最小二乘）

        算法原理：
        假设羽毛球水平方向匀速运动（忽略水平方向微小空气阻力），
        垂直方向受重力影响呈抛物线（二次曲线）。

        步骤：
        1. 取最近 7 个点的时间序列（相对时间，当前帧为0，之前为负）
        2. Z方向拟合二次曲线：z = a*t² + b*t + c
           求解落地时间：a*t² + b*t + (c-0.2) = 0，取下降根（减号）
        3. X,Y方向拟合一次直线：x = k*t + b，外推到落地时间
        4. 射程检查：如果预测距离 >24米，视为拟合错误

        修复说明：
        - 移除原固定 2秒时间限制，改用 3.5秒（适应挑球）
        - 增加射程检查防止拟合发散

        参数:
            trajectory_3d: 3D轨迹
            current_idx: 当前索引

        返回:
            成功: numpy 数组 [x_land, y_land]
            失败: None（触发上层 fallback 到 early_phase）
        """
        window = 7  # 拟合窗口大小：最近 7 帧（约 233ms @30fps）
        start_idx = max(0, current_idx - window + 1)

        # 收集时间和坐标数据
        times, xs, ys, zs = [], [], [], []
        for i in range(start_idx, current_idx + 1):
            if i < len(trajectory_3d) and trajectory_3d[i] is not None:
                # 相对时间：当前帧为0，之前为负值（秒）
                times.append((i - current_idx) * self.dt)
                x, y, z = trajectory_3d[i]
                xs.append(x)
                ys.append(y)
                zs.append(z)

        # 至少需要 3 个点拟合二次曲线
        if len(times) < 3:
            return None

        times = np.array(times)
        xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)

        try:
            # === Z方向：二次曲线拟合（抛物线）===
            # z(t) = a*t² + b*t + c
            z_coeffs = np.polyfit(times, zs, 2)
            a_z, b_z, c_z = z_coeffs

            # 求解落地时间：a*t² + b*t + (c - LAND_THRESHOLD) = 0
            c_adj = c_z - self.LAND_THRESHOLD

            # 判别式：b² - 4ac
            discriminant = b_z ** 2 - 4 * a_z * c_adj

            if discriminant < 0:
                # 判别式为负：按当前趋势不会落地（理论上不应发生，a<0开口向下必落地）
                # 可能是拟合噪声导致，估算顶点时间并加上 0.8 秒下落时间
                if abs(a_z) > 0.01:
                    t_vertex = -b_z / (2 * a_z)  # 顶点时间（最高点）
                    t_land = t_vertex + 0.8  # 估算落地时间
                else:
                    t_land = 1.5  # 默认值（几乎线性下降）
            else:
                # 取下降阶段的根（减号分支，因为 a<0 开口向下）
                sqrt_disc = np.sqrt(discriminant)
                t_land = (-b_z - sqrt_disc) / (2 * a_z)
                # 注意：a_z 通常为负（抛物线开口向下），所以分子分母符号需仔细分析
                # 实际上由于重力，a_z 应该接近 -0.5*g = -4.9，确实为负

            # === 修复：时间合理性检查 ===
            if t_land < 0:
                # 落地时间为负：说明已经落地或拟合错误
                # 根据当前高度估算剩余下落时间（自由落体公式 h = 0.5*g*t²）
                current_z = zs[-1]
                if current_z > 2.0:
                    t_land = np.sqrt(2 * current_z / self.GRAVITY)
                else:
                    t_land = 0.5  # 已接近地面
            elif t_land > self.MAX_FLIGHT_TIME:  # >3.5秒
                # 时间过长可能是拟合错误（如开口向上的抛物线 a_z>0，这在物理上不可能）
                # 强制截断为最大飞行时间
                t_land = self.MAX_FLIGHT_TIME

            # === X,Y方向：线性拟合（水平匀速假设）===
            if len(times) >= 2:
                x_coeffs = np.polyfit(times, xs, 1)  # 一次拟合
                y_coeffs = np.polyfit(times, ys, 1)
                # 外推到落地时间
                x_land = np.polyval(x_coeffs, t_land)
                y_land = np.polyval(y_coeffs, t_land)
            else:
                # 单点：无法拟合，使用当前位置（理论上不会触发，因为前面检查了<3返回None）
                x_land, y_land = xs[-1], ys[-1]

            # === 修复：射程合理性检查 ===
            last_pos = np.array([xs[-1], ys[-1]])
            landing_pos = np.array([x_land, y_land])
            if np.linalg.norm(landing_pos - last_pos) > self.MAX_FLIGHT_DIST * 1.2:  # >24米
                # 射程异常（如 30米），可能是拟合错误（a_z 异常小导致 t_land 极大）
                return None

            # 最终数值检查：确保不是 NaN 或 Inf
            if not np.isfinite([x_land, y_land]).all():
                return None

            return np.array([x_land, y_land])

        except Exception:
            # 拟合过程异常（如矩阵奇异），返回 None 触发 fallback
            return None

    def predict_late_phase(self, trajectory_3d, current_idx, hitter):
        """
        收尾阶段预测 - 保守外推（限制最大 2米距离）

        使用场景：
        - 球还在高空（>1.5m）但跟踪即将结束（可能是出界或遮挡）
        - 此时外推距离过长容易出错（球路可能大幅弯曲）

        策略：
        1. 使用最近 2-3 点计算瞬时速度
        2. 速度异常检查：>50m/s 视为噪声，钳制到 20m/s
        3. 限制最大外推距离 2米（硬截断）
        4. 限制模拟时间 2秒（硬截断）

        参数:
            trajectory_3d: 3D轨迹
            current_idx: 当前索引
            hitter: 击球方（用于极端失败时的 fallback）

        返回:
            2D 坐标数组（强制返回，即使截断也返回当前外推位置）
        """
        # 收集最近最多 3 个点
        points = []
        for i in range(max(0, current_idx - 2), current_idx + 1):
            if i < len(trajectory_3d) and trajectory_3d[i] is not None:
                points.append(trajectory_3d[i])
                if len(points) >= 2:
                    break  # 只需要最后2个点算速度

        if len(points) < 2:
            # 数据不足，使用方向性 fallback（理论上不应发生，因为 late 阶段应有数据）
            return self._directional_fallback(trajectory_3d, current_idx, hitter)[0]

        last_p = points[-1]  # 当前位置
        prev_p = points[-2]  # 前一位置
        vel = (last_p - prev_p) / self.dt  # 瞬时速度

        # 速度异常检查（>50m/s 视为噪声，钳制到 20m/s）
        v_horizontal = np.linalg.norm(vel[:2])
        if v_horizontal > 50:
            vel[:2] = vel[:2] / v_horizontal * 20  # 归一化后乘以 20m/s

        # 晚期限制：最大外推 2米（防止晚期过度预测）
        max_dist = 2.0
        initial_xy = np.array([last_p[0], last_p[1]])

        # 显式欧拉积分（最多 2秒，约 60 帧）
        x, y, z = last_p[0], last_p[1], last_p[2]
        vx, vy, vz = vel[0], vel[1], vel[2]
        max_steps = int(2.0 / self.dt)

        for _ in range(max_steps):
            vz -= self.GRAVITY * self.dt
            x += vx * self.dt
            y += vy * self.dt
            z += vz * self.dt

            # 检查落地或距离超限
            current_xy = np.array([x, y])
            dist_moved = np.linalg.norm(current_xy - initial_xy)
            if z <= self.LAND_THRESHOLD or dist_moved > max_dist:
                return np.array([x, y])

        # 超时返回最终位置（硬截断）
        return np.array([x, y])

    def predict_score(self, landing_pos, hitter):
        """
        得分判断 - 基于场地规则判定预测落点是得分还是失误

        判定逻辑：
        1. 过网判定：击球方近端（hitter='near'）则落点应在远端（Y<6.7-0.5），
           反之亦然。允许 0.5米误差处理网前球（擦网后落同侧）。
        2. 界内判定：X∈[0,6.1] 且 Y∈[0,13.4]

        返回结构化原因字符串用于 HUD 显示：
        - "NO_NET": 未过网（发球失误或下网）
        - "LEFT_OUT": 左出界（X<0）
        - "RIGHT_OUT": 右出界（X>6.1）
        - "NEAR_OUT": 近端出界（Y<0，近端底线外）
        - "FAR_OUT": 远端出界（Y>13.4，远端底线外）
        - 可组合（如 "LEFT_OUT|FAR_OUT" 左远角出界）

        参数:
            landing_pos: 预测落点 2D 坐标 [x, y]
            hitter: 击球方 'near'（近端，Y较小）或 'far'（远端，Y较大）

        返回:
            result: 'score'（界内且过网）, 'lose'（出界或没过网）, 'undecided'（无效预测）
            reason: 具体原因字符串（用于显示和调试）
        """
        # 有效性检查
        if landing_pos is None or not np.isfinite(landing_pos).all():
            return "undecided", "Invalid prediction"

        x, y = landing_pos[0], landing_pos[1]

        # 物理不可能检查（用于调试，正常不应触发）
        if abs(x) > 50 or abs(y) > 50:
            return "undecided", "Physically impossible"

        # === 1. 过网判定（允许 0.5米误差）===
        net_crossed = True
        if hitter == 'near':
            # 近端击球，球应飞向远端（Y 应显著小于 6.7，即 Y < 6.2）
            # 允许 0.5米误差：处理网前球（如扑球后落同侧但很近）
            if y > self.NET_Y - 0.5:  # Y > 6.2 视为未过网（或刚过网就落）
                net_crossed = False
        else:
            # 远端击球，球应飞向近端（Y 应显著大于 6.7，即 Y > 7.2）
            if y < self.NET_Y + 0.5:  # Y < 7.2 视为未过网
                net_crossed = False

        if not net_crossed:
            return "lose", "NO_NET"  # 没过网，失分

        # === 2. 界内判定 ===
        in_x = 0 <= x <= self.COURT_WIDTH  # X在[0, 6.1]内
        in_y = 0 <= y <= self.COURT_LENGTH  # Y在[0, 13.4]内

        if in_x and in_y:
            return "score", "IN_BOUNDS"  # 界内，得分
        else:
            # 构建出界原因列表（可多原因组合）
            reasons = []
            if not in_x:
                if x < 0:
                    reasons.append("LEFT_OUT")  # 左侧出界
                else:
                    reasons.append("RIGHT_OUT")  # 右侧出界
            if not in_y:
                if y < 0:
                    reasons.append("NEAR_OUT")  # 近端出界（底线外）
                else:
                    reasons.append("FAR_OUT")  # 远端出界（底线外）
            return "lose", "|".join(reasons)  # 出界，失分

    def _fallback_prediction(self, trajectory_3d, current_idx, hitter):

        """
        基础回退预测（简单经验公式）

        基于最后有效点 + 固定距离（3米）的方向性预测。
        比 _directional_fallback 更简单，不限制具体范围，
        仅确保返回合理数值。

        用于 middle 和 early 都失败时的次级 fallback。

        参数:
            trajectory_3d: 3D轨迹
            current_idx: 当前索引
            hitter: 击球方

        返回:
            2D 坐标数组 [x, y]
        """
        # 查找最后一个有效点
        last_pos = None
        for i in range(current_idx, -1, -1):
            if i < len(trajectory_3d) and trajectory_3d[i] is not None:
                last_pos = trajectory_3d[i]
                break

        # 极端情况：返回球场中心（3.05, 6.7）
        if last_pos is None:
            return np.array([3.05, 6.7])

        default_dist = 3.0  # 默认飞行距离 3米（保守）

        if hitter == 'near':
            # 向远端打，Y 增加，限制不超过 12米（底线前 1.4米）
            return np.array([last_pos[0], min(last_pos[1] + default_dist, 12.0)])
        else:
            # 向近端打，Y 减少，限制不低于 1.4米（底线前）
            return np.array([last_pos[0], max(last_pos[1] - default_dist, 1.4)])

    def _emergency_prediction(self, trajectory_3d, current_idx, hitter):
        """
        紧急预测（最后保险机制）

        当所有其他方法都失败时，使用最简化的物理模拟：
        - 使用最近 2 点计算速度
        - 强制设置垂直速度为 -5m/s（确保向下运动）
        - 模拟最多 5秒（超长）直到落地

        此方法几乎不会失败，但精度低，置信度标记为 0.1。

        参数:
            trajectory_3d: 3D轨迹
            current_idx: 当前索引
            hitter: 击球方（用于查找轨迹点）

        返回:
            2D 坐标数组（保证非 None）
        """
        # 收集最近 2 个点（当前和前一点）
        points = []
        for i in range(current_idx, -1, -1):
            if i < len(trajectory_3d) and trajectory_3d[i] is not None:
                points.append(trajectory_3d[i])
                if len(points) >= 2:
                    break

        # 如果只有1个点，使用 fallback
        if len(points) < 2:
            return self._fallback_prediction(trajectory_3d, current_idx, hitter)

        # 计算速度
        vel = (points[0] - points[1]) / self.dt
        pos = points[0]

        # 保险：如果垂直速度向上或不显著向下，强制设置为 -5m/s（向下）
        # 这确保球一定会落地，防止无限循环或向上飞
        if vel[2] >= -1.0:
            vel[2] = -5.0

        x, y, z = pos[0], pos[1], pos[2]
        vx, vy, vz = vel[0], vel[1], vel[2]

        # 超长模拟（最多 5秒，约 150 帧 @30fps）
        max_steps = int(5.0 / self.dt)
        for _ in range(max_steps):
            vz -= self.GRAVITY * self.dt  # 重力更新
            x += vx * self.dt  # 位置更新
            y += vy * self.dt
            z += vz * self.dt

            if z <= self.LAND_THRESHOLD:  # 落地检查
                return np.array([x, y])

        # 5秒后仍未落地（理论上球已在地下很远），返回最终投影位置
        return np.array([x, y])


class TrajectoryReconstructor:
    """
    3D轨迹重建器
    核心功能：通过优化算法（SLSQP/L-BFGS-B）拟合物理模型参数，
    使得重建的3D轨迹投影回2D后与观测轨迹最吻合（重投影误差最小）

    优化变量（7维状态向量）：
    - x0, y0, z0: 初始位置（击球点3D坐标，米）
    - vx, vy, vz: 初始速度（米/秒）
    - Cd: 空气阻力系数（无量纲，羽毛球约为0.5-1.0）

    约束条件（软约束，通过惩罚项实现）：
    - 起点接近击球手位置（硬约束：击球点应在击球手附近，高度约2米）
    - 终点接近接球手位置（球应飞向接球手方向）
    - 必须过网（羽毛球必须越过球网上空，Z>1.55m at Y=6.7m）
    - 物理合理性（不穿透地面Z>=0，速度上限<80m/s，击球手位置合理）

    目标函数（损失函数）组成：
    - 重投影误差（主要项，像素级）：3D轨迹投影回2D应与TrackNet检测结果吻合
    - 起点/终点约束（软约束）：确保轨迹连接两个球员
    - 过网约束（硬惩罚）：未过网给予大惩罚值
    - 物理约束：防止不合理轨迹（如穿地、超音速）
    """

    def __init__(self, camera: 'CameraCalibrator', physics, fps: float = 30.0, scale_y=1.0):
        self.cam = camera  # 相机标定器（提供project/unproject功能）
        self.physics = physics  # PhysicsModel实例（空气阻力模拟）
        self.fps = fps
        self.predictor = SmartPredictor(fps=fps)  # 落点预测器
        self.scale_y = scale_y  # Y轴缩放（如有需要，通常为1.0）

    def reconstruct(self, shot: 'Shot') -> Optional[np.ndarray]:
        """
        重建单次击球的完整3D轨迹

        完整流程：
        1. 准备观测数据：提取可见的2D轨迹点，计算对应时间戳
        2. 球员位置反投影：将击球手/接球手的2D像素坐标反投影到地面（Z=0），作为位置约束
        3. 初始化优化参数：
           - 击球点：击球手位置上方2米（z0=2.0）
           - 初速度：基于2D位移粗略估算（像素差×    经验比例/时间）
           - Cd：0.5（典型羽毛球阻力系数）
        4. 非线性优化：最小化损失函数（重投影误差+约束项）
           - 先尝试SLSQP（支持约束），失败则回退到L-BFGS-B（仅边界）
        5. 生成密集3D轨迹：使用优化后的参数模拟完整飞行过程
        6. 截断落地后轨迹：球落地后保持最后位置（不再移动）

        参数：
            shot: Shot对象，包含2D轨迹、球员位置、时间信息

        返回：3D轨迹数组 [N,3]（单位：米，世界坐标系），失败返回None
        """
        traj_2d = shot.traj_2d
        visible_mask = shot.is_visible
        frames = shot.frames
        visible_traj_2d = traj_2d[visible_mask]
        visible_frames = frames[visible_mask]

        if len(visible_traj_2d) < 3:
            return None  # 可见点太少，无法约束优化问题

        # 计算时间戳（相对于击球时刻，第一帧为t=0）
        t_obs = (visible_frames - shot.start_frame) / self.fps
        if len(t_obs) < 2:
            return None

        try:
            # 将球员2D位置反投影到地面（Z=0）作为位置约束
            # hitter_pos_2d是击球时刻击球手的像素坐标
            x_H_3d = self.cam.unproject_to_ground(shot.hitter_pos_2d, z=0)
            x_R_3d = self.cam.unproject_to_ground(shot.receiver_pos_2d, z=0)
        except Exception as e:
            print(f"反投影失败: {e}")
            return None

        # 限制在场地内（防止反投影超出合理范围，如球员被遮挡导致错误检测）
        x_H_3d = np.array([np.clip(x_H_3d[0], 0.5, 5.6), np.clip(x_H_3d[1], 0.5, 12.9), 2.0])
        x_R_3d = np.array([np.clip(x_R_3d[0], 0.5, 5.6), np.clip(x_R_3d[1], 0.5, 12.9), 2.0])

        is_near = (shot.hitter == 'near')
        # 确保击球手在正确半场（near应在Y>6.7，far应在Y<6.7），否则强制修正
        if is_near and x_H_3d[1] < 6.7:
            x_H_3d[1] = 6.7 + 1.0  # 强制移到远端半场
        elif not is_near and x_H_3d[1] > 6.7:
            x_H_3d[1] = 6.7 - 1.0  # 强制移到近端半场

        # 基于2D位移估算初始速度（用于初始化优化，加速收敛）
        dx_2d = visible_traj_2d[-1] - visible_traj_2d[0]  # 总像素位移
        dt_total = max(t_obs[-1] - t_obs[0], 0.1)  # 总时间（防止除零）
        direction_y = -1 if is_near else 1  # near向Y减小的方向打（向远端），far相反

        if np.linalg.norm(dx_2d) > 10:  # 有足够的2D位移（>10像素）
            # 粗略估算：2D像素差 × 经验比例因子（0.05米/像素） / 时间
            # 注意：这是粗糙估计，实际3D速度通过优化求解
            v0_init = np.array([
                dx_2d[0] * 0.05 / dt_total,
                direction_y * abs(dx_2d[1] * 0.05 / dt_total),
                0.0  # 垂直速度初始设为0，优化器会调整
            ])
        else:
            v0_init = np.array([0, direction_y * 15.0, 5.0])  # 默认值：水平15m/s，垂直5m/s

        # 速度上限检查（防止初始值过大导致优化不稳定）
        v_mag = np.linalg.norm(v0_init)
        if v_mag > 80:
            v0_init = v0_init / v_mag * 80

        x0_init = x_H_3d + np.array([0, 0, 0.5])  # 击球点略高于地面（球拍击球高度约0.5-2米）
        Cd_init = 0.5  # 初始空气阻力系数（典型值）
        x_init = np.concatenate([x0_init, v0_init, [Cd_init]])  # 7维初始状态

        # 优化变量的边界（防止优化器探索不合理空间，提高稳定性）
        bounds = [
            (0.01, 6.09), (0.01, 13.39), (0.5, 5.0),  # x0,y0,z0：必须在场地内，高度0.5-5米
            (-100, 100), (-100, 100), (-50, 50),  # vx,vy,vz：速度上下限（羽毛球速度通常<30m/s，给宽裕边界）
            (0.05, 2.0)  # Cd：阻力系数合理范围（羽毛球约0.5-1.0，给更宽边界0.05-2.0）
        ]

        # 确保初始值在边界内（边界检查，防止初始值越界导致优化器错误）
        for i, (lb, ub) in enumerate(bounds):
            x_init[i] = np.clip(x_init[i], lb + 1e-6, ub - 1e-6)

        # 定义损失函数（闭包，捕获当前观测数据）
        def loss_fn(x):
            return self._loss_soft(x, t_obs, visible_traj_2d, x_H_3d, x_R_3d, is_near)

        # 优化：先尝试SLSQP（支持等式/不等式约束），失败则回退到L-BFGS-B（仅支持边界约束）
        result = None
        try:
            result = minimize(fun=loss_fn, x0=x_init, method='SLSQP', bounds=bounds,
                              options={'maxiter': 2000, 'ftol': 1e-6})
            if not result.success:
                raise ValueError("SLSQP未收敛")
        except:
            try:
                result = minimize(fun=loss_fn, x0=x_init, method='L-BFGS-B', bounds=bounds,
                                  options={'maxiter': 2000, 'ftol': 1e-6})
            except Exception as e2:
                print(f"优化失败: {e2}")
                return None

        if result is None or not result.success:
            return None

        # 提取优化结果（7维状态向量）
        x_opt = result.x
        x0_opt, v0_opt, Cd_opt = x_opt[:3], x_opt[3:6], x_opt[6]

        # 生成完整时间序列的轨迹（从击球到结束，每帧一个3D点）
        duration_frames = shot.end_frame - shot.start_frame
        if duration_frames <= 0:
            duration_frames = int(self.fps * 2)  # 默认2秒飞行时间

        t_full = np.arange(duration_frames) / self.fps  # 完整时间序列
        traj_3d_dense = self.physics.simulate(x0_opt, v0_opt, Cd_opt, t_full)

        # 截断落地后的轨迹（球落地后不再移动，保持最后位置）
        landing_idx = None
        for i, pos in enumerate(traj_3d_dense):
            if pos[2] <= 0.2:  # 找到落地点（Z<0.2米）
                landing_idx = i
                break
        if landing_idx is not None and landing_idx < len(traj_3d_dense) - 1:
            last_valid = traj_3d_dense[landing_idx].copy()
            for i in range(landing_idx + 1, len(traj_3d_dense)):
                traj_3d_dense[i] = last_valid  # 落地后保持静止

        shot.traj_3d = traj_3d_dense  # 保存到Shot对象
        return traj_3d_dense

    def get_landing_point_from_trajectory(self, trajectory_3d: np.ndarray) -> Optional[np.ndarray]:
        """
        从重建的3D轨迹中提取实际落地点（Z最低的点）

        逻辑：遍历轨迹，找到Z坐标最小的点（即最低点，应为落地瞬间）
        如果最小Z<0.3米（确实落地了），返回该点XY坐标
        否则（未落地，如最后一击被拦截），返回最后点的XY坐标
        """
        if trajectory_3d is None or len(trajectory_3d) == 0:
            return None
        min_z_idx = np.argmin(trajectory_3d[:, 2])
        min_z = trajectory_3d[min_z_idx, 2]
        if min_z < 0.3:  # 确实落地了（Z<0.3米）
            return trajectory_3d[min_z_idx, :2]
        return trajectory_3d[-1, :2]  # 否则返回最后点（球被回击或截断）

    def predict_landing_realtime(self, shot: 'Shot', current_idx: int):
        """
        实时预测接口：基于当前已重建的轨迹部分，预测最终落地点

        这是系统的核心功能，用于在球飞行过程中实时预测落点（如第5帧预测第30帧的落点）
        包装SmartPredictor.predict，增加双重保险确保不返回None

        参数：
            shot: Shot对象（包含traj_3d重建轨迹）
            current_idx: 当前帧索引（相对于shot起始帧）

        返回：(landing_pos, result, reason, confidence)
            landing_pos: 预测落点XY坐标（numpy数组，保证非None）
            result: 得分判断结果（'score'/'lose'）
            reason: 判断原因
            confidence: 预测置信度（0-1）
        """
        if shot.traj_3d is None or len(shot.traj_3d) < 2:
            return None, None, None, 0.0

        current_idx = min(current_idx, len(shot.traj_3d) - 1)  # 边界检查

        # 调用SmartPredictor（已修复为永不返回None）
        landing_pos, method, confidence = self.predictor.predict(
            shot.traj_3d, current_idx, shot.hitter, shot.is_last_in_rally, debug=False
        )

        # 双重保险（理论上不需要，但确保系统鲁棒性）
        if landing_pos is None:
            landing_pos = np.array([3.05, 6.7])  # 球场中心（最后保险）

        result, reason = self.predictor.predict_score(landing_pos, shot.hitter)
        return landing_pos, result, reason, confidence

    def _loss_soft(self, x, t_obs, traj_2d_obs, x_H_3d, x_R_3d, is_near):
        """
        优化损失函数（软约束版本）

        各项权重设计哲学：
        1. 重投影误差（sigma=1e-4）：主要目标，但数值本身较大（像素级），所以权重很小
        2. 起点约束（0.5）：击球点应在击球手附近，较重要
        3. 终点约束（0.3）：球应飞向接球手方向，相对次要（羽毛球可能打空或出界）
        4. 过网约束（大惩罚-10）：必须过网，否则给予显著惩罚（但用负值表示奖励？实际代码逻辑需确认）
           实际逻辑：若过网高度>1.55，奖励-10（降低损失）；否则惩罚（高度差×5）
        5. 物理约束：
           - ground_penalty：不穿透地面（Z<0给予大惩罚，1000倍）
           - velocity_penalty：速度上限80m/s（羽毛球通常<30m/s）
           - court_penalty：击球手应在合理半场

        参数：
            x: 7维优化变量 [x0,y0,z0,vx,vy,vz,Cd]
            t_obs: 观测时间序列
            traj_2d_obs: 观测的2D轨迹（像素）
            x_H_3d: 击球手3D位置（约束起点）
            x_R_3d: 接球手3D位置（约束终点）
            is_near: 是否为近端击球（用于半场约束）

        返回：标量损失值（越小越好）
        """
        x0, v0, Cd = x[:3], x[3:6], x[6]
        net_y = 6.7
        net_height = 1.55

        try:
            # 使用当前参数模拟轨迹（空气阻力模型）
            traj_3d = self.physics.simulate(x0, v0, Cd, t_obs)
            # 投影回2D图像（与观测对比）
            traj_2d_proj = self.cam.project(traj_3d)
        except Exception:
            return 1e10  # 模拟失败（如数值爆炸），返回极大损失值

        # 1. 重投影误差（像素级，均方误差）
        reproj_error = np.mean(np.sum((traj_2d_proj - traj_2d_obs) ** 2, axis=1))
        # 2. 起点约束（应与击球手位置接近，高度约2米）
        start_error = np.sum((traj_3d[0] - x_H_3d) ** 2)
        # 3. 终点约束（应与接球手位置接近，仅XY平面）
        end_error = np.sum((traj_3d[-1, :2] - x_R_3d[:2]) ** 2)

        # 4. 过网约束（羽毛球必须越过球网上空）
        y_vals = traj_3d[:, 1]
        min_y, max_y = y_vals.min(), y_vals.max()
        net_penalty = 0
        if (min_y < net_y < max_y):
            # 轨迹跨越网的位置（Y=6.7），检查过网高度
            for j in range(len(y_vals) - 1):
                if (y_vals[j] - net_y) * (y_vals[j + 1] - net_y) < 0:
                    # 线性插值计算过网点的高度
                    ratio = (net_y - y_vals[j]) / (y_vals[j + 1] - y_vals[j])
                    z_cross = traj_3d[j, 2] + ratio * (traj_3d[j + 1, 2] - traj_3d[j, 2])
                    # 如果高度高于网高，给予奖励（-10，降低总损失）；否则惩罚
                    net_penalty = -10.0 if z_cross > net_height else (net_height - z_cross) * 5.0
                    break
        else:
            # 未跨越网（可能是回球未过网或高球），惩罚为到网的距离
            net_penalty = min(abs(min_y - net_y), abs(max_y - net_y)) * 10

        # 5. 物理合理性约束
        ground_penalty = np.sum(np.maximum(0, -traj_3d[:, 2]) ** 2) * 1000  # Z<0（穿透地面）给予大惩罚
        velocity_penalty = max(0, np.linalg.norm(v0) - 80) ** 2 * 0.01  # 速度上限80m/s
        # 击球手位置合理性（near应在Y>6.7远端，far应在Y<6.7近端）
        court_penalty = max(0, net_y - x0[1]) ** 2 * 5.0 if is_near else max(0, x0[1] - net_y) ** 2 * 5.0

        # 加权组合（sigma=1e-4使重投影误差数值范围与其他项平衡）
        sigma = 1e-4
        loss = (reproj_error * sigma + 0.5 * start_error + 0.3 * end_error +
                net_penalty + ground_penalty + velocity_penalty + court_penalty)
        return loss


# ==================== 5. 可视化 ====================

def visualize_results(video_path: str, shots: List[Shot],
                      trajectories_3d: List[np.ndarray],
                      camera: CameraCalibrator,
                      output_path: str = 'result.png'):
    """
    生成多维度可视化图表（6个子图，2×3布局）

    子图说明：
    1. 3D轨迹（三维空间）：显示所有重建轨迹，场地边界，网柱位置，起点（绿）终点（红）
    2. 高度-时间曲线：显示各轨迹的高度变化，标记网高（1.55m虚线）
    3. 侧视图（Z-Y平面，沿X轴看）：显示高度vs场地长度，标记过网点（星号）
    4. 顶视图（X-Y平面，从上往下看）：显示场地平面投影，预测落点（X标记），飞行方向箭头
    5. 正视图（Z-X平面，沿Y轴看）：显示高度vs场地宽度
    6. 2D图像投影对比：显示原始视频帧，叠加2D检测点和3D投影轨迹对比

    参数：
        video_path: 原始视频路径（用于提取第一帧作为背景）
        shots: Shot对象列表
        trajectories_3d: 对应的3D轨迹列表（与shots一一对应，可能包含None）
        camera: 相机标定器（用于3D到2D投影验证）
        output_path: 保存的图像文件路径
    """
    fig = plt.figure(figsize=(20, 12))
    rally_ids = sorted(list(set([s.rally_id for s in shots])))
    # 为不同回合分配不同颜色（tab10色图）
    base_colors = plt.cm.tab10(np.linspace(0, 1, max(len(rally_ids), 4)))
    colors = []
    for shot in shots:
        rally_idx = rally_ids.index(shot.rally_id)
        color = base_colors[rally_idx % len(base_colors)]
        brightness = 0.7 + 0.3 * ((shot.shot_number % 3) / 2)  # 同一回合内不同击球用亮度区分
        color = np.array(color) * brightness
        color = np.clip(color, 0, 1)
        colors.append(color)

    # 子图1: 3D轨迹
    ax3d = fig.add_subplot(231, projection='3d')
    # 绘制场地边界（底线和边线，Z=0平面）
    court_x = [0, 6.1, 6.1, 0, 0]
    court_y = [0, 0, 13.4, 13.4, 0]
    court_z = [0, 0, 0, 0, 0]
    ax3d.plot(court_x, court_y, court_z, 'k-', linewidth=2)
    ax3d.plot([0, 6.1], [6.7, 6.7], [0, 0], 'g-', linewidth=3)  # 网（绿色粗线）
    ax3d.plot([0, 0], [6.7, 6.7], [0, 1.55], 'g-', linewidth=3)  # 左网柱
    ax3d.plot([6.1, 6.1], [6.7, 6.7], [0, 1.55], 'g-', linewidth=3)  # 右网柱
    ax3d.text(3, 3, 0, 'Far Court', fontsize=10, ha='center')
    ax3d.text(3, 10, 0, 'Near Court', fontsize=10, ha='center')

    # 绘制所有成功重建的轨迹
    success_shots = []
    for i, (shot, traj) in enumerate(zip(shots, trajectories_3d)):
        if traj is not None:
            success_shots.append((shot, traj, colors[i]))
            ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=colors[i], linewidth=2, alpha=0.7)
            ax3d.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color='green', s=40)  # 起点（击球点）
            ax3d.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color='red', s=40, marker='x')  # 终点（落地）

    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.set_title('3D Trajectory')
    ax3d.set_xlim(-1, 7)
    ax3d.set_ylim(-1, 15)
    ax3d.set_zlim(0, 5)

    # 子图2: 高度-时间曲线
    ax_height = fig.add_subplot(232)
    for shot, traj, color in success_shots:
        t = np.arange(len(traj)) / 30.0  # 时间（秒）
        ax_height.plot(t, traj[:, 2], color=color, linewidth=2,
                       label=f"R{shot.rally_id}S{shot.shot_number}")
    ax_height.axhline(y=1.55, color='g', linestyle='--', alpha=0.5, label='Net')
    ax_height.set_xlabel('Time (s)')
    ax_height.set_ylabel('Height (m)')
    ax_height.set_title('Height vs Time')
    ax_height.legend(fontsize=8, loc='upper right')
    ax_height.grid(True)
    ax_height.set_ylim(0, 5)

    # 子图3: 侧视图（Z-Y平面，沿X轴看）
    ax_zy = fig.add_subplot(233)
    for shot, traj, color in success_shots:
        ax_zy.plot(traj[:, 1], traj[:, 2], color=color, linewidth=2)
        ax_zy.scatter(traj[0, 1], traj[0, 2], color='green', s=30)
        ax_zy.scatter(traj[-1, 1], traj[-1, 2], color='red', s=30, marker='x')
        # 标记过网点（如果轨迹跨越网位置Y=6.7）
        y_vals = traj[:, 1]
        if y_vals.min() < 6.7 < y_vals.max():
            for j in range(len(y_vals) - 1):
                if (y_vals[j] - 6.7) * (y_vals[j + 1] - 6.7) < 0:
                    z_cross = traj[j, 2] + (traj[j + 1, 2] - traj[j, 2]) * \
                              (6.7 - y_vals[j]) / (y_vals[j + 1] - y_vals[j])
                    ax_zy.scatter([6.7], [z_cross], color=color, s=100, marker='*', zorder=5)
                    break

    ax_zy.axhline(y=1.55, color='g', linestyle='--', alpha=0.5, label='Net height')
    ax_zy.axvline(x=6.7, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Net pos')
    ax_zy.set_xlabel('Y (m) - Court Length')
    ax_zy.set_ylabel('Z (m) - Height')
    ax_zy.set_title('Side View (Z-Y)')
    ax_zy.legend()
    ax_zy.grid(True)
    ax_zy.set_xlim(-1, 14.4)
    ax_zy.set_ylim(0, 5)

    # 子图4: 顶视图（X-Y平面，从上往下看）
    ax_xy = fig.add_subplot(234)
    for shot, traj, color in success_shots:
        ax_xy.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2)
        ax_xy.scatter(traj[0, 0], traj[0, 1], color='green', s=30, marker='o')
        ax_xy.scatter(traj[-1, 0], traj[-1, 1], color='red', s=30, marker='x')

        # 绘制预测落点（如果存在）
        if shot.predicted_landing is not None:
            if shot.prediction_method == "early_phase":
                pred_color = 'orange'
            elif shot.prediction_method == "late_phase":
                pred_color = 'purple'
            else:
                pred_color = 'blue'
            ax_xy.scatter(shot.predicted_landing[0], shot.predicted_landing[1],
                          color=pred_color, s=100, marker='x', linewidths=3, alpha=0.8)

        # 绘制飞行方向箭头（中点处）
        mid_idx = len(traj) // 2
        if mid_idx > 0:
            dx = traj[mid_idx + 1, 0] - traj[mid_idx, 0] if mid_idx + 1 < len(traj) else 0
            dy = traj[mid_idx + 1, 1] - traj[mid_idx, 1] if mid_idx + 1 < len(traj) else 0
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                ax_xy.annotate('', xy=(traj[mid_idx, 0] + dx * 3, traj[mid_idx, 1] + dy * 3),
                               xytext=(traj[mid_idx, 0], traj[mid_idx, 1]),
                               arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    # 绘制场地边界和区域填充
    ax_xy.plot([0, 6.1, 6.1, 0, 0], [0, 0, 13.4, 13.4, 0], 'k-', linewidth=2)
    ax_xy.plot([0, 6.1], [6.7, 6.7], 'g--', linewidth=2, alpha=0.7, label='Net')
    ax_xy.fill_between([0, 6.1], [0, 0], [6.7, 6.7], alpha=0.1, color='blue', label='Far Court')
    ax_xy.fill_between([0, 6.1], [6.7, 6.7], [13.4, 13.4], alpha=0.1, color='yellow', label='Near Court')

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Start'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='red', markersize=8, label='End'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='blue', markersize=8, label='Predicted'),
    ]
    ax_xy.legend(handles=legend_elements, loc='upper left', fontsize=8)
    ax_xy.set_xlabel('X (m)')
    ax_xy.set_ylabel('Y (m)')
    ax_xy.set_title('Top View (X-Y)')
    ax_xy.grid(True)
    ax_xy.set_xlim(-1, 7)
    ax_xy.set_ylim(-1, 14)
    ax_xy.set_aspect('equal')

    # 子图5: 正视图（Z-X平面，沿Y轴看）
    ax_zx = fig.add_subplot(235)
    for shot, traj, color in success_shots:
        ax_zx.plot(traj[:, 0], traj[:, 2], color=color, linewidth=2)
        ax_zx.scatter(traj[0, 0], traj[0, 2], color='green', s=30)
        ax_zx.scatter(traj[-1, 0], traj[-1, 2], color='red', s=30, marker='x')
    ax_zx.axhline(y=1.55, color='g', linestyle='--', alpha=0.5)
    ax_zx.set_xlabel('X (m)')
    ax_zx.set_ylabel('Z (m)')
    ax_zx.set_title('Side View (Z-X)')
    ax_zx.grid(True)
    ax_zx.set_xlim(-0.5, 6.6)
    ax_zx.set_ylim(0, 5)

    # 子图6: 2D图像投影对比（验证重投影精度）
    ax_2d = fig.add_subplot(236)
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax_2d.imshow(frame)
        for shot, traj, color in success_shots:
            visible = shot.is_visible
            if np.any(visible):
                ax_2d.scatter(shot.traj_2d[visible, 0], shot.traj_2d[visible, 1],
                              s=15, alpha=0.5, color=color)  # 原始2D检测点
            # 投影3D轨迹回2D进行对比（应为虚线，颜色相同）
            traj_2d_proj = camera.project(traj)
            ax_2d.plot(traj_2d_proj[:, 0], traj_2d_proj[:, 1], '--', linewidth=2, color=color)
        try:
            # 投影场地3D模型到2D验证标定准确性（应为白色矩形）
            court_pts = camera.project(camera.court_3d[:4])
            ax_2d.plot(court_pts[[0, 1, 3, 2, 0], 0], court_pts[[0, 1, 3, 2, 0], 1], 'w-', linewidth=2)
        except:
            pass

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n结果已保存至 {output_path}")


# ==================== 6. 视频渲染器（保持不变） ====================

class VideoRenderer:
    """
    视频渲染器 - 前15帧预测隐藏版（Warmup模式）

    核心功能：
    1. 读取原始视频，渲染带有3D轨迹和预测落点的可视化视频
    2. 前15帧（击球后0-167ms）进行数据积累但隐藏预测结果，显示"ANALYZING..."
    3. 第16帧起根据轨迹分析显示预测落点、置信度和比赛结果（得分/出界）
    4. 支持多目标（shots）同时渲染，每个shot使用不同颜色区分
    """

    def __init__(self, video_path: str, output_path: str, camera):
        """
        初始化视频渲染器

        参数:
            video_path: 输入视频文件路径
            output_path: 输出视频文件路径（MP4格式）
            camera: Camera对象，负责3D坐标到2D像素坐标的投影（project）和反投影
        """
        # 视频源和输出配置
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)  # 打开输入视频
        self.camera = camera  # 相机投影模型（用于3D->2D坐标转换）
        self.output_path = output_path

        # 视频基本参数提取
        self.fps = camera.fps  # 帧率（从camera对象获取，而非视频本身）
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频宽度（像素）
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频高度（像素）
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数

        # 视频写入器配置（使用MP4V编码器）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        # 颜色调色板 - 用于区分不同shots（回合中的不同击球）
        # 依次为：青色、品红、黄色、绿色、橙蓝、紫蓝、橙黄、粉红
        self.colors = [
            (255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 255, 0),
            (0, 128, 255), (128, 0, 255), (255, 128, 0), (255, 0, 128),
        ]

        # 预测稳定性阈值：预测落点与当前球位置的距离超过25米视为不合理预测
        self.MAX_REASONABLE_DIST = 25.0

        # 预热帧数：前15帧（约0.5秒@30fps）隐藏预测，仅显示"ANALYZING..."
        # 这段时间用于积累足够的轨迹点进行稳定预测
        self.WARMUP_FRAMES = 15

    def render(
        self,
        shots,
        trajectories_3d,
        track_draw_by_frame: Optional[Dict[int, Any]] = None,
        pose_render_conf: float = 0.25,
        pose_draw_skeleton: bool = True,
    ):
        """
        主渲染函数 - 处理整个视频帧序列

        参数:
            shots: 列表，每个元素是一个Shot对象（包含击球信息、2D轨迹、预测结果等）
            trajectories_3d: 与shots对应的3D轨迹点列表（世界坐标系，单位：米）
                            每个traj是N×3的数组，对应shot的N个轨迹点
            track_draw_by_frame: 可选，extract_poses_by_frame_from_track 返回的 draw 字典；
                每帧为 (xy, conf, ids)，使用 render_player_pose_2d.draw_track_results_on_frame 绘制骨架/踝线/中点（与 HitNet 无关）

        处理流程：
        1. 构建帧索引映射（frame_map）：建立frame_idx到该帧应显示的所有数据的映射
        2. 逐帧读取视频，绘制轨迹历史（渐隐效果）、当前球位置、HUD信息
        3. 绘制预测落点（前15帧跳过）
        4. 写入输出视频
        """
        print(f"\n开始渲染视频...")
        print(f"视频尺寸: {self.width}x{self.height}, FPS: {self.fps}")
        print(f"前{self.WARMUP_FRAMES}帧预测计算但不显示")
        print(f"共 {len(shots)} 个 shots，{self.total_frames} 帧")

        # ============================================================
        # 阶段1：构建帧到数据的映射表（Frame Map）
        # 目标：快速查找任意帧需要渲染的所有shots数据
        # ============================================================
        frame_map = {}

        # 遍历每个shot及其对应的3D轨迹
        for shot_idx, (shot, traj) in enumerate(zip(shots, trajectories_3d)):
            if traj is None:
                continue  # 跳过无3D轨迹的shot

            # 为该shot分配颜色（循环使用调色板）
            color = self.colors[shot_idx % len(self.colors)]

            # 遍历该shot的所有帧（与击球帧的相对帧）
            for i in range(len(shot.frames)):
                frame_idx = int(shot.frames[i])  # 全局帧索引

                # 可见性检查：如果该点在2D跟踪中被标记为不可见（被遮挡/丢失），则跳过
                if hasattr(shot, 'is_visible') and i < len(shot.is_visible):
                    if not shot.is_visible[i]:
                        continue

                # 边界检查：确保2D坐标在图像范围内
                if i < len(shot.traj_2d):
                    x_2d = int(shot.traj_2d[i][0])
                    y_2d = int(shot.traj_2d[i][1])
                    if not (0 <= x_2d < self.width and 0 <= y_2d < self.height):
                        continue

                    # 计算在3D轨迹数组中的索引（相对于击球起始帧的偏移）
                    traj_idx = frame_idx - shot.start_frame

                    # 获取当前帧对应的3D位置（如果轨迹数据有效）
                    pos_3d = None
                    if 0 <= traj_idx < len(traj):
                        pos_3d = traj[traj_idx]

                    # 将数据加入frame_map：一个帧可能包含多个shots的数据（多球同时存在）
                    if frame_idx not in frame_map:
                        frame_map[frame_idx] = []

                    frame_map[frame_idx].append({
                        'shot': shot,  # shot对象引用（包含预测结果等）
                        'shot_idx': shot_idx,  # shot的全局索引（用于颜色区分）
                        'pos_2d': (x_2d, y_2d),  # 当前帧的2D像素坐标
                        'pos_3d': pos_3d,  # 当前帧的3D世界坐标（米）
                        'color': color,  # 该shot的专属颜色
                        'traj': traj,  # 完整的3D轨迹（用于后续计算）
                        'traj_idx': traj_idx,  # 在轨迹数组中的索引
                        'frame_offset': traj_idx  # 相对于击球帧的时间偏移（用于warmup判断）
                    })

        print(f"共 {len(frame_map)} 帧包含轨迹数据")

        # ============================================================
        # 阶段2：主渲染循环
        # ============================================================
        frame_idx = 0  # 当前处理的帧计数器
        history = []  # 历史帧数据缓存（用于绘制渐隐轨迹）

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break  # 视频读取结束

            # 获取当前帧需要渲染的所有shot数据（从预计算的frame_map中查找）
            current_data = frame_map.get(frame_idx, [])

            # 维护历史缓存（最近5帧），用于绘制渐隐尾迹效果
            if current_data:
                history.append((frame_idx, current_data))
            if len(history) > 5:
                history.pop(0)  # 移除最旧的历史帧

            # 绘制1：渐隐轨迹（历史5帧的轨迹点，透明度随时间递减）
            # 注意：此绘制不受WARMUP限制，始终显示实际轨迹
            self._draw_gradient_trajectory(frame, history, frame_idx)

            if track_draw_by_frame:
                pack = track_draw_by_frame.get(frame_idx)
                if pack is not None:
                    xy, kconf, tids = pack
                    draw_track_results_on_frame(
                        frame, xy, kconf, tids,
                        conf_thr=pose_render_conf,
                        draw_skeleton=pose_draw_skeleton,
                    )

            # 绘制2：当前球位置标记和HUD信息面板（左上角状态显示）
            # 内部处理warmup逻辑（前15帧显示"ANALYZING..."）
            if current_data:
                self._draw_current_frame_info(frame, current_data, frame_idx)

            # 绘制3：预测落点标记（虚线+圆圈）和实际落点标记
            for item in current_data:
                shot = item['shot']
                current_pos_3d = item['pos_3d']
                frame_offset = item['frame_offset']  # 相对于击球的帧偏移

                # === Warmup逻辑：前15帧不显示预测落点标记 ===
                # 但计算照常进行，只是不渲染，避免早期不稳定预测干扰观看
                if frame_offset < self.WARMUP_FRAMES:
                    continue  # 跳过预测落点绘制

                # 第16帧起：正常显示预测落点
                # 获取该帧对应的预测数据（支持逐帧更新预测结果）
                if hasattr(shot, 'frame_predictions') and frame_idx in shot.frame_predictions:
                    # 使用逐帧更新的动态预测结果（实时修正）
                    pred_data = shot.frame_predictions[frame_idx]
                    landing_pos = pred_data['pos']  # 预测落点2D坐标
                    method = pred_data['method']  # 预测方法（如middle_ransac）
                    confidence = pred_data.get('confidence', 0.5)  # 置信度0-1
                else:
                    # 回退：使用shot级别的静态预测结果（整个轨迹使用同一预测）
                    landing_pos = shot.predicted_landing
                    method = shot.prediction_method
                    confidence = 1.0

                # 绘制预测落点（虚线连接当前球位置到落点，落点处绘制标记）
                if landing_pos is not None:
                    self._draw_landing_prediction(frame, landing_pos, current_pos_3d,
                                                  item['color'], method, confidence)

                # 特殊：最后一帧（击球结束帧）绘制实际落点（白色方框）
                # 不受warmup限制，因为实际落点是已发生的事实，不是预测
                if shot.actual_landing is not None and frame_idx == shot.end_frame:
                    self._draw_actual_landing(frame, shot.actual_landing)

            # 写入输出视频
            self.writer.write(frame)
            frame_idx += 1

            # 进度打印（每100帧）
            if frame_idx % 100 == 0:
                print(f"  已处理 {frame_idx}/{self.total_frames} 帧...")

        # 释放资源
        self.writer.release()
        self.cap.release()
        print(f"✓ 视频已保存至: {self.output_path}")

    def _draw_current_frame_info(self, frame, current_data, current_frame_idx):
        """
        绘制HUD（抬头显示）信息面板 - 位于画面左上角

        显示内容：
        - 当前球3D坐标（实时）
        - 回合ID和击球手（如 R1S2 (NEAR)）
        - 预测状态（ANALYZING.../SCORE/LOSE/PREDICTING...）
        - 预测置信度
        - 预测落点坐标（warmup结束后）

        参数:
            frame: 当前处理的OpenCV图像帧（numpy数组）
            current_data: 当前帧的所有shot数据列表
            current_frame_idx: 当前帧的全局索引
        """
        y_offset = 30  # 第一个面板的起始Y坐标
        line_height = 95  # 每个shot面板的高度（防止重叠）

        # 遍历当前帧的所有shots（支持多球同时显示）
        for item in current_data:
            pos_2d = item['pos_2d']  # 当前2D像素坐标
            color = item['color']  # 该shot的专属颜色
            pos_3d = item['pos_3d']  # 当前3D世界坐标
            shot = item['shot']  # shot对象
            frame_offset = item['frame_offset']  # 相对于击球的时间偏移（关键参数）

            # ===== 图形标记：绘制当前球位置 =====
            # 三层同心圆：白色外圈（描边）+ 彩色填充 + 白色中心点
            cv2.circle(frame, pos_2d, 15, (255, 255, 255), 3, cv2.LINE_AA)  # 外圈白边
            cv2.circle(frame, pos_2d, 10, color, -1, cv2.LINE_AA)  # 中间填充色
            cv2.circle(frame, pos_2d, 4, (255, 255, 255), -1, cv2.LINE_AA)  # 中心白点
            # 十字准星线
            cv2.line(frame, (pos_2d[0] - 20, pos_2d[1]), (pos_2d[0] + 20, pos_2d[1]), color, 2, cv2.LINE_AA)
            cv2.line(frame, (pos_2d[0], pos_2d[1] - 20), (pos_2d[0], pos_2d[1] + 20), color, 2, cv2.LINE_AA)

            # ===== 获取预测状态数据 =====
            if hasattr(shot, 'frame_predictions') and current_frame_idx in shot.frame_predictions:
                # 动态预测数据（逐帧更新）
                pred_data = shot.frame_predictions[current_frame_idx]
                current_method = pred_data['method']
                current_conf = pred_data.get('confidence', 0.5)
                current_result = pred_data.get('result', 'predicting')  # 'score', 'lose', 'predicting'
                current_reason = pred_data.get('reason', '')
                landing_pos = pred_data.get('pos')
            else:
                # 静态预测数据（整个轨迹使用单一结果）
                current_method = shot.prediction_method or "final"
                current_conf = 1.0
                current_result = shot.score_result or "unknown"
                current_reason = shot.score_reason or ''
                landing_pos = shot.predicted_landing

            # ===== 准备基础文本 =====
            # 3D坐标显示（世界坐标系，单位米）
            if pos_3d is not None:
                text_3d = f"3D: ({pos_3d[0]:.2f}, {pos_3d[1]:.2f}, {pos_3d[2]:.2f})m"
            else:
                text_3d = "3D: N/A"

            # 回合信息：R{回合ID}S{击球序号} ({击球手})
            text_shot = f"R{shot.rally_id}S{shot.shot_number} ({shot.hitter.upper()})"

            # ===== 关键逻辑：根据阶段确定显示状态 =====

            # 情况1：Warmup阶段（前15帧）- 强制显示"ANALYZING..."
            if frame_offset < self.WARMUP_FRAMES:
                status_color = (255, 255, 0)  # 黄色（警告/处理中）
                status_text = "ANALYZING..."
                reason_display = f"Warmup {frame_offset + 1}/{self.WARMUP_FRAMES}"  # 如 "Warmup 2/15"
                display_coords = False  # 不显示预测坐标
                display_landing = False  # 不显示落点标记（已在render中跳过）
                method_indicator = "[INIT]"  # 初始化阶段标记

            else:
                # 情况2：正常阶段（第16帧起）- 根据实际状态判断

                # 判断是否为早期预测（数据不足时使用的方法）
                is_early = current_method in [
                    "early_phase", "early_limited", "early_fallback_dir",
                    "too_few", "directional_fallback", "unstable_center",
                    "early_fallback"
                ]

                # 判断预测是否合理（防止异常值）
                # 计算预测落点与当前球位置的距离，超过25米视为不合理
                is_unreasonable = False
                if landing_pos is not None and pos_3d is not None:
                    dist = np.linalg.norm(landing_pos - pos_3d[:2])  # 只比较XY平面距离
                    if dist > self.MAX_REASONABLE_DIST or not np.isfinite(landing_pos).all():
                        is_unreasonable = True

                # 子情况2a：不合理预测且非早期阶段 → 显示错误状态
                if is_unreasonable and not is_early:
                    status_color = (0, 0, 255)  # 红色（错误）
                    status_text = "UNSTABLE PRED"  # 不稳定预测
                    reason_display = "Check trajectory data"
                    display_coords = False
                    method_indicator = "[ERR]"

                # 子情况2b：早期阶段（数据不足）→ 显示方向性预测
                elif is_early:
                    # 根据击球手位置判断对方场地
                    if shot.hitter == 'near':
                        status_text = "PRED: FAR COURT >>"  # 预测远场落点
                    else:
                        status_text = "PRED: NEAR COURT >>"  # 预测近场落点
                    status_color = (255, 255, 0)  # 黄色（警告）
                    reason_display = "Calculating..."
                    display_coords = False  # 早期不显示具体坐标，只显示方向
                    method_indicator = "[EARLY]"

                # 子情况2c：预测完成且球将落在界内 → SCORE（得分）
                elif current_result == "score":
                    status_color = (0, 255, 0)  # 绿色（良好）
                    status_text = "SCORE"
                    reason_display = "IN BOUNDS"  # 界内
                    display_coords = True  # 显示预测坐标

                # 子情况2d：预测完成但球将出界 → LOSE（失分）
                elif current_result == "lose":
                    status_color = (0, 0, 255)  # 红色（错误/警告）
                    status_text = "LOSE"

                    # 解析出界原因并转换为可读文本
                    if current_reason == "NO_NET":
                        reason_display = "NO NET CROSSING"  # 未过网
                    elif "OUT" in current_reason:
                        # 解析组合原因（如 "LEFT_OUT|FAR_OUT"）
                        parts = current_reason.split("|")
                        display_parts = []
                        for part in parts:
                            if part == "LEFT_OUT":
                                display_parts.append("LEFT OUT")  # 左侧出界
                            elif part == "RIGHT_OUT":
                                display_parts.append("RIGHT OUT")  # 右侧出界
                            elif part == "NEAR_OUT":
                                display_parts.append("NEAR OUT")  # 近端出界
                            elif part == "FAR_OUT":
                                display_parts.append("FAR OUT")  # 远端出界
                            else:
                                display_parts.append(part)
                        reason_display = " | ".join(display_parts)
                    else:
                        reason_display = current_reason if current_reason else "OUT OF BOUNDS"
                    display_coords = True  # 即使出界也显示预测坐标

                # 子情况2e：正在预测中（数据积累中）
                else:
                    status_color = (255, 255, 0)  # 黄色
                    status_text = "PREDICTING..."
                    reason_display = "Analyzing..."
                    display_coords = True

                # 方法缩写映射（在HUD中显示短标签节省空间）
                method_map = {
                    "early_phase": "[EARLY]",
                    "early_limited": "[EARLY-L]",
                    "directional_fallback": "[DIR]",  # 方向回退
                    "middle_ransac": "[MID]",  # 中段RANSAC拟合
                    "late_phase": "[LATE]",  # 后段稳定预测
                    "unstable_center": "[ERR]",  # 不稳定中心
                    "middle_fallback_early": "[MID-F]",  # 中段早期回退
                    "late_fallback": "[LATE-F]",  # 后段回退
                    "emergency": "[EMRG]"  # 紧急回退
                }
                method_indicator = method_map.get(current_method, f"[{current_method.upper()[:6]}]")

            # ===== 计算背景框尺寸（自适应文本宽度） =====
            conf_str = f"{current_conf * 100:.0f}%"  # 置信度百分比
            main_text = f"{method_indicator} {status_text} ({conf_str})"  # 主状态行

            # 测量各文本段的像素宽度，取最大值作为框宽
            (text_w, _), _ = cv2.getTextSize(text_3d, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            (main_w, _), _ = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            (reason_w, _), _ = cv2.getTextSize(reason_display, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            max_w = max(text_w, main_w, reason_w, 200)  # 最小宽度200像素

            # 根据显示内容动态调整框高度（出界时多一行显示原因）
            box_height = 100 if (frame_offset >= self.WARMUP_FRAMES and current_result == "lose") else 75

            # 绘制半透明黑色背景（提升文字可读性）
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, y_offset - 25), (20 + max_w, y_offset + box_height), (0, 0, 0), -1)
            # alpha混合：overlay 60% + 原图 40%
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # ===== 绘制文本（四行信息） =====
            # 第1行：3D坐标（始终显示当前球位置，该坐标是已知的而非预测）
            cv2.putText(frame, text_3d, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            # 第2行：回合信息（白色）
            cv2.putText(frame, text_shot, (10, y_offset + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)

            # 第3行：状态和方法（带颜色：绿/红/黄）
            cv2.putText(frame, main_text, (10, y_offset + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2,
                        cv2.LINE_AA)

            # 第4行：坐标或原因（根据阶段不同）
            if frame_offset < self.WARMUP_FRAMES:
                # Warmup阶段：显示进度（如 "Warmup 2/15"）
                cv2.putText(frame, reason_display, (10, y_offset + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2,
                            cv2.LINE_AA)

            elif display_coords and landing_pos is not None:
                # 正常阶段且显示坐标：显示预测落点XY坐标
                coord_text = f"PRED: ({landing_pos[0]:.1f}, {landing_pos[1]:.1f})"
                cv2.putText(frame, coord_text, (10, y_offset + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2,
                            cv2.LINE_AA)

                # 如果是出界(lose)，在第5行显示具体原因
                if current_result == "lose":
                    cv2.putText(frame, reason_display, (10, y_offset + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                status_color, 2, cv2.LINE_AA)
            else:
                # 其他情况（如早期阶段）：显示原因/状态文本
                cv2.putText(frame, reason_display, (10, y_offset + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2,
                            cv2.LINE_AA)

            # 更新Y偏移：为下一个shot的面板预留空间
            y_offset += line_height

    def _draw_gradient_trajectory(self, frame, history, current_frame_idx):
        """
        绘制渐变色轨迹（历史帧渐隐效果）

        算法：
        - 使用最近最多5帧的历史数据
        - 每帧的透明度随时间递减（当前帧100%，前1帧80%，前2帧60%...）
        - 连接同一shot的历史点形成轨迹线

        参数:
            frame: 当前处理的图像帧
            history: 历史帧数据列表，格式为 [(frame_idx, [data_items, ...]), ...]
            current_frame_idx: 当前帧索引（用于计算时间差）
        """
        if len(history) < 2:
            return  # 至少需要2帧才能绘制轨迹

        # 遍历历史帧（从旧到新）
        for i, (hist_frame_idx, hist_data) in enumerate(history):
            # 计算该历史帧距离现在的"年龄"（帧数差）
            age = current_frame_idx - hist_frame_idx
            if age > 4:
                continue  # 只显示最近5帧（age 0-4）

            # 计算透明度：age越大，alpha越小（越透明）
            # age=0（当前帧）: alpha=1.0
            # age=4（4帧前）: alpha=0.2
            alpha = 1.0 - (age * 0.2)

            # 绘制该历史帧中所有shots的轨迹点
            for item in hist_data:
                pos = item['pos_2d']  # 历史2D坐标
                base_color = item['color']  # 基础颜色

                # 根据alpha混合颜色：越旧越接近白色（255,255,255）
                color = tuple(int(c * alpha + 255 * (1 - alpha) * 0.3) for c in base_color)

                # 轨迹点半径随时间递减（越旧越小）
                radius = max(2, int(6 - age))
                cv2.circle(frame, pos, radius, color, -1, cv2.LINE_AA)

                # 连接相邻历史帧的同一shot点形成轨迹线
                if i < len(history) - 1:
                    # 查找同一shot在下一历史帧中的位置
                    next_frame_data = history[i + 1][1]
                    for next_item in next_frame_data:
                        if next_item['shot_idx'] == item['shot_idx']:
                            next_pos = next_item['pos_2d']
                            # 线条粗细也随时间递减
                            thickness = max(1, int(3 - age * 0.5))
                            cv2.line(frame, pos, next_pos, color, thickness, cv2.LINE_AA)
                            break

    def _draw_landing_prediction(self, frame, landing_pos_2d, current_pos_3d, color, method, confidence):
        """
        绘制预测落点标记（仅在第16帧后被render调用）

        视觉元素：
        1. 从当前球位置到预测落点的虚线（表示运动趋势）
        2. 落点处的圆圈标记（大小随置信度变化）
        3. 落点坐标文本和置信度百分比

        参数:
            frame: 当前图像帧
            landing_pos_2d: 预测落点的2D世界坐标（XY平面，单位米）
            current_pos_3d: 当前球的3D世界坐标（用于绘制连线起点）
            color: 该shot的基础颜色
            method: 预测方法（影响颜色编码）
            confidence: 置信度0.0-1.0（影响标记大小和透明度）
        """
        if landing_pos_2d is None:
            return

        # 将预测落点从3D世界坐标投影到2D像素坐标
        # landing_pos_2d是XY平面坐标（Z=0，地面），添加Z=0组成3D坐标
        landing_3d = np.array([*landing_pos_2d, 0.0])
        uv = self.camera.project(landing_3d)
        u, v = int(uv[0]), int(uv[1])

        # 边界检查：确保落点在图像范围内
        if not (0 <= u < self.width and 0 <= v < self.height):
            return

        # 根据预测阶段选择颜色编码（便于观察系统状态）
        if method in ["early_phase", "early_limited"]:
            base_color = (0, 165, 255)  # 橙色（早期，不稳定）
        elif method in ["late_phase", "late_fallback"]:
            base_color = (255, 0, 255)  # 紫色（后期，稳定）
        elif "fallback" in method or "emergency" in method:
            base_color = (128, 128, 128)  # 灰色（回退/紧急模式）
        else:
            base_color = (0, 255, 0)  # 绿色（正常/中段预测）

        # 根据置信度调整透明度和标记大小
        # 高置信度 = 不透明 + 大标记
        alpha = 0.3 + 0.7 * confidence
        size = int(8 + 12 * confidence)  # 大小范围：8-20像素

        # 颜色alpha混合（更亮的显示颜色）
        display_color = tuple(int(c * alpha + 255 * (1 - alpha) * 0.5) for c in base_color)

        # 绘制从当前位置到落点的虚线（如果当前位置有效）
        if current_pos_3d is not None:
            current_uv = self.camera.project(current_pos_3d)
            cu, cv = int(current_uv[0]), int(current_uv[1])
            self._draw_dashed_line(frame, (cu, cv), (u, v), display_color, 2, 10)

        # 绘制落点标记（十字瞄准线风格）
        # 外圈（空心）
        cv2.circle(frame, (u, v), size, display_color, 2, cv2.LINE_AA)
        # 内圈（实心填充）
        cv2.circle(frame, (u, v), size // 2, display_color, -1, cv2.LINE_AA)
        # 十字线（水平和垂直）
        cv2.line(frame, (u - size, v), (u + size, v), display_color, 2, cv2.LINE_AA)
        cv2.line(frame, (u, v - size), (u, v + size), display_color, 2, cv2.LINE_AA)

        # 绘制落点坐标文本背景（黑色半透明矩形）
        text = f"PRED: ({landing_pos_2d[0]:.1f}, {landing_pos_2d[1]:.1f})"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        overlay = frame.copy()
        cv2.rectangle(overlay, (u - text_w // 2 - 5, v - 30), (u + text_w // 2 + 5, v - 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # 绘制落点坐标文本（位于落点上方）
        cv2.putText(frame, text, (u - text_w // 2, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, display_color, 2,
                    cv2.LINE_AA)

        # 绘制置信度百分比（位于落点下方）
        conf_text = f"{confidence * 100:.0f}%"
        cv2.putText(frame, conf_text, (u - 10, v + size + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, display_color, 1,
                    cv2.LINE_AA)

    def _draw_actual_landing(self, frame, actual_landing_2d):
        """
        绘制实际落点标记（白色方框）- 不受15帧warmup限制

        在实际落点发生时（最后一帧）标记真实落点位置，用于与预测对比

        参数:
            frame: 当前图像帧
            actual_landing_2d: 实际落点的2D世界坐标（XY平面）
        """
        # 将实际落点投影到像素坐标
        actual_3d = np.array([*actual_landing_2d, 0.0])
        uv = self.camera.project(actual_3d)
        u, v = int(uv[0]), int(uv[1])

        # 边界检查
        if not (0 <= u < self.width and 0 <= v < self.height):
            return

        # 使用白色方框标记实际落点（与预测的圆圈区分）
        size = 12
        # 外框（空心正方形）
        cv2.rectangle(frame, (u - size, v - size), (u + size, v + size), (255, 255, 255), 2, cv2.LINE_AA)
        # 中心填充（实心小正方形）
        cv2.rectangle(frame, (u - 4, v - 4), (u + 4, v + 4), (255, 255, 255), -1, cv2.LINE_AA)

        # 文本标签（位于方框右侧）
        text = f"ACTUAL: ({actual_landing_2d[0]:.1f}, {actual_landing_2d[1]:.1f})"
        cv2.putText(frame, text, (u + 15, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_dashed_line(self, frame, pt1, pt2, color, thickness=2, dash_length=10):
        """
        绘制虚线（辅助函数）

        算法：计算两点间距离，等分为多段，交替绘制实线和空白

        参数:
            frame: 图像帧
            pt1, pt2: 起点和终点坐标 (x, y)
            color: BGR颜色元组
            thickness: 线宽
            dash_length: 每段虚线长度（像素）
        """
        x1, y1 = pt1
        x2, y2 = pt2

        # 计算两点间距离
        dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if dist < 1:
            return  # 距离太短不绘制

        # 计算需要多少段虚线
        dashes = int(dist / dash_length)

        # 每隔一段绘制（i=0,2,4...为实线，i=1,3,5...为空白）
        for i in range(0, dashes, 2):
            # 计算当前段的起点和终点在总线段上的比例位置
            start_ratio = i / dashes
            end_ratio = min((i + 1) / dashes, 1.0)  # 确保不超过1.0

            # 线性插值计算实际像素坐标
            x_start = int(x1 + (x2 - x1) * start_ratio)
            y_start = int(y1 + (y2 - y1) * start_ratio)
            x_end = int(x1 + (x2 - x1) * end_ratio)
            y_end = int(y1 + (y2 - y1) * end_ratio)

            # 绘制该段实线
            cv2.line(frame, (x_start, y_start), (x_end, y_end), color, thickness, cv2.LINE_AA)

    def release(self):
        """
        释放所有OpenCV资源（视频捕获器和写入器）

        应在渲染完成后调用，避免资源泄漏
        """
        if self.writer:
            self.writer.release()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()  # 关闭所有OpenCV窗口（如果有的话）


# ==================== 7. 主程序 ====================
class PhysicsModel:
    """
    物理模型：带空气阻力的抛体运动

    动力学方程（状态空间表示）：
    状态变量 s = [x, y, z, vx, vy, vz] （位置+速度）

    ds/dt = [
        vx,
        vy,
        vz,
        -Cd * v * vx,    (X方向阻力)
        -Cd * v * vy,    (Y方向阻力)
        -g - Cd * v * vz (Z方向阻力+重力)
    ]
    其中 v = sqrt(vx^2 + vy^2 + vz^2) 为速率
    Cd为空气阻力系数（包含空气密度、截面积、质量等因素的综合参数）

    数值积分：使用scipy.integrate.odeint（LSODA算法，自适应步长）
    """

    def __init__(self, g: float = 9.81):
        self.g = np.array([0, 0, -g])  # 重力加速度向量（向下）

    def dynamics(self, state, t, Cd):
        """
        计算状态导数（用于odeint）

        参数：
            state: [x, y, z, vx, vy, vz] 当前状态
            t: 时间（odeint要求，此处未显式使用，因autonomous系统）
            Cd: 空气阻力系数

        返回：ds/dt 列表
        """
        x, y, z, vx, vy, vz = state
        v = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)  # 速率标量

        if v < 0.01:
            drag = 0  # 速度接近0时无阻力（防止除零）
        else:
            drag = -Cd * v  # 阻力大小与速度平方成正比（Cd已包含密度、截面积、质量等参数）
            # 实际物理：F_drag = -0.5 * rho * C_d * A * |v| * v_vec
            # 此处 Cd = 0.5 * rho * C_d * A / mass，为简化综合参数

        ax = drag * vx / v if v > 0.01 else 0  # 阻力方向与速度相反
        ay = drag * vy / v if v > 0.01 else 0
        az = self.g[2] + (drag * vz / v if v > 0.01 else 0)  # 重力+阻力

        return [vx, vy, vz, ax, ay, az]

    def simulate(self, x0, v0, Cd, t_span):
        """
        模拟轨迹

        参数：
            x0: 初始位置 [x,y,z]
            v0: 初始速度 [vx,vy,vz]
            Cd: 阻力系数
            t_span: 时间序列（numpy数组，如np.arange(0, 2, 1/30)）

        返回：轨迹数组 [N, 3]（位置），每行对应t_span的一个时间点
        """
        state0 = [*x0, *v0]  # 拼接为6维状态向量
        # odeint数值积分：rtol/atol控制精度，args传递额外参数Cd
        states = odeint(self.dynamics, state0, t_span, args=(Cd,), rtol=1e-6, atol=1e-6)
        return states[:, :3]  # 返回位置部分（前3列）


def main():
    """
    主程序流程（7个步骤）：
    1. 生成2D轨迹：调用TrackNet预测羽毛球在图像中的2D坐标（如果尚未存在）
    2. 初始化击球检测模型（HitNet）：加载权重，准备推理
    3. 加载数据并运行击球检测：分割视频为Shots，检测每次击球
    4. 相机标定：交互式6点标定，计算投影矩阵P
    5. 3D轨迹重建：对每个Shot进行物理优化，重建3D轨迹
    6. 实时落点预测：逐帧预测落点，计算预测误差和得分判断
    7. 保存结果：CSV数据文件+可视化视频+分析图表

    配置参数（需根据实际文件路径修改）：
    - VIDEO_PATH: 输入视频路径
    - TRAJ_CSV: TrackNet输出的2D轨迹CSV
    - HITNET_WEIGHTS: 击球检测模型权重
    - OUTPUT_VIDEO: 输出渲染视频路径
    - OUTPUT_CSV: 输出数据CSV路径
    - COURT_CORNERS: 场地4个角点像素坐标（用于特征归一化）
    """
    # 配置路径（根据实际情况修改）
    VIDEO_PATH = 'data/video/clip3.mp4'
    TRAJ_CSV = 'data/tmp/clip3.csv'
    HITNET_WEIGHTS = 'data/weights/hitnet_output/hitnet_overfit_best.pth' #只需要优化这个模型
    OUTPUT_VIDEO = 'result/clip3/output_3d.mp4'
    OUTPUT_CSV = 'result/clip3/output_reconstructed_3d.csv'
    if os.path.dirname(OUTPUT_CSV):
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    if os.path.dirname(OUTPUT_VIDEO):
        os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    OUTPUT_PLAYERS_CSV = os.path.join(os.path.dirname(OUTPUT_CSV) or '.', 'output_players_3d.csv')
    OUTPUT_PLAYERS_HEATMAP = os.path.join(os.path.dirname(OUTPUT_CSV) or '.', 'output_players_heatmap.png')
    OUTPUT_PLAYERS_STATS = os.path.join(os.path.dirname(OUTPUT_CSV) or '.', 'output_players_stats.csv')
    POSE_model = './yolov8x-pose.pt' #人体关键点检测

    print("=== 羽毛球3D轨迹重建系统（固定坐标系+稳定预测版） ===")
    print("标定规则：画面左侧(点1,3) -> X=0.0 | 画面右侧(点2,4) -> X=6.1")

    print("\n[步骤1] 相机标定（固定映射）...")
    try:
        calib = CameraCalibrator(VIDEO_PATH)
        calib.collect_calibration_points()  # 交互式标定
        print(f"\n  标定结果:")
        print(f"    重投影误差: {calib.reproj_error:.2f}像素")
        print(f"    相机位置: ({calib.camera_pos[0]:.2f}, {calib.camera_pos[1]:.2f}, {calib.camera_pos[2]:.2f})")
    except Exception as e:
        print(f"  错误：标定失败 - {e}")
        return
    # 新增：自动获取标定的羽毛球场坐标
    COURT_CORNERS = [[x[0], x[1]] for i, x in enumerate(calib.points_2d_original) if i < 4]
    print("\n[步骤2] 生成2D轨迹...")
    # 步骤0：生产轨迹文件（如果tmp.csv不存在，调用TrackNet生成）
    predict_trajectory(
        video_file=VIDEO_PATH,
        tracknet_file=r'data/weights/ckpts/TrackNet_best.pt',
        inpaintnet_file=None,  # 可选，用于修复遮挡
        batch_size=4, #16改为4，减少GPU资源消耗
        eval_mode='nonoverlap',
        large_video=True,
        output_video=False,  # 不输出中间视频
        save_dir='data/tmp',
        out_csv_file=TRAJ_CSV,
        return_dict=False
    )
    print("  使用已有轨迹文件")

    poses_by_frame: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    track_draw_by_frame: Dict[int, Any] = {}
    print("\n[步骤2b] YOLO pose + ByteTrack（HitNet 输入与球员路程共用，整段视频只跑这一次 pose）...")
    try:
        poses_by_frame, track_draw_by_frame = extract_poses_by_frame_from_track(
            VIDEO_PATH,
            POSE_model,
            conf_thr=0.25,
            device=None,
            tracker="bytetrack.yaml",
        )
    except Exception as e:
        print(f"  警告：track 姿态失败，HitNet 将退回逐帧 YOLO；球员路程与叠加可能缺失 - {e}")
        poses_by_frame = {}
        track_draw_by_frame = {}

    print("\n[步骤3] 初始化击球检测模型...")
    try:
        hit_config = HitNetConfig(court_corners=COURT_CORNERS, video_path=VIDEO_PATH, weights_path=HITNET_WEIGHTS, POSE_model = POSE_model, fps=30)
        inference_runner = HitInferenceRunner(hit_config)
        print("  击球检测模型加载成功（HitNet 使用逐帧 YOLO 推理，与 main2 一致）")
    except Exception as e:
        print(f"  错误：模型加载失败 - {e}")
        return

    print("\n[步骤3b] 加载数据并运行击球检测...")
    try:
        loader = DataLoader(TRAJ_CSV, inference_runner=inference_runner)
        shots = loader.get_shots()
        if len(shots) == 0:
            print("  错误：未分割到任何有效的shots")
            return
        print(f"  成功加载 {len(shots)} 个shots")
    except Exception as e:
        print(f"  错误：数据加载失败 - {e}")
        import traceback
        traceback.print_exc()
        return
    finally:
        if hasattr(inference_runner, 'cap'):
            inference_runner.release()

    # print("\n[步骤3] 相机标定（固定映射）...")
    # try:
    #     calib = CameraCalibrator(VIDEO_PATH)
    #     calib.collect_calibration_points()  # 交互式标定
    #     print(f"\n  标定结果:")
    #     print(f"    重投影误差: {calib.reproj_error:.2f}像素")
    #     print(f"    相机位置: ({calib.camera_pos[0]:.2f}, {calib.camera_pos[1]:.2f}, {calib.camera_pos[2]:.2f})")
    # except Exception as e:
    #     print(f"  错误：标定失败 - {e}")
    #     return

    print(f"\n[步骤4] 3D轨迹重建...")
    physics = PhysicsModel()
    reconstructor = TrajectoryReconstructor(calib, physics, fps=calib.fps, scale_y=1.0)

    trajectories_3d = []
    success_count = 0

    # 处理每个Shot（逐个进行3D重建和实时预测）
    for i, shot in enumerate(shots):
        print(f"\n  处理 [{i + 1}/{len(shots)}] R{shot.rally_id}S{shot.shot_number} ({shot.hitter})")
        traj = reconstructor.reconstruct(shot)  # 核心：物理优化重建
        trajectories_3d.append(traj)

        if traj is not None:
            success_count += 1
            print(f"    ✓ 重建成功，轨迹长度: {len(traj)}帧")

            # 计算逐帧实时预测（模拟实时应用场景）
            shot.frame_predictions = {}
            pred_count = 0

            for frame_idx in range(shot.start_frame, shot.end_frame + 1):
                current_idx = frame_idx - shot.start_frame
                if current_idx >= len(traj):
                    break

                # 实时预测：基于当前已观察到的轨迹部分，预测最终落点
                landing_pos, score_result, score_reason, confidence = reconstructor.predict_landing_realtime(
                    shot, current_idx
                )

                # 确定预测方法（用于可视化颜色编码）
                valid_points = sum(1 for j in range(current_idx + 1) if j < len(traj))
                if valid_points < 5:
                    method = "early_phase"
                elif shot.is_last_in_rally and current_idx > len(traj) * 0.8:
                    method = "late_phase"
                else:
                    method = "middle_ransac"

                # 保存该帧的预测状态（用于后续渲染）
                shot.frame_predictions[frame_idx] = {
                    'pos': landing_pos,
                    'method': method,
                    'result': score_result,
                    'reason': score_reason,
                    'confidence': confidence
                }
                pred_count += 1

            # 记录最终预测结果（最后一帧的预测作为该shot的最终预测）
            final_frame = shot.end_frame
            if final_frame in shot.frame_predictions:
                final_pred = shot.frame_predictions[final_frame]
                shot.predicted_landing = final_pred['pos']
                shot.prediction_method = final_pred['method']
                shot.score_result = final_pred['result']
                shot.score_reason = final_pred['reason']

                # 计算实际落点和误差（用于评估预测准确度）
                actual_landing = reconstructor.get_landing_point_from_trajectory(traj)
                shot.actual_landing = actual_landing

                if actual_landing is not None:
                    error = np.linalg.norm(shot.predicted_landing - actual_landing)
                    shot.prediction_error = error

                    in_court = (0 <= shot.predicted_landing[0] <= 6.1 and
                                0 <= shot.predicted_landing[1] <= 13.4)

                    print(f"    ✓ 生成 {pred_count} 帧预测 (100%)")
                    print(f"    ✓ 最终方法: {shot.prediction_method}, 置信度: {confidence:.2f}")
                    print(f"    ✓ 预测: ({shot.predicted_landing[0]:.2f}, {shot.predicted_landing[1]:.2f}) "
                          f"{'[界内]' if in_court else '[界外]'} | 结果: {shot.score_result.upper()}")
                    print(f"    ✓ 实际: ({actual_landing[0]:.2f}, {actual_landing[1]:.2f}), 误差: {error:.2f}m")
        else:
            print(f"    ✗ 重建失败")

    print(f"\n  重建统计: {success_count}/{len(shots)} 成功 ({success_count / len(shots) * 100:.1f}%)")

    # 保存CSV结果（包含所有重建轨迹和预测数据）
    print(f"\n[步骤5] 保存结果到 {OUTPUT_CSV}...")
    output_data = []
    for i, (shot, traj) in enumerate(zip(shots, trajectories_3d)):
        if traj is None:
            continue
        vel = instantaneous_velocity_from_trajectory(traj, calib.fps)
        for j in range(len(traj)):
            frame_num = shot.start_frame + j
            pred_x, pred_y = None, None
            actual_x, actual_y = None, None
            error_val = None
            method_val = None
            conf_val = None

            # 该帧的实时预测数据（如果存在）
            if frame_num in shot.frame_predictions:
                pred = shot.frame_predictions[frame_num]
                pred_x, pred_y = pred['pos'][0], pred['pos'][1]
                method_val = pred['method']
                conf_val = pred['confidence']

            # 只在最后一帧记录实际落点和误差（避免数据冗余）
            if j == len(traj) - 1:
                if shot.actual_landing is not None:
                    actual_x = shot.actual_landing[0]
                    actual_y = shot.actual_landing[1]
                error_val = shot.prediction_error

            pos = traj[j]
            vvec = vel[j]
            spd = float(np.linalg.norm(vvec))
            output_data.append({
                'rally_id': shot.rally_id,
                'shot_number': shot.shot_number,
                'frame': frame_num,
                'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2]),
                'vx': float(vvec[0]), 'vy': float(vvec[1]), 'vz': float(vvec[2]),
                'speed_mps': spd,
                'hitter': shot.hitter,
                'is_last_in_rally': 1 if shot.is_last_in_rally else 0,
                'predicted_landing_x': pred_x, 'predicted_landing_y': pred_y,
                'prediction_method': method_val, 'prediction_confidence': conf_val,
                'actual_landing_x': actual_x, 'actual_landing_y': actual_y,
                'prediction_error': error_val,
                'score_result': shot.score_result if j == len(traj) - 1 else None,
                'score_reason': shot.score_reason if j == len(traj) - 1 else None
            })

    if output_data:
        df = pd.DataFrame(output_data)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"  ✓ 已保存 {len(output_data)} 行数据")

    OUTPUT_REPROJ_JSON = os.path.join(os.path.dirname(OUTPUT_CSV) or '.', 'output_reproj_metrics.json')
    try:
        import json as _json
        from pipeline import _compute_per_shot_reproj_metrics as _reproj_fn
        _metrics = _reproj_fn(shots, trajectories_3d, calib)
        with open(OUTPUT_REPROJ_JSON, 'w', encoding='utf-8') as _f:
            _json.dump(_metrics, _f, ensure_ascii=False, indent=2)
        print(
            f"  ✓ 重投影指标: calib={_metrics['calib_reproj_error_px']:.2f}px"
            f" | 球轨道 mean={_metrics['overall_mean_px']:.2f}px"
            f" median={_metrics['overall_median_px']:.2f}px"
            f" rms={_metrics['overall_rms_px']:.2f}px"
            f" n_obs={_metrics['overall_n_obs']} -> {OUTPUT_REPROJ_JSON}"
        )
    except Exception as _e:
        print(f"  警告：reproj metrics 写入失败 - {_e}")

    if poses_by_frame and calib.P is not None:
        print(f"\n[步骤5b] 球员踝点与中点 3D 轨迹 -> {OUTPUT_PLAYERS_CSV}...")
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
            players_df.to_csv(OUTPUT_PLAYERS_CSV, index=False)
            print(f"  ✓ 已保存 {len(players_df)} 行球员数据")
            print(f"  ✓ 运动路程（双踝中点、连续有效帧）: 近端 {total_near_m:.2f} m | 远端 {total_far_m:.2f} m")
            if np.isfinite(avg_near_seg) or np.isfinite(avg_far_seg):
                print(
                    f"  ✓ 段速平均（连续帧 |Δs|/Δt）: 近 {avg_near_seg:.3f} m/s | 远 {avg_far_seg:.3f} m/s"
                )
            if np.isfinite(avg_near_span) or np.isfinite(avg_far_span):
                print(
                    f"  ✓ 跨度平均（总路程/首末有效帧间隔）: 近 {avg_near_span:.3f} m/s | 远 {avg_far_span:.3f} m/s"
                )
            if np.isfinite(max_near_seg) or np.isfinite(max_far_seg):
                print(f"  ✓ 最大段速: 近 {max_near_seg:.3f} m/s | 远 {max_far_seg:.3f} m/s")
            pd.DataFrame(
                [{
                    'total_path_near_m': total_near_m,
                    'total_path_far_m': total_far_m,
                    'avg_speed_segment_near_mps': avg_near_seg,
                    'avg_speed_segment_far_mps': avg_far_seg,
                    'max_speed_segment_near_mps': max_near_seg,
                    'max_speed_segment_far_mps': max_far_seg,
                    'avg_speed_span_near_mps': avg_near_span,
                    'avg_speed_span_far_mps': avg_far_span,
                    'fps': float(calib.fps),
                }]
            ).to_csv(OUTPUT_PLAYERS_STATS, index=False)
            print(f"  ✓ 球员统计摘要: {OUTPUT_PLAYERS_STATS}")
            save_player_movement_heatmap(
                players_df,
                OUTPUT_PLAYERS_HEATMAP,
                avg_near_segment_mps=avg_near_seg,
                avg_far_segment_mps=avg_far_seg,
                avg_near_span_mps=avg_near_span,
                avg_far_span_mps=avg_far_span,
                max_near_segment_mps=max_near_seg,
                max_far_segment_mps=max_far_seg,
                total_near_m=total_near_m,
                total_far_m=total_far_m,
                rally_label='回合 1',
            )
            print(f"  ✓ 球员跑动热力图: {OUTPUT_PLAYERS_HEATMAP}")
        except Exception as e:
            print(f"  警告：球员 CSV/路程 导出失败 - {e}")

    # 渲染可视化视频和图表（如果至少有一个成功重建）
    if success_count > 0:
        print(f"\n[步骤6] 渲染可视化视频...")
        print(f"  输出文件: {OUTPUT_VIDEO}")
        try:
            renderer = VideoRenderer(VIDEO_PATH, OUTPUT_VIDEO, calib)
            renderer.render(
                shots,
                trajectories_3d,
                track_draw_by_frame=track_draw_by_frame or None,
                pose_render_conf=0.25,
                pose_draw_skeleton=True,
            )
            print("  ✓ 视频渲染完成")
            print("  生成分析图表...")
            visualize_results(VIDEO_PATH, shots, trajectories_3d, calib, 'result/clip3/output_3d_result.png')
            print("  ✓ 图表已保存")
        except Exception as e:
            print(f"  错误：渲染失败 - {e}")
            import traceback
            traceback.print_exc()

    print("\n=== 处理完成（固定坐标系+稳定预测版） ===")


if __name__ == '__main__':
    main()