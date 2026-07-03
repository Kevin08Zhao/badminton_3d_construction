import os
import cv2
import json
import math
import parse
import shutil
import numpy as np
import pandas as pd

from collections import deque
from PIL import Image, ImageDraw
from .model import TrackNet, InpaintNet

# ==================== 全局配置参数 ====================
HEIGHT = 288  # 网络输入图像高度（标准配置，与WIDTH保持16:9比例）
WIDTH = 512  # 网络输入图像宽度
SIGMA = 2.5  # 高斯热力图标准差（控制目标球的大小）
DELTA_T = 1 / math.sqrt(HEIGHT ** 2 + WIDTH ** 2)  # 归一化距离单位（对角线倒数）
COOR_TH = DELTA_T * 50  # 坐标差异阈值（50像素对应的归一化距离）
IMG_FORMAT = 'png'  # 帧图像保存格式


class ResumeArgumentParser():
    """
    训练参数恢复解析器
    用于从检查点文件(checkpoint)中恢复训练配置参数
    支持训练中断后的恢复（resume training）
    """

    def __init__(self, param_dict):
        self.model_name = param_dict['model_name']  # 模型名称：TrackNet或InpaintNet
        self.seq_len = param_dict['seq_len']  # 输入序列长度（帧数）
        self.epochs = param_dict['epochs']  # 总训练轮数
        self.batch_size = param_dict['batch_size']  # 批次大小
        self.optim = param_dict['optim']  # 优化器类型（Adam等）
        self.learning_rate = param_dict['learning_rate']  # 初始学习率
        self.lr_scheduler = param_dict['lr_scheduler']  # 学习率调度策略
        self.bg_mode = param_dict['bg_mode']  # 背景处理模式
        self.alpha = param_dict['alpha']  # 损失函数权重系数
        self.frame_alpha = param_dict['frame_alpha']  # 帧混合增强参数（mixup）
        self.mask_ratio = param_dict['mask_ratio']  # 掩码比例（用于InpaintNet训练）
        self.tolerance = param_dict['tolerance']  # 评估容差阈值
        self.resume_training = param_dict['resume_training']  # 是否恢复训练标志
        self.seed = param_dict['seed']  # 随机种子（保证可复现性）
        self.save_dir = param_dict['save_dir']  # 模型保存目录
        self.debug = param_dict['debug']  # 调试模式标志
        self.verbose = param_dict['verbose']  # 详细日志输出标志


###################################  模型与工具函数 ###################################

def get_model_tmp(model_name, seq_len=None, bg_mode=None):
    """
    根据模型名称和配置参数创建对应模型实例

    Args:
        model_name (str): 模型类型
            - 'TrackNet': 轨迹检测网络，输入帧序列输出热力图
            - 'InpaintNet': 轨迹修复网络，输入坐标序列输出修复后坐标
        seq_len (int, optional): TrackNet输入序列长度（帧数），影响输入通道数
        bg_mode (str, optional): TrackNet背景处理模式，决定输入通道配置
            - '': 纯RGB模式，输入通道 = seq_len × 3
            - 'subtract': 差分图模式，输入通道 = seq_len × 1（灰度差分）
            - 'subtract_concat': RGB+差分模式，输入通道 = seq_len × 4
            - 'concat': 背景拼接模式，输入通道 = (seq_len+1) × 3（首帧为背景）

    Returns:
        model (torch.nn.Module): 配置好的模型实例，移至GPU/CPU由调用者处理

    Raises:
        ValueError: 如果model_name不是TrackNet或InpaintNet
    """

    if model_name == 'TrackNet':
        if bg_mode == 'subtract':
            # 差分模式：每帧1通道（灰度差分），总输入通道=seq_len
            model = TrackNet(in_dim=seq_len, out_dim=seq_len)
        elif bg_mode == 'subtract_concat':
            # RGB+差分：每帧4通道（RGB+灰度差分），总输入通道=seq_len×4
            model = TrackNet(in_dim=seq_len * 4, out_dim=seq_len)
        elif bg_mode == 'concat':
            # 背景拼接：(seq_len+1)帧×3通道（RGB），多一帧作为背景参考
            model = TrackNet(in_dim=(seq_len + 1) * 3, out_dim=seq_len)
        else:
            # 标准RGB模式：seq_len帧×3通道
            model = TrackNet(in_dim=seq_len * 3, out_dim=seq_len)
    elif model_name == 'InpaintNet':
        # InpaintNet固定输入维度（坐标+可见性），不受bg_mode影响
        model = InpaintNet()
    else:
        raise ValueError('Invalid model name. 必须是 TrackNet 或 InpaintNet')

    return model


def show_model_size(model):
    """
    估算并打印模型大小（参数+缓冲区）
    用于评估模型内存占用和存储需求

    Args:
        model (torch.nn.Module): 待评估的PyTorch模型

    Reference:
        https://discuss.pytorch.org/t/finding-model-size/130275/2
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print(f'Model size: {size_all_mb:.3f}MB')


def list_dirs(directory):
    """
    扩展版目录列表函数
    返回完整路径而非仅文件名，并保持字母顺序

    Args:
        directory (str): 目标目录路径

    Returns:
        List[str]: 完整路径列表（已排序）
    """
    return sorted([os.path.join(directory, path) for path in os.listdir(directory)])


def to_img(image):
    """
    将归一化图像（[0,1]浮点）转换回标准图像格式（[0,255] uint8）
    用于可视化或保存预测结果

    Args:
        image (numpy.ndarray): 归一化图像，范围[0, 1]

    Returns:
        numpy.ndarray: 标准图像，范围[0, 255]，类型uint8
    """
    image = image * 255
    image = image.astype('uint8')
    return image


def to_img_format(input, num_ch=1):
    """
    将模型输入张量格式转换为图像序列格式
    用于将网络输入(N, L*C, H, W)分离为(N, L, H, W, C)便于可视化

    Args:
        input (numpy.ndarray): 模型输入张量，形状(N, L*C, H, W)
            N: batch size, L: 序列长度, C: 每帧通道数, H/W: 高/宽
        num_ch (int): 每帧的通道数（1为灰度，3为RGB，4为RGB+差分）

    Returns:
        numpy.ndarray: 图像序列，形状(N, L, H, W)或(N, L, H, W, 3)
    """
    assert len(input.shape) == 4, 'Input must be 4D tensor.'

    if num_ch == 1:
        # 单通道差分图：(N, L, H, W)，直接返回
        return input
    else:
        # 多通道图：(N, L*C, H, W) -> (N, L, H, W, C)
        input = np.transpose(input, (0, 2, 3, 1))  # 转换为(N, H, W, L*C)
        seq_len = int(input.shape[-1] / num_ch)
        img_seq = np.array([]).reshape(0, seq_len, HEIGHT, WIDTH, 3)  # 初始化输出容器

        # 遍历batch中的每个样本
        for n in range(input.shape[0]):
            frame = np.array([]).reshape(0, HEIGHT, WIDTH, 3)
            # 按通道分离每帧（步长为num_ch）
            for f in range(0, input.shape[-1], num_ch):
                img = input[n, :, :, f:f + 3]  # 提取3个RGB通道
                frame = np.concatenate((frame, img.reshape(1, HEIGHT, WIDTH, 3)), axis=0)
            img_seq = np.concatenate((img_seq, frame.reshape(1, seq_len, HEIGHT, WIDTH, 3)), axis=0)

        return img_seq


###################################  数据获取函数 ###################################

def get_num_frames(rally_dir):
    """
    获取回合目录中的帧数量（用于验证数据完整性）

    Args:
        rally_dir (str): 回合帧目录路径
            格式：'{data_dir}/{split}/match{match_id}/frame/{rally_id}'

    Returns:
        int: 该回合的帧数（仅统计IMG_FORMAT指定格式的文件）

    Raises:
        ValueError: 如果目录不存在
    """
    try:
        frame_files = list_dirs(rally_dir)
    except:
        raise ValueError(f'{rally_dir} does not exist.')
    frame_files = [f for f in frame_files if f.split('.')[-1] == IMG_FORMAT]
    return len(frame_files)


def get_rally_dirs(data_dir, split):
    """
    递归获取指定split下的所有回合目录路径

    目录结构假设：
    data_dir/
      split/
        match{match_id}/
          frame/
            {rally_id}/
              0.png, 1.png, ...

    Args:
        data_dir (str): 数据集根目录
        split (str): 数据划分（train/val/test）

    Returns:
        List[str]: 回合目录路径列表（相对路径）
            格式：['{split}/match{match_id}/frame/{rally_id}', ...]
    """
    rally_dirs = []

    # 获取该split下的所有比赛目录
    match_dirs = os.listdir(os.path.join(data_dir, split))
    match_dirs = [os.path.join(split, d) for d in match_dirs]
    # 按match_id数字排序（确保顺序一致性）
    match_dirs = sorted(match_dirs, key=lambda s: int(s.split('match')[-1]))

    # 遍历每个比赛，获取其下的所有回合
    for match_dir in match_dirs:
        rally_dir = os.listdir(os.path.join(data_dir, match_dir, 'frame'))
        rally_dir = sorted(rally_dir)
        rally_dir = [os.path.join(match_dir, 'frame', d) for d in rally_dir]
        rally_dirs.extend(rally_dir)

    return rally_dirs


def generate_frames(video_file):
    """
    从视频文件中逐帧采样（用于数据预处理或推理）

    Args:
        video_file (str): 视频文件路径（.mp4格式）

    Returns:
        List[numpy.ndarray]: 采样得到的帧列表（BGR格式，OpenCV默认）

    Raises:
        AssertionError: 如果视频格式非.mp4
    """
    assert video_file[-4:] == '.mp4', 'Invalid video file format.'

    cap = cv2.VideoCapture(video_file)
    frame_list = []
    success = True

    # 逐帧读取直到视频结束
    while success:
        success, frame = cap.read()
        if success:
            frame_list.append(frame)

    return frame_list


###################################  可视化与结果保存 ###################################

def draw_traj(img, traj, radius=3, color='red'):
    """
    在图像上绘制羽毛球轨迹（连续点序列）
    使用白色填充、彩色边框的圆点表示轨迹点

    Args:
        img (numpy.ndarray): 输入图像（BGR格式，OpenCV标准）
        traj (deque): 轨迹点队列，每个元素为[x, y]或None（表示该时刻无检测）
        radius (int): 轨迹点绘制半径（像素）
        color (str): 轨迹颜色（'red'真实值，'yellow'预测值）

    Returns:
        numpy.ndarray: 绘制轨迹后的图像（BGR格式）
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转RGB供PIL使用
    img = Image.fromarray(img)

    # 遍历轨迹中的每个点
    for i in range(len(traj)):
        if traj[i] is not None:
            draw_x = traj[i][0]
            draw_y = traj[i][1]
            # 计算椭圆边界框（圆形）
            bbox = (draw_x - radius, draw_y - radius, draw_x + radius, draw_y + radius)
            draw = ImageDraw.Draw(img)
            # 白色填充，指定颜色边框（便于区分预测和真实轨迹）
            draw.ellipse(bbox, fill='rgb(255,255,255)', outline=color)
            del draw

    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # RGB转回BGR
    return img


def write_pred_video(video_file, pred_dict, save_file, traj_len=8, label_df=None):
    """
    生成带轨迹可视化的预测结果视频

    功能：
    1. 读取原视频保持相同帧率、分辨率和编码格式
    2. 绘制预测轨迹（黄色）和真实轨迹（红色，可选）
    3. 使用队列维护轨迹长度（滑动窗口可视化）

    Args:
        video_file (str): 输入视频路径
        pred_dict (Dict): 预测结果字典
            {'Frame': 帧ID列表, 'X': x坐标列表, 'Y': y坐标列表, 'Visibility': 可见性列表}
        save_file (str): 输出视频保存路径
        traj_len (int): 绘制的轨迹长度（历史轨迹点数量）
        label_df (pd.DataFrame, optional): 真实标签数据框（用于对比显示）

    Returns:
        None（直接写入视频文件）
    """
    # 读取视频参数
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    # 读取真实标签（如果提供）
    if label_df is not None:
        f_i, x, y, vis = label_df['Frame'], label_df['X'], label_df['Y'], label_df['Visibility']

    # 解析预测结果
    x_pred, y_pred, vis_pred = pred_dict['X'], pred_dict['Y'], pred_dict['Visibility']

    # 初始化视频写入器
    out = cv2.VideoWriter(save_file, fourcc, fps, (w, h))

    # 创建轨迹队列（deque支持高效的头尾操作）
    pred_queue = deque()
    if label_df is not None:
        gt_queue = deque()

    # 逐帧处理并绘制轨迹
    i = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        # 维护队列长度（只保留最近traj_len个轨迹点）
        if len(pred_queue) >= traj_len:
            pred_queue.pop()
        if label_df is not None and len(gt_queue) >= traj_len:
            gt_queue.pop()

        # 将当前帧坐标加入队列（如果可见性为1）
        if label_df is not None:
            gt_queue.appendleft([x[i], y[i]]) if vis[i] and i < len(label_df) else gt_queue.appendleft(None)
        pred_queue.appendleft([x_pred[i], y_pred[i]]) if vis_pred[i] else pred_queue.appendleft(None)

        # 绘制真实轨迹（红色）和预测轨迹（黄色）
        if label_df is not None:
            frame = draw_traj(frame, gt_queue, color='red')
        frame = draw_traj(frame, pred_queue, color='yellow')

        out.write(frame)
        i += 1

    out.release()
    cap.release()


def write_pred_csv(pred_dict, save_file, save_inpaint_mask=False):
    """
    将预测结果保存为CSV文件

    支持两种格式：
    1. 标准格式（用于最终提交）：帧号、可见性、XY坐标
    2. 训练格式（用于InpaintNet训练）：额外包含真实值和修复掩码

    Args:
        pred_dict (Dict): 预测结果字典
            标准格式：{'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
            训练格式：额外包含'Inpaint_Mask', 'X_GT', 'Y_GT', 'Visibility_GT'
        save_file (str): CSV保存路径
        save_inpaint_mask (bool): 是否保存InpaintNet所需的训练数据（掩码和真实值）

    Returns:
        None（直接写入CSV文件）
    """
    if save_inpaint_mask:
        # 保存临时训练数据（用于InpaintNet训练阶段）
        pred_df = pd.DataFrame({
            'Frame': pred_dict['Frame'],
            'Visibility_GT': pred_dict['Visibility_GT'],  # 真实可见性
            'X_GT': pred_dict['X_GT'],  # 真实X坐标
            'Y_GT': pred_dict['Y_GT'],  # 真实Y坐标
            'Visibility': pred_dict['Visibility'],  # 预测可见性（TrackNet输出）
            'X': pred_dict['X'],  # 预测X坐标
            'Y': pred_dict['Y'],  # 预测Y坐标
            'Inpaint_Mask': pred_dict['Inpaint_Mask']  # 需要修复的帧标记（1=需修复）
        })
    else:
        # 标准提交格式
        pred_df = pd.DataFrame({
            'Frame': pred_dict['Frame'],
            'Visibility': pred_dict['Visibility'],
            'X': pred_dict['X'],
            'Y': pred_dict['Y']
        })
    pred_df.to_csv(save_file, index=False)


def convert_gt_to_coco_json(data_dir, split, drop=False):
    """
    将真实标签CSV转换为COCO格式JSON（用于标准目标检测评估）

    COCO格式包含：
    - images: 图像文件路径和尺寸信息
    - annotations: 目标边界框（bbox）和类别（羽毛球=1）
    - categories: 类别定义

    Args:
        data_dir (str): 数据集根目录
        split (str): 数据划分（train/val/test）
        drop (bool): 是否丢弃部分帧（用于处理测试集的特殊分段）

    Returns:
        None（生成data_dir/coco_format_gt.json文件）
    """
    # 如果启用drop，加载预定义的丢弃帧范围（用于特定测试场景）
    if split == 'test' and drop:
        drop_frame_dict = json.load(open(os.path.join(data_dir, 'drop_frame.json')))
        start_frame, end_frame = drop_frame_dict['start'], drop_frame_dict['end']

    bbox_size = 10  # COCO格式要求边界框尺寸（羽毛球视为10x10像素目标）

    # 获取所有回合目录
    rally_dirs = get_rally_dirs(data_dir, split)
    rally_dirs = [os.path.join(data_dir, rally_dir) for rally_dir in rally_dirs]

    image_info = []  # COCO images字段
    annotations = []  # COCO annotations字段
    sample_count = 0  # 全局样本计数器（作为image_id和annotation_id）

    for rally_dir in rally_dirs:
        # 解析目录结构获取match_id和rally_id
        file_format_str = os.path.join('{}', 'frame', '{}')
        match_dir, rally_id = parse.parse(file_format_str, rally_dir)
        match_id = match_dir.split('match')[-1]

        # 读取标签CSV（test集使用corrected_csv，其他使用csv）
        csv_file = os.path.join(match_dir, 'corrected_csv', f'{rally_id}_ball.csv') if split == 'test' \
            else os.path.join(match_dir, 'csv', f'{rally_id}_ball.csv')
        label_df = pd.read_csv(csv_file, encoding='utf8')
        f, x, y, v = label_df['Frame'].values, label_df['X'].values, label_df['Y'].values, label_df['Visibility'].values

        # 如果启用drop，截取指定帧范围
        if split == 'test' and drop:
            rally_key = f'{match_id}_{rally_id}'
            start_f, end_f = start_frame[rally_key], end_frame[rally_key]
            f, x, y, v = f[start_f:end_f], x[start_f:end_f], y[start_f:end_f], v[start_f:end_f]

        # 获取该回合的图像尺寸（读取第一帧）
        w, h = Image.open(f'{match_dir}/frame/{rally_id}/0.{IMG_FORMAT}').size

        # 逐帧生成COCO格式数据
        for i, cx, cy, vis in zip(f, x, y, v):
            image_info.append({
                'id': sample_count,
                'width': w,
                'height': h,
                'file_name': f'{match_dir}/frame/{rally_id}/{i}.{IMG_FORMAT}'
            })

            # 仅当羽毛球可见时（vis>0）添加边界框标注
            if vis > 0:
                annotations.append({
                    'id': sample_count,
                    'image_id': sample_count,
                    'category_id': 1,  # 羽毛球类别
                    'bbox': [int(cx - bbox_size / 2), int(cy - bbox_size / 2), bbox_size, bbox_size],
                    'ignore': 0,
                    'area': bbox_size * bbox_size,
                    'segmentation': [],
                    'iscrowd': 0
                })
            sample_count += 1

    # 组装COCO数据结构
    coco_data = {
        'info': {},
        'licenses': [],
        'categories': [{'id': 1, 'name': 'shuttlecock'}],  # 定义羽毛球类别
        'images': image_info,
        'annotations': annotations,
    }

    with open(f'{data_dir}/coco_format_gt.json', 'w') as f:
        json.dump(coco_data, f)


###################################  数据预处理函数 ###################################

def generate_data_frames(video_file):
    """
    从数据集视频中提取帧并保存为图像文件
    同时生成该回合的中值背景图（用于背景减除）

    文件结构：
    - 视频路径：{data_dir}/{split}/match{match_id}/video/{rally_id}.mp4
    - 帧保存路径：{data_dir}/{split}/match{match_id}/frame/{rally_id}/0.png, 1.png, ...

    Args:
        video_file (str): 视频文件路径（需符合上述格式）

    Returns:
        None（生成帧图像和中值图文件）

    Actions:
        1. 检查视频与对应CSV标签文件匹配
        2. 逐帧提取并保存为PNG格式
        3. 计算所有帧的中值背景并保存为median.npz
    """
    # 验证文件格式和存在性
    try:
        assert video_file[-4:] == '.mp4', 'Invalid video file format.'
    except:
        raise ValueError(f'{video_file} is not a video file.')

    # 解析路径获取对应CSV文件
    file_format_str = os.path.join('{}', 'video', '{}.mp4')
    match_dir, rally_id = parse.parse(file_format_str, video_file)
    csv_file = os.path.join(match_dir, 'csv', f'{rally_id}_ball.csv')
    label_df = pd.read_csv(csv_file, encoding='utf8')

    assert os.path.exists(video_file) and os.path.exists(csv_file), 'Video file or csv file does not exist.'

    rally_dir = os.path.join(match_dir, 'frame', rally_id)

    # 检查是否已处理（避免重复提取）
    if not os.path.exists(rally_dir):
        os.makedirs(rally_dir)  # 首次处理，创建目录
    else:
        # 已存在目录，检查帧数是否匹配（防止之前处理中断或出错）
        label_df = pd.read_csv(csv_file, encoding='utf8')
        if len(list_dirs(rally_dir)) < len(label_df):
            # 帧数不足，删除重建
            shutil.rmtree(rally_dir)
            os.makedirs(rally_dir)
        else:
            # 已完整处理，跳过
            return

    cap = cv2.VideoCapture(video_file)
    frames = []
    success = True

    # 逐帧读取并保存，直到视频结束或达到标签帧数上限
    while success and len(frames) != len(label_df):
        success, frame = cap.read()
        if success:
            frames.append(frame)
            # 保存为PNG（无损压缩，保持图像质量）
            cv2.imwrite(os.path.join(rally_dir, f'{len(frames) - 1}.{IMG_FORMAT}'), frame)

    # 计算中值背景图（像素级中值，有效抑制移动物体）
    median = np.median(np.array(frames), 0)
    median = median[..., ::-1]  # BGR转RGB（与PIL兼容）
    # 保存为NPZ（无损保存浮点精度，不能存为图像格式避免精度损失）
    np.savez(os.path.join(rally_dir, 'median.npz'), median=median)


def get_match_median(match_dir):
    """
    生成比赛级别的中值背景图（基于该比赛所有回合的中值图）
    用于该比赛下没有独立中值图的回合（或作为更稳定的背景参考）

    Args:
        match_dir (str): 比赛目录路径
            格式：'{data_dir}/{split}/match{match_id}'

    Returns:
        None（保存median.npz到比赛目录）
    """
    medians = []

    # 遍历该比赛的所有回合
    rally_dirs = list_dirs(os.path.join(match_dir, 'frame'))
    for rally_dir in rally_dirs:
        file_format_str = os.path.join('{}', 'frame', '{}')
        _, rally_id = parse.parse(file_format_str, rally_dir)

        # 如果回合没有中值图，临时生成
        if not os.path.exists(os.path.join(rally_dir, 'median.npz')):
            get_rally_median(os.path.join(match_dir, 'video', f'{rally_id}.mp4'))

        # 加载回合中值图
        frame = np.load(os.path.join(rally_dir, 'median.npz'))['median']
        medians.append(frame)

    # 计算跨回合的中值（更稳定的背景，消除单回合的特殊噪声）
    median = np.median(np.array(medians), 0)
    np.savez(os.path.join(match_dir, 'median.npz'), median=median)


def get_rally_median(video_file):
    """
    从视频生成回合级别的中值背景图

    中值背景作用：
    1. 背景减除：突出移动物体（羽毛球、球员）
    2. 处理遮挡：当羽毛球被球员遮挡时，中值图提供位置先验

    Args:
        video_file (str): 视频文件路径
            格式：'{data_dir}/{split}/match{match_id}/video/{rally_id}.mp4'

    Returns:
        None（保存median.npz到对应frame目录）
    """
    frames = []

    # 解析路径获取保存目录
    file_format_str = os.path.join('{}', 'video', '{}.mp4')
    match_dir, rally_id = parse.parse(file_format_str, video_file)
    save_dir = os.path.join(match_dir, 'frame', rally_id)

    # 采样所有帧（小视频可直接全采样，大视频应间隔采样以节省内存）
    cap = cv2.VideoCapture(video_file)
    success = True
    while success:
        success, frame = cap.read()
        if success:
            frames.append(frame)

    # 沿时间轴取中值（每个像素位置取所有帧的中值）
    median = np.median(np.array(frames), 0)[..., ::-1]  # BGR转RGB
    np.savez(os.path.join(save_dir, 'median.npz'), median=median)


def re_generate_median_files(data_dir):
    """
    批量重新生成所有中值背景文件（修复或更新用）
    遍历train/val/test所有比赛的所有回合

    Args:
        data_dir (str): 数据集根目录
    """
    for split in ['train', 'val', 'test']:
        match_dirs = list_dirs(os.path.join(data_dir, split))
        for match_dir in match_dirs:
            match_name = match_dir.split('/')[-1]
            video_files = list_dirs(os.path.join(match_dir, 'video'))

            # 先为每个回合生成中值图
            for video_file in video_files:
                print(f'Processing {video_file}...')
                get_rally_median(video_file)

            # 再为整场比赛生成中值图
            get_match_median(match_dir)
            print(f'Finish processing {match_name}.')