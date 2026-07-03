import os
import cv2
import math
import parse  # 用于字符串解析和模式匹配
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, IterableDataset
from .general_tmp import get_rally_dirs, get_match_median, HEIGHT, WIDTH, SIGMA, IMG_FORMAT

data_dir = 'data'


class Shuttlecock_Trajectory_Dataset(Dataset):
    """
    羽毛球轨迹数据集（Shuttlecock Trajectory Dataset）
    支持两种工作模式：
    1. heatmap模式：用于TrackNet训练，输出高斯热力图作为标签
    2. coordinate模式：用于InpaintNet训练，输出坐标值

    数据集格式参考：https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw
    """

    def __init__(self,
                 root_dir=data_dir,
                 split='train',
                 seq_len=8,  # 输入序列长度（帧数），TrackNet默认8帧
                 sliding_step=1,  # 滑动窗口步长，控制数据重叠度
                 data_mode='heatmap',  # 'heatmap'或'coordinate'
                 bg_mode='',  # 背景处理模式：'', 'subtract', 'subtract_concat', 'concat'
                 frame_alpha=-1,  # 帧混合(mixup)参数，>0时启用数据增强，-1表示禁用
                 rally_dir=None,  # 指定单个回合目录（用于单独处理某个回合）
                 frame_arr=None,  # 直接传入帧数组（用于推理阶段）
                 pred_dict=None,  # 预测结果字典（用于InpaintNet推理）
                 padding=False,  # 序列末尾是否填充（仅在sliding_step==seq_len时有效）
                 debug=False,  # 调试模式，只加载部分数据
                 HEIGHT=HEIGHT,  # 网络输入图像高度（默认512）
                 WIDTH=WIDTH,  # 网络输入图像宽度（默认512）
                 SIGMA=SIGMA,  # 高斯热力图标准差（控制目标大小）
                 median=None  # 预计算的中值背景图像
                 ):
        """
        初始化数据集

        Args:
            root_dir (str): 数据集根目录路径
            split (str): 数据划分，'train'/'test'/'val'
            seq_len (int): 输入序列长度（帧数）
            sliding_step (int): 滑动窗口步长，生成输入序列时的步幅
            data_mode (str): 数据模式
                - 'heatmap': TrackNet输入，返回热力图标签
                - 'coordinate': InpaintNet输入，返回坐标值
            bg_mode (str): 背景处理模式，用于背景减除增强
                - '': 返回原始帧序列（RGB，3通道）
                - 'subtract': 返回差分帧（单通道，绝对差值）
                - 'subtract_concat': 返回RGB+差分（4通道）
                - 'concat': 首帧为背景，后续为原始帧（用于背景建模）
            frame_alpha (float): 帧间混合系数（Beta分布参数），用于时序数据增强
            rally_dir (str): 指定特定回合目录路径（单回合处理模式）
            frame_arr (numpy.ndarray): 直接传入的帧序列数组（推理模式）
            pred_dict (Dict): TrackNet预测结果字典（InpaintNet推理用）
                格式：{
                    'X': x坐标列表,
                    'Y': y坐标列表,
                    'Visibility': 可见性列表,
                    'Inpaint_Mask': 需要修复的掩码列表,
                    'Img_scaler': 图像缩放因子元组,
                    'Img_shape': 原始图像尺寸元组
                }
            padding (bool): 序列不足时是否用最后一帧填充
            debug (bool): 调试模式（仅加载256条数据）
            HEIGHT/WIDTH (int): 网络输入图像尺寸
            SIGMA (int): 高斯热力图sigma值（半径）
            median (np.ndarray): 预计算的中值背景图像（用于背景减除）
        """

        # 参数校验
        assert split in ['train', 'test', 'val'], f'Invalid split: {split}, should be train, test or val'
        assert data_mode in ['heatmap', 'coordinate'], f'Invalid data_mode: {data_mode}'
        assert bg_mode in ['', 'subtract', 'subtract_concat', 'concat'], f'Invalid bg_mode: {bg_mode}'

        # 图像尺寸配置
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH

        # 高斯热力图参数：mag控制峰值强度，sigma控制扩散范围
        self.mag = 1
        self.sigma = SIGMA

        self.root_dir = root_dir
        # 如果是单回合模式，自动从路径解析split；否则使用传入的split
        self.split = split if rally_dir is None else self._get_split(rally_dir)
        self.seq_len = seq_len
        self.sliding_step = sliding_step
        self.data_mode = data_mode
        self.bg_mode = bg_mode
        self.frame_alpha = frame_alpha

        # 推理模式数据（直接传入，不从磁盘加载）
        self.frame_arr = frame_arr
        self.pred_dict = pred_dict
        # padding仅在滑动步长等于序列长度时启用（无重叠滑动）
        self.padding = padding and self.sliding_step == self.seq_len

        # 根据输入类型选择数据加载策略
        if self.frame_arr is not None:
            # 模式1：TrackNet推理 - 直接从帧数组生成输入
            assert self.data_mode == 'heatmap', 'frame_arr仅适用于heatmap模式'
            self.data_dict, self.img_config = self._gen_input_from_frame_arr()
            if self.bg_mode:
                # 计算或使用中值背景图
                if median is None:
                    median = np.median(self.frame_arr, 0)  # 沿时间轴取中值
                if self.bg_mode == 'concat':
                    # concat模式需要调整维度顺序以匹配网络输入 (C,H,W)
                    median = Image.fromarray(median.astype('uint8'))
                    median = np.array(median.resize(size=(self.WIDTH, self.HEIGHT)))
                    self.median = np.moveaxis(median, -1, 0)
                else:
                    self.median = median

        elif self.pred_dict is not None:
            # 模式2：InpaintNet推理 - 从预测字典生成输入
            assert self.data_mode == 'coordinate', 'pred_dict仅适用于coordinate模式'
            self.data_dict, self.img_config = self._gen_input_from_pred_dict()

        else:
            # 模式3：训练/评估模式 - 从磁盘加载整个数据集

            # 构建回合索引字典（用于快速查找回合路径和索引映射）
            self.rally_dict = self._get_rally_dict()

            # 生成或加载图像配置文件（包含每个回合的缩放因子和原始尺寸）
            img_config_file = os.path.join(self.root_dir, f'img_config_{self.HEIGHT}x{self.WIDTH}_{self.split}.npz')
            if not os.path.exists(img_config_file):
                self._gen_rally_img_congif_file(img_config_file)
            img_config = np.load(img_config_file)
            self.img_config = {key: img_config[key] for key in img_config.keys()}

            # 生成输入数据
            if rally_dir is not None:
                # 单回合训练/测试模式
                self.data_dict = self._gen_input_from_rally_dir(rally_dir)
            else:
                # 整数据集训练/测试模式：生成或加载预处理的npz文件
                input_file = os.path.join(self.root_dir,
                                          f'data_l{self.seq_len}_s{self.sliding_step}_{self.data_mode}_{self.split}.npz')
                if not os.path.exists(input_file):
                    self._gen_input_file(file_name=input_file)
                data_dict = np.load(input_file)
                self.data_dict = {key: data_dict[key] for key in data_dict.keys()}

            # 调试模式：仅保留前256条数据用于快速测试
            if debug:
                num_data = 256
                for key in self.data_dict.keys():
                    self.data_dict[key] = self.data_dict[key][:num_data]

    def _get_rally_dict(self):
        """
        构建回合索引映射字典
        Returns:
            dict: {'i2p': {索引: 路径}, 'p2i': {路径: 索引}}
        """
        rally_dirs = get_rally_dirs(self.root_dir, self.split)
        rally_dict = {
            'i2p': {i: os.path.join(self.root_dir, rally_dir) for i, rally_dir in enumerate(rally_dirs)},
            'p2i': {os.path.join(self.root_dir, rally_dir): i for i, rally_dir in enumerate(rally_dirs)}
        }
        return rally_dict

    def _get_rally_i(self, rally_dir):
        """ 根据回合路径获取对应索引 """
        return self.rally_dict['p2i'].get(rally_dir, None)

    def _get_split(self, rally_dir):
        """ 从回合目录路径解析split（train/test/val） """
        file_format_str = os.path.join(self.root_dir, '{}', 'match{}')
        split, _ = parse.parse(file_format_str, rally_dir)
        return split

    def _gen_rally_img_congif_file(self, file_name):
        """
        生成回合图像配置文件（每个回合的缩放因子和原始尺寸）
        用于将网络输出坐标映射回原图分辨率
        """
        img_scaler = []  # 缩放因子列表 (num_rally, 2)
        img_shape = []  # 原始尺寸列表 (num_rally, 2)

        for rally_i, rally_dir in tqdm(self.rally_dict['i2p'].items()):
            # 读取第一帧获取原始尺寸
            w, h = Image.open(os.path.join(rally_dir, f'0.{IMG_FORMAT}')).size
            w_scaler, h_scaler = w / self.WIDTH, h / self.HEIGHT
            img_scaler.append((w_scaler, h_scaler))
            img_shape.append((w, h))

        np.savez(file_name, img_scaler=img_scaler, img_shape=img_shape)

    def _gen_input_file(self, file_name):
        """
        生成整个数据集的预处理文件（.npz格式）
        遍历所有回合，合并生成训练/测试用的输入序列
        """
        print('Generate input file...')

        if self.data_mode == 'heatmap':
            # heatmap模式存储：ID、帧文件路径、坐标、可见性
            id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
            frame_file = np.array([]).reshape(0, self.seq_len)
            coor = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)

            # 遍历所有回合生成输入序列
            for rally_i, rally_dir in tqdm(self.rally_dict['i2p'].items()):
                data_dict = self._gen_input_from_rally_dir(rally_dir)
                id = np.concatenate((id, data_dict['id']), axis=0)
                frame_file = np.concatenate((frame_file, data_dict['frame_file']), axis=0)
                coor = np.concatenate((coor, data_dict['coor']), axis=0)
                vis = np.concatenate((vis, data_dict['vis']), axis=0)

            np.savez(file_name, id=id, frame_file=frame_file, coor=coor, vis=vis)

        else:
            # coordinate模式存储：ID、真实坐标、预测坐标、真实可见性、预测可见性、修复掩码
            id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
            coor = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            coor_pred = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)
            pred_vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)
            inpaint_mask = np.array([], dtype=np.float32).reshape(0, self.seq_len)

            for rally_i, rally_dir in tqdm(self.rally_dict['i2p'].items()):
                data_dict = self._gen_input_from_rally_dir(rally_dir)
                id = np.concatenate((id, data_dict['id']), axis=0)
                coor = np.concatenate((coor, data_dict['coor']), axis=0)
                coor_pred = np.concatenate((coor_pred, data_dict['coor_pred']), axis=0)
                vis = np.concatenate((vis, data_dict['vis']), axis=0)
                pred_vis = np.concatenate((pred_vis, data_dict['pred_vis']), axis=0)
                inpaint_mask = np.concatenate((inpaint_mask, data_dict['inpaint_mask']), axis=0)

            np.savez(file_name, id=id, coor=coor, coor_pred=coor_pred,
                     vis=vis, pred_vis=pred_vis, inpaint_mask=inpaint_mask)

    def _gen_input_from_rally_dir(self, rally_dir):
        """
        从单个回合目录生成输入序列（滑动窗口）

        对于heatmap模式：
            - 从CSV读取真实标签（test集从corrected_csv读取，train/val从csv读取）
            - 生成帧文件路径列表
            - 滑动窗口提取序列

        对于coordinate模式：
            - 从predicted_csv读取TrackNet预测结果作为输入
            - 同时读取真实标签作为监督信号
            - inpaint_mask标记需要修复的帧（预测失败或遮挡）
        """
        rally_i = self._get_rally_i(rally_dir)

        # 解析目录结构：{match_dir}/frame/{rally_id} 或 {match_dir}/{rally_id}
        file_format_str = os.path.join('{}', 'frame', '{}')
        match_dir, rally_id = parse.parse(file_format_str, rally_dir)

        if self.data_mode == 'heatmap':
            # 读取标签CSV文件
            if 'test' in rally_dir:
                csv_file = os.path.join(match_dir, 'corrected_csv', f'{rally_id}_ball.csv')
            else:
                csv_file = os.path.join(match_dir, 'csv', f'{rally_id}_ball.csv')

            assert os.path.exists(csv_file), f'{csv_file} does not exist.'
            label_df = pd.read_csv(csv_file, encoding='utf8').sort_values(by='Frame').fillna(0)

            # 构建帧文件路径列表
            f_file = np.array([os.path.join(rally_dir, f'{f_id}.{IMG_FORMAT}')
                               for f_id in label_df['Frame']])
            x, y, v = np.array(label_df['X']), np.array(label_df['Y']), np.array(label_df['Visibility'])

            # 初始化数组
            id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
            frame_file = np.array([]).reshape(0, self.seq_len)
            coor = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)

            # 滑动窗口生成序列
            last_idx = -1
            for i in range(0, len(f_file), self.sliding_step):
                tmp_idx, tmp_frames, tmp_coor, tmp_vis = [], [], [], []

                # 构建单个输入序列
                for f in range(self.seq_len):
                    if i + f < len(f_file):
                        tmp_idx.append((rally_i, i + f))  # (回合索引, 帧偏移)
                        tmp_frames.append(f_file[i + f])
                        tmp_coor.append((x[i + f], y[i + f]))
                        tmp_vis.append(v[i + f])
                        last_idx = i + f
                    else:
                        # 序列不足时填充最后一帧（如果启用padding）
                        if self.padding:
                            tmp_idx.append((rally_i, last_idx))
                            tmp_frames.append(f_file[last_idx])
                            tmp_coor.append((x[last_idx], y[last_idx]))
                            tmp_vis.append(v[last_idx])
                        else:
                            break

                # 仅保留完整长度的序列
                if len(tmp_frames) == self.seq_len:
                    assert len(tmp_frames) == len(tmp_coor) == len(tmp_vis), \
                        'Length of frames, coordinates and visibilities are not equal.'
                    id = np.concatenate((id, [tmp_idx]), axis=0)
                    frame_file = np.concatenate((frame_file, [tmp_frames]), axis=0)
                    coor = np.concatenate((coor, [tmp_coor]), axis=0)
                    vis = np.concatenate((vis, [tmp_vis]), axis=0)

            return dict(id=id, frame_file=frame_file, coor=coor, vis=vis)

        else:
            # coordinate模式：读取TrackNet预测结果
            pred_csv_file = os.path.join(match_dir, 'predicted_csv', f'{rally_id}_ball.csv')
            assert os.path.exists(pred_csv_file), f'{pred_csv_file} does not exist.'
            pred_df = pd.read_csv(pred_csv_file, encoding='utf8').sort_values(by='Frame').fillna(0)

            f_file = np.array([os.path.join(rally_dir, f'{f_id}.{IMG_FORMAT}')
                               for f_id in pred_df['Frame']])
            # GT = Ground Truth（真实标签）
            x, y, v = np.array(pred_df['X_GT']), np.array(pred_df['Y_GT']), np.array(pred_df['Visibility_GT'])
            # 预测值（TrackNet输出，作为InpaintNet的输入）
            x_pred, y_pred, v_pred = np.array(pred_df['X']), np.array(pred_df['Y']), np.array(pred_df['Visibility'])
            inpaint = np.array(pred_df['Inpaint_Mask'])  # 1表示需要修复，0表示无需修复

            id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
            coor = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            coor_pred = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)
            pred_vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)
            inpaint_mask = np.array([], dtype=np.float32).reshape(0, self.seq_len)

            last_idx = -1
            for i in range(0, len(f_file), self.sliding_step):
                tmp_idx, tmp_coor, tmp_coor_pred, tmp_vis, tmp_vis_pred, tmp_inpaint = [], [], [], [], [], []

                for f in range(self.seq_len):
                    if i + f < len(f_file):
                        tmp_idx.append((rally_i, i + f))
                        tmp_coor.append((x[i + f], y[i + f]))
                        tmp_coor_pred.append((x_pred[i + f], y_pred[i + f]))
                        tmp_vis.append(v[i + f])
                        tmp_vis_pred.append(v_pred[i + f])
                        tmp_inpaint.append(inpaint[i + f])
                    else:
                        if self.padding:
                            tmp_idx.append((rally_i, last_idx))
                            tmp_coor.append((x[last_idx], y[last_idx]))
                            tmp_coor_pred.append((x_pred[last_idx], y_pred[last_idx]))
                            tmp_vis.append(v[last_idx])
                            tmp_vis_pred.append(v_pred[last_idx])
                            tmp_inpaint.append(inpaint[last_idx])
                        else:
                            break

                if len(tmp_idx) == self.seq_len:
                    id = np.concatenate((id, [tmp_idx]), axis=0)
                    coor = np.concatenate((coor, [tmp_coor]), axis=0)
                    coor_pred = np.concatenate((coor_pred, [tmp_coor_pred]), axis=0)
                    vis = np.concatenate((vis, [tmp_vis]), axis=0)
                    pred_vis = np.concatenate((pred_vis, [tmp_vis_pred]), axis=0)
                    inpaint_mask = np.concatenate((inpaint_mask, [tmp_inpaint]), axis=0)

            return dict(id=id, coor=coor, coor_pred=coor_pred, vis=vis, pred_vis=pred_vis, inpaint_mask=inpaint_mask)

    def _gen_input_from_frame_arr(self):
        """
        从帧数组生成输入（用于TrackNet推理）
        不加载标签，仅生成数据索引和图像尺寸配置
        """
        # 计算图像缩放因子（原图 -> 网络输入尺寸）
        h, w, _ = self.frame_arr[0].shape
        h_scaler, w_scaler = h / self.HEIGHT, w / self.WIDTH

        id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
        last_idx = -1

        # 滑动窗口生成索引序列
        for i in range(0, len(self.frame_arr), self.sliding_step):
            tmp_idx = []
            for f in range(self.seq_len):
                if i + f < len(self.frame_arr):
                    tmp_idx.append((0, i + f))
                    last_idx = i + f
                else:
                    if self.padding:
                        tmp_idx.append((0, last_idx))
                    else:
                        break
            if len(tmp_idx) == self.seq_len:
                id = np.concatenate((id, [tmp_idx]), axis=0)

        return dict(id=id), dict(img_scaler=(w_scaler, h_scaler), img_shape=(w, h))

    def _gen_input_from_pred_dict(self):
        """
        从预测字典生成输入（用于InpaintNet推理）
        将TrackNet的预测结果组织成序列格式
        """
        id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
        coor_pred = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
        pred_vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)
        inpaint_mask = np.array([], dtype=np.float32).reshape(0, self.seq_len)

        x_pred, y_pred, vis_pred = self.pred_dict['X'], self.pred_dict['Y'], self.pred_dict['Visibility']
        inpaint = self.pred_dict['Inpaint_Mask']

        assert len(x_pred) == len(y_pred) == len(vis_pred) == len(inpaint)

        last_idx = -1
        for i in range(0, len(inpaint), self.sliding_step):
            tmp_idx, tmp_coor_pred, tmp_vis_pred, tmp_inpaint = [], [], [], []

            for f in range(self.seq_len):
                if i + f < len(inpaint):
                    tmp_idx.append((0, i + f))
                    tmp_coor_pred.append((x_pred[i + f], y_pred[i + f]))
                    tmp_vis_pred.append(vis_pred[i + f])
                    tmp_inpaint.append(inpaint[i + f])
                    last_idx = i + f
                else:
                    if self.padding:
                        tmp_idx.append((0, last_idx))
                        tmp_coor_pred.append((x_pred[last_idx], y_pred[last_idx]))
                        tmp_vis_pred.append(vis_pred[last_idx])
                        tmp_inpaint.append(inpaint[last_idx])
                    else:
                        break

            if len(tmp_idx) == self.seq_len:
                coor_pred = np.concatenate((coor_pred, [tmp_coor_pred]), axis=0)
                pred_vis = np.concatenate((pred_vis, [tmp_vis_pred]), axis=0)
                inpaint_mask = np.concatenate((inpaint_mask, [tmp_inpaint]), axis=0)
                id = np.concatenate((id, [tmp_idx]), axis=0)

        return (dict(id=id, coor_pred=coor_pred, pred_vis=pred_vis, inpaint_mask=inpaint_mask),
                dict(img_scaler=self.pred_dict['Img_scaler'], img_shape=self.pred_dict['Img_shape']))

    def _get_heatmap(self, cx, cy):
        """
        生成以(cx, cy)为中心的高斯热力图

        Args:
            cx, cy: 中心点坐标（在网络输入尺寸下的坐标）
        Returns:
            归一化的高斯热力图，形状(1, HEIGHT, WIDTH)
        """
        if cx == cy == 0:
            # 不可见/缺失目标返回全零图
            return np.zeros((1, self.HEIGHT, self.WIDTH))

        # 生成网格坐标
        x, y = np.meshgrid(np.linspace(1, self.WIDTH, self.WIDTH),
                           np.linspace(1, self.HEIGHT, self.HEIGHT))

        # 计算距离平方和
        heatmap = ((y - (cy + 1)) ** 2) + ((x - (cx + 1)) ** 2)

        # 二值化：sigma范围内为1，范围外为0（简化的高斯，非连续分布）
        heatmap[heatmap <= self.sigma ** 2] = 1.
        heatmap[heatmap > self.sigma ** 2] = 0.
        heatmap = heatmap * self.mag

        return heatmap.reshape(1, self.HEIGHT, self.WIDTH)

    def __len__(self):
        """ 返回数据集样本数量 """
        return len(self.data_dict['id'])

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本

        根据配置返回不同格式：
        - heatmap训练模式: (data_idx, frames, heatmaps, coor, vis)
        - coordinate训练模式: (data_idx, coor_pred, coor_gt, vis_pred, vis_gt, inpaint)
        - heatmap推理模式: (data_idx, frames)
        - coordinate推理模式: (data_idx, coor_pred, inpaint)

        注意：如果启用frame_alpha>0（帧混合增强），会生成插值帧和对应标签
        """
        if self.frame_arr is not None:
            # TrackNet推理模式
            data_idx = self.data_dict['id'][idx]  # (seq_len, 2)
            imgs = self.frame_arr[data_idx[:, 1], ...]  # 根据索引提取帧 (seq_len, H, W, 3)

            if self.bg_mode:
                median_img = self.median

            # 处理帧序列：调整尺寸、背景减除、维度重排
            frames = np.array([]).reshape(0, self.HEIGHT, self.WIDTH)
            for i in range(self.seq_len):
                img = Image.fromarray(imgs[i])
                if self.bg_mode == 'subtract':
                    # 背景减除：转灰度差分图
                    img = Image.fromarray(np.sum(np.absolute(img - median_img), 2).astype('uint8'))
                    img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                    img = img.reshape(1, self.HEIGHT, self.WIDTH)
                elif self.bg_mode == 'subtract_concat':
                    # RGB + 差分图（4通道）
                    diff_img = Image.fromarray(np.sum(np.absolute(img - median_img), 2).astype('uint8'))
                    diff_img = np.array(diff_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    diff_img = diff_img.reshape(1, self.HEIGHT, self.WIDTH)
                    img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                    img = np.moveaxis(img, -1, 0)  # HWC -> CHW
                    img = np.concatenate((img, diff_img), axis=0)
                else:
                    img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                    img = np.moveaxis(img, -1, 0)

                frames = np.concatenate((frames, img), axis=0)

            if self.bg_mode == 'concat':
                # 首帧为背景图
                frames = np.concatenate((median_img, frames), axis=0)

            frames /= 255.  # 归一化到[0,1]
            return data_idx, frames

        elif self.pred_dict is not None:
            # InpaintNet推理模式
            data_idx = self.data_dict['id'][idx]
            coor_pred = self.data_dict['coor_pred'][idx]  # (seq_len, 2)
            inpaint = self.data_dict['inpaint_mask'][idx].reshape(-1, 1)  # (seq_len, 1)
            w, h = self.img_config['img_shape']

            # 坐标归一化到[0,1]
            coor_pred[:, 0] = coor_pred[:, 0] / w
            coor_pred[:, 1] = coor_pred[:, 1] / h

            return data_idx, coor_pred, inpaint

        elif self.data_mode == 'heatmap':
            if self.frame_alpha > 0:
                # 启用帧混合数据增强（Frame Mixup）
                # 通过混合相邻帧生成更多训练样本，增强时序一致性
                data_idx = self.data_dict['id'][idx]
                frame_file = self.data_dict['frame_file'][idx]
                coor = self.data_dict['coor'][idx]
                vis = self.data_dict['vis'][idx]
                w, h = self.img_config['img_shape'][data_idx[0][0]]
                w_scaler, h_scaler = self.img_config['img_scaler'][data_idx[0][0]]

                # 加载中值背景图（用于背景减除）
                if self.bg_mode:
                    file_format_str = os.path.join('{}', 'frame', '{}', '{}.' + IMG_FORMAT)
                    match_dir, rally_id, _ = parse.parse(file_format_str, frame_file[0])
                    median_file = os.path.join(match_dir, 'median.npz')
                    if not os.path.exists(median_file):
                        median_file = os.path.join(match_dir, 'frame', rally_id, 'median.npz')
                    assert os.path.exists(median_file), f'{median_file} does not exist.'
                    median_img = np.load(median_file)['median']

                # 从Beta分布采样混合系数lambda
                lamb = np.random.beta(self.frame_alpha, self.frame_alpha)

                # 初始化第一帧
                prev_img = Image.open(frame_file[0])
                # 应用背景减除（根据bg_mode）
                if self.bg_mode == 'subtract':
                    prev_img = Image.fromarray(np.sum(np.absolute(prev_img - median_img), 2).astype('uint8'))
                    prev_img = np.array(prev_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    prev_img = prev_img.reshape(1, self.HEIGHT, self.WIDTH)
                elif self.bg_mode == 'subtract_concat':
                    diff_img = Image.fromarray(np.sum(np.absolute(prev_img - median_img), 2).astype('uint8'))
                    diff_img = np.array(diff_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    diff_img = diff_img.reshape(1, self.HEIGHT, self.WIDTH)
                    prev_img = np.array(prev_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    prev_img = np.moveaxis(prev_img, -1, 0)
                    prev_img = np.concatenate((prev_img, diff_img), axis=0)
                else:
                    prev_img = np.array(prev_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    prev_img = np.moveaxis(prev_img, -1, 0)

                # 初始化标签
                prev_coor = coor[0]
                prev_vis = vis[0]
                prev_heatmap = self._get_heatmap(int(coor[0][0] / w_scaler), int(coor[0][1] / h_scaler))

                # 保持第一维为时间步，便于后续重采样
                if self.bg_mode == 'subtract':
                    frames = prev_img.reshape(1, 1, self.HEIGHT, self.WIDTH)
                elif self.bg_mode == 'subtract_concat':
                    frames = prev_img.reshape(1, 4, self.HEIGHT, self.WIDTH)
                else:
                    frames = prev_img.reshape(1, 3, self.HEIGHT, self.WIDTH)

                tmp_coor = prev_coor.reshape(1, -1)
                tmp_vis = prev_vis.reshape(1, -1)
                heatmaps = prev_heatmap

                # 遍历剩余帧，生成混合帧和标签
                for i in range(1, self.seq_len):
                    cur_img = Image.open(frame_file[i])
                    if self.bg_mode == 'subtract':
                        cur_img = Image.fromarray(np.sum(np.absolute(cur_img - median_img), 2).astype('uint8'))
                        cur_img = np.array(cur_img.resize(size=(self.WIDTH, self.HEIGHT)))
                        cur_img = cur_img.reshape(1, self.HEIGHT, self.WIDTH)
                    elif self.bg_mode == 'subtract_concat':
                        diff_img = Image.fromarray(np.sum(np.absolute(cur_img - median_img), 2).astype('uint8'))
                        diff_img = np.array(diff_img.resize(size=(self.WIDTH, self.HEIGHT)))
                        diff_img = diff_img.reshape(1, self.HEIGHT, self.WIDTH)
                        cur_img = np.array(cur_img.resize(size=(self.WIDTH, self.HEIGHT)))
                        cur_img = np.moveaxis(cur_img, -1, 0)
                        cur_img = np.concatenate((cur_img, diff_img), axis=0)
                    else:
                        cur_img = np.array(cur_img.resize(size=(self.WIDTH, self.HEIGHT)))
                        cur_img = np.moveaxis(cur_img, -1, 0)

                    # 帧混合：prev * lambda + cur * (1-lambda)
                    inter_img = prev_img * lamb + cur_img * (1 - lamb)

                    # 标签混合逻辑
                    if vis[i] == 0:
                        # 当前帧不可见，继承前一帧标签
                        inter_coor = prev_coor
                        inter_vis = prev_vis
                        cur_heatmap = prev_heatmap
                        inter_heatmap = cur_heatmap
                    elif prev_vis == 0 or math.sqrt(
                            pow(prev_coor[0] - coor[i][0], 2) + pow(prev_coor[1] - coor[i][1], 2)) < 10:
                        # 前一帧不可见或距离过近，使用当前帧标签
                        inter_coor = coor[i]
                        inter_vis = vis[i]
                        cur_heatmap = self._get_heatmap(int(inter_coor[0] / w_scaler), int(inter_coor[1] / h_scaler))
                        inter_heatmap = cur_heatmap
                    else:
                        # 线性插值坐标和热力图
                        inter_coor = coor[i]
                        inter_vis = vis[i]
                        cur_heatmap = self._get_heatmap(int(coor[i][0] / w_scaler), int(coor[i][1] / h_scaler))
                        inter_heatmap = prev_heatmap * lamb + cur_heatmap * (1 - lamb)

                    # 拼接数据（保存插值帧和原始帧，后续重采样）
                    tmp_coor = np.concatenate((tmp_coor, inter_coor.reshape(1, -1), coor[i].reshape(1, -1)), axis=0)
                    tmp_vis = np.concatenate(
                        (tmp_vis, np.array([inter_vis]).reshape(1, -1), np.array([vis[i]]).reshape(1, -1)), axis=0)
                    frames = np.concatenate((frames, inter_img[None, :, :, :], cur_img[None, :, :, :]), axis=0)
                    heatmaps = np.concatenate((heatmaps, inter_heatmap, cur_heatmap), axis=0)

                    prev_img, prev_heatmap, prev_coor, prev_vis = cur_img, cur_heatmap, coor[i], vis[i]

                # 重采样：从2*seq_len-1帧中随机选择seq_len帧
                rand_id = np.random.choice(len(frames), self.seq_len, replace=False)
                rand_id = np.sort(rand_id)
                tmp_coor = tmp_coor[rand_id]
                tmp_vis = tmp_vis[rand_id]
                frames = frames[rand_id]
                heatmaps = heatmaps[rand_id]

                if self.bg_mode == 'concat':
                    median_img = Image.fromarray(median_img.astype('uint8'))
                    median_img = np.array(median_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    median_img = np.moveaxis(median_img, -1, 0)
                    frames = np.concatenate((median_img.reshape(1, 3, self.HEIGHT, self.WIDTH), frames), axis=0)

                # 调整维度顺序以匹配网络输入
                frames = frames.reshape(-1, self.HEIGHT, self.WIDTH)

                frames /= 255.
                tmp_coor[:, 0] = tmp_coor[:, 0] / w
                tmp_coor[:, 1] = tmp_coor[:, 1] / h

                return data_idx, frames, heatmaps, tmp_coor, tmp_vis

            else:
                # 标准heatmap模式（无mixup增强）
                data_idx = self.data_dict['id'][idx]
                frame_file = self.data_dict['frame_file'][idx]
                coor = self.data_dict['coor'][idx]
                vis = self.data_dict['vis'][idx]
                w, h = self.img_config['img_shape'][data_idx[0][0]]
                w_scaler, h_scaler = self.img_config['img_scaler'][data_idx[0][0]]

                # 加载背景图（用于背景减除）
                if self.bg_mode:
                    file_format_str = os.path.join('{}', 'frame', '{}', '{}.' + IMG_FORMAT)
                    match_dir, rally_id, _ = parse.parse(file_format_str, frame_file[0])
                    median_file = os.path.join(match_dir, 'median.npz')
                    if not os.path.exists(median_file):
                        median_file = os.path.join(match_dir, 'frame', rally_id, 'median.npz')
                    assert os.path.exists(median_file)
                    median_img = np.load(median_file)['median']

                frames = np.array([]).reshape(0, self.HEIGHT, self.WIDTH)
                heatmaps = np.array([]).reshape(0, self.HEIGHT, self.WIDTH)

                # 逐帧处理
                for i in range(self.seq_len):
                    img = Image.open(frame_file[i])
                    if self.bg_mode == 'subtract':
                        img = Image.fromarray(np.sum(np.absolute(img - median_img), 2).astype('uint8'))
                        img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                        img = img.reshape(1, self.HEIGHT, self.WIDTH)
                    elif self.bg_mode == 'subtract_concat':
                        diff_img = Image.fromarray(np.sum(np.absolute(img - median_img), 2).astype('uint8'))
                        diff_img = np.array(diff_img.resize(size=(self.WIDTH, self.HEIGHT)))
                        diff_img = diff_img.reshape(1, self.HEIGHT, self.WIDTH)
                        img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                        img = np.moveaxis(img, -1, 0)
                        img = np.concatenate((img, diff_img), axis=0)
                    else:
                        img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                        img = np.moveaxis(img, -1, 0)

                    # 生成当前帧的热力图标签（将原图坐标映射到网络输入尺寸）
                    heatmap = self._get_heatmap(int(coor[i][0] / w_scaler), int(coor[i][1] / h_scaler))
                    frames = np.concatenate((frames, img), axis=0)
                    heatmaps = np.concatenate((heatmaps, heatmap), axis=0)

                if self.bg_mode == 'concat':
                    frames = np.concatenate((median_img, frames), axis=0)

                frames /= 255.
                coor[:, 0] = coor[:, 0] / w
                coor[:, 1] = coor[:, 1] / h

                return data_idx, frames, heatmaps, coor, vis

        elif self.data_mode == 'coordinate':
            # InpaintNet模式：返回坐标和修复掩码
            data_idx = self.data_dict['id'][idx]
            coor = self.data_dict['coor'][idx]  # 真实坐标 (seq_len, 2)
            coor_pred = self.data_dict['coor_pred'][idx]  # 预测坐标 (seq_len, 2)
            vis = self.data_dict['vis'][idx]  # 真实可见性 (seq_len,)
            vis_pred = self.data_dict['pred_vis'][idx]  # 预测可见性 (seq_len,)
            inpaint = self.data_dict['inpaint_mask'][idx]  # 修复掩码 (seq_len,)

            # 坐标归一化到网络输入尺寸（相对于WIDTH/HEIGHT的比例）
            coor[:, 0] = coor[:, 0] / self.WIDTH
            coor[:, 1] = coor[:, 1] / self.HEIGHT
            coor_pred[:, 0] = coor_pred[:, 0] / self.WIDTH
            coor_pred[:, 1] = coor_pred[:, 1] / self.HEIGHT

            return data_idx, coor_pred, coor, vis_pred.reshape(-1, 1), vis.reshape(-1, 1), inpaint.reshape(-1, 1)
        else:
            raise NotImplementedError


class Video_IterableDataset(IterableDataset):
    """
    视频流式数据集（用于大视频推理）
    使用IterableDataset避免一次性加载整个视频到内存
    支持滑动窗口生成序列
    """

    def __init__(self,
                 video_file,  # 视频文件路径
                 seq_len=8,  # 输入序列长度
                 sliding_step=1,  # 滑动步长
                 bg_mode='',  # 背景处理模式（同主数据集）
                 HEIGHT=HEIGHT,
                 WIDTH=WIDTH,
                 max_sample_num=1800,  # 生成中值背景图时的最大采样帧数
                 video_range=None,  # 生成背景图的视频时间范围（秒）
                 median=None  # 预计算的中值背景图
                 ):
        """
        初始化视频流式数据集

        Args:
            video_file: 视频文件路径
            seq_len: 输入序列长度
            sliding_step: 滑动窗口步长
            bg_mode: 背景处理模式
            HEIGHT/WIDTH: 网络输入尺寸
            max_sample_num: 生成中值图的最大采样数（避免内存溢出）
            video_range: 元组(start_sec, end_sec)，指定用于生成背景图的视频片段
            median: 预计算的中值背景图（numpy数组）
        """
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH

        self.video_file = video_file
        self.cap = cv2.VideoCapture(self.video_file)
        self.video_len = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.w, self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 计算原图到网络输入的缩放因子（用于后续坐标映射）
        self.w_scaler, self.h_scaler = self.w / self.WIDTH, self.h / self.HEIGHT

        self.seq_len = seq_len
        self.sliding_step = sliding_step
        self.bg_mode = bg_mode

        # 如果需要背景减除，生成或使用中值背景图
        if self.bg_mode:
            self.median = median if median is not None else self.__gen_median__(max_sample_num, video_range)

    def __iter__(self):
        """
        迭代生成数据序列（流式处理）
        使用滑动窗口从视频流中实时读取帧
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到视频开头
        success = True
        start_f_id, end_f_id = 0, 0
        frame_list = []

        while success:
            # 采样帧直到填满一个序列
            while len(frame_list) < self.seq_len:
                success, frame = self.cap.read()
                if not success:
                    break
                frame_list.append(frame)
                end_f_id += 1

            # 构建数据索引（回合0，帧号范围）
            data_idx = [(0, i) for i in range(start_f_id, end_f_id)]

            # 处理序列不足的情况：用最后一帧填充
            if len(data_idx) < self.seq_len:
                data_idx.extend([(0, end_f_id - 1)] * (self.seq_len - len(data_idx)))
                frame_list.extend([frame_list[-1]] * (self.seq_len - len(frame_list)))

            data_idx = np.array(data_idx)
            # BGR转RGB（OpenCV默认BGR，PIL使用RGB）
            frames = self.__process__(np.array(frame_list)[..., ::-1])
            yield data_idx, frames

            # 滑动窗口更新：移除前sliding_step帧
            frame_list = frame_list[self.sliding_step:]
            start_f_id = start_f_id + self.sliding_step

        self.cap.release()

    def __gen_median__(self, max_sample_num, video_range):
        """
        生成中值背景图（用于背景减除）
        从视频片段中均匀采样帧，计算像素级中值

        Args:
            max_sample_num: 最大采样帧数（控制内存使用）
            video_range: 时间范围（秒），None表示整个视频
        Returns:
            median_img: 中值背景图（RGB格式）
        """
        print('Generate median image...')

        # 确定采样帧范围
        if video_range is None:
            start_frame, end_frame = 0, self.video_len
        else:
            start_frame = max(0, video_range[0] * self.fps)
            end_frame = min(video_range[1] * self.fps, self.video_len)

        video_seg_len = end_frame - start_frame

        # 计算采样步长（均匀采样）
        if video_seg_len > max_sample_num:
            sample_step = video_seg_len // max_sample_num
        else:
            sample_step = 1

        frame_list = []
        for i in range(start_frame, end_frame, sample_step):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = self.cap.read()
            if not success:
                break
            frame_list.append(frame)

        # 计算中值（沿时间轴），并BGR转RGB
        median = np.median(frame_list, 0)[..., ::-1]

        # concat模式下需要调整维度
        if self.bg_mode == 'concat':
            median = Image.fromarray(median.astype('uint8'))
            median = np.array(median.resize(size=(self.WIDTH, self.HEIGHT)))
            median = np.moveaxis(median, -1, 0)

        print('Median image generated.')
        return median

    def __process__(self, imgs):
        """
        处理帧序列：调整尺寸、背景减除、维度重排、归一化
        同主数据集的帧处理逻辑

        Args:
            imgs: numpy数组，形状(seq_len, H, W, 3)，RGB格式
        Returns:
            frames: 处理后的帧序列，根据bg_mode可能是3通道或4通道
        """
        if self.bg_mode:
            median_img = self.median

        frames = np.array([]).reshape(0, self.HEIGHT, self.WIDTH)

        for i in range(self.seq_len):
            img = Image.fromarray(imgs[i])

            if self.bg_mode == 'subtract':
                # 灰度差分图
                img = Image.fromarray(np.sum(np.absolute(img - median_img), 2).astype('uint8'))
                img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                img = img.reshape(1, self.HEIGHT, self.WIDTH)
            elif self.bg_mode == 'subtract_concat':
                # RGB + 差分
                diff_img = Image.fromarray(np.sum(np.absolute(img - median_img), 2).astype('uint8'))
                diff_img = np.array(diff_img.resize(size=(self.WIDTH, self.HEIGHT)))
                diff_img = diff_img.reshape(1, self.HEIGHT, self.WIDTH)
                img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                img = np.moveaxis(img, -1, 0)
                img = np.concatenate((img, diff_img), axis=0)
            else:
                img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                img = np.moveaxis(img, -1, 0)

            frames = np.concatenate((frames, img), axis=0)

        if self.bg_mode == 'concat':
            frames = np.concatenate((median_img, frames), axis=0)

        frames /= 255.  # 归一化
        return frames