import os
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .dataset_tmp import Shuttlecock_Trajectory_Dataset, data_dir
from .general_tmp import *
from .metric import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 预测类型映射：TP（真正例）、TN（真负例）、FP1（定位不准）、FP2（误检）、FN（漏检）
pred_types = ['TP', 'TN', 'FP1', 'FP2', 'FN']
pred_types_map = {pred_type: i for i, pred_type in enumerate(pred_types)}
# InpaintNet三种评估模式：修复模式、重建模式、基线模式
inpaintnet_eval_types = ['inpaint', 'reconstruct', 'baseline']


def get_ensemble_weight(seq_len, eval_mode):
    """ 获取时序集成的权重

    在overlap评估模式下，同一帧被多个滑动窗口预测，通过加权平均提高稳定性

    Args:
        seq_len (int): 输入序列长度（窗口大小）
        eval_mode (str): 集成模式
            - 'average': 均匀权重，所有窗口平均
            - 'weight': 位置加权，中心窗口权重最高（金字塔形分布）

    Returns:
        weight (torch.Tensor): 归一化权重张量，形状(seq_len,)
    """

    if eval_mode == 'average':
        # 均匀权重：简单平均，适用于对时序一致性要求不高的场景
        weight = torch.ones(seq_len) / seq_len
    elif eval_mode == 'weight':
        # 位置加权：中心帧（最可靠）权重最高，边缘帧权重较低
        # 例如seq_len=8时，权重分布为[1,2,3,4,4,3,2,1]/20
        weight = torch.ones(seq_len)
        for i in range(math.ceil(seq_len / 2)):
            weight[i] = (i + 1)  # 前半段递增
            weight[seq_len - i - 1] = (i + 1)  # 后半段对称递增
        weight = weight / weight.sum()  # 归一化
    else:
        raise ValueError('Invalid mode')

    return weight


def predict_location(heatmap):
    """ 从单帧热力图中提取羽毛球坐标（连通区域中心）

    算法流程：
    1. 查找所有连通轮廓
    2. 选择最大面积的轮廓作为预测结果（假设羽毛球是画面中最显著的响应）
    3. 返回该轮廓的边界框

    Args:
        heatmap (numpy.ndarray): 单帧热力图，形状(H, W)

    Returns:
        tuple: (x, y, w, h) 最大连通区域的边界框坐标和尺寸
            如果热力图全零（无响应），返回(0,0,0,0)表示无球
    """
    if np.amax(heatmap) == 0:
        # 热力图无响应（全黑），判定该帧无球
        return 0, 0, 0, 0
    else:
        # 查找所有白色连通区域（轮廓）
        # cv2.RETR_EXTERNAL: 只检测外层轮廓
        # cv2.CHAIN_APPROX_SIMPLE: 压缩水平/垂直/对角线段，保留端点
        (cnts, _) = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(ctr) for ctr in cnts]

        # 选择最大面积的轮廓作为预测
        max_area_idx = 0
        max_area = rects[0][2] * rects[0][3]
        for i in range(1, len(rects)):
            area = rects[i][2] * rects[i][3]
            if area > max_area:
                max_area_idx = i
                max_area = area
        x, y, w, h = rects[max_area_idx]

        return x, y, w, h


def evaluate(indices, y_true=None, y_pred=None, c_true=None, c_pred=None, tolerance=4., img_scaler=(1, 1),
             output_bbox=False, output_gt=False):
    """ 核心评估函数：计算单帧预测与真实值的匹配关系（TP/TN/FP/FN）

    支持两种输入模式（互斥）：
    1. 热力图模式（y_true, y_pred）：TrackNet输出，需从热力图提取坐标
    2. 坐标模式（c_true, c_pred）：InpaintNet输出，直接使用坐标值

    混淆矩阵定义（5分类）：
    - TP (True Positive): 正确检测到球，且坐标与GT距离<tolerance
    - TN (True Negative): 正确判定为无球帧（预测和GT都不可见）
    - FP1 (False Positive Type 1): 检测到球但位置偏差过大（>tolerance，通常是鬼影/错位）
    - FP2 (False Positive Type 2): 背景误检为球（GT不可见但预测可见，通常是噪声/干扰物）
    - FN (False Negative): 漏检（GT可见但预测不可见，通常是遮挡/运动模糊）

    Args:
        indices (torch.Tensor): 帧索引，形状(N, L, 2)，用于去重和定位
        y_true/y_pred (torch.Tensor, optional): 真实/预测热力图，形状(N, L, H, W)
        c_true/c_pred (torch.Tensor, optional): 真实/预测坐标，形状(N, L, 2)，值范围[0,1]（归一化）
        tolerance (float): FP1判定阈值（像素距离），默认4个像素（输入尺寸下）
        img_scaler (tuple): 图像缩放因子(w_scaler, h_scaler)，用于坐标映射回原图
        output_bbox (bool): 是否输出检测框（用于COCO评估）
        output_gt (bool): 是否输出真实值坐标（用于详细分析）

    Returns:
        pred_dict (dict): 预测结果字典，包含：
            - 'Frame': 帧号列表
            - 'X'/'Y': 预测坐标（已反缩放）
            - 'Visibility': 预测可见性（0/1）
            - 'Type': 预测类型编码（0-4对应TP/TN/FP1/FP2/FN）
            - 'BBox'/'Confidence': 检测框和置信度（output_bbox=True时）
            - 'X_GT'/'Y_GT'/'Visibility_GT': 真实值（output_gt=True时）
    """

    pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': [], 'Type': [],
                 'BBox': [], 'Confidence': [], 'X_GT': [], 'Y_GT': [], 'Visibility_GT': []}

    batch_size, seq_len = indices.shape[0], indices.shape[1]
    indices = indices.detach().cpu().numpy().tolist() if torch.is_tensor(indices) else indices.numpy().tolist()

    # 模式1：热力图输入（TrackNet）
    if y_true is not None and y_pred is not None:
        assert c_true is None and c_pred is None, 'Invalid input'
        # 转换为numpy并调整格式为(N, L, H, W)
        y_true = y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else y_true
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_true = to_img_format(y_true)  # (N, L, H, W)
        y_pred = to_img_format(y_pred)  # (N, L, H, W)
        h_pred = y_pred > 0.5  # 二值化：>0.5为1（球区域），否则为0

    # 模式2：坐标输入（InpaintNet）
    if c_true is not None and c_pred is not None:
        assert y_true is None and y_pred is None, 'Invalid input'
        assert output_bbox == False, 'Coordinate prediction cannot output detection'
        c_true = c_true.detach().cpu().numpy() if torch.is_tensor(c_true) else c_true
        c_pred = c_pred.detach().cpu().numpy() if torch.is_tensor(c_pred) else c_pred
        # 反归一化：乘以网络输入尺寸得到绝对坐标
        c_true[..., 0] = c_true[..., 0] * WIDTH
        c_true[..., 1] = c_true[..., 1] * HEIGHT
        c_pred[..., 0] = c_pred[..., 0] * WIDTH
        c_pred[..., 1] = c_pred[..., 1] * HEIGHT

    for n in range(batch_size):
        prev_d_i = [-1, -1]  # 记录上一帧索引，用于去重（滑动窗口重叠会导致重复）
        for f in range(seq_len):
            d_i = indices[n][f]
            if d_i != prev_d_i:  # 跳过重复帧（同一帧在overlap模式下出现多次）
                if c_true is not None and c_pred is not None:
                    # 坐标模式评估逻辑
                    c_t = c_true[n][f]
                    c_p = c_pred[n][f]
                    cx_true, cy_true = int(c_t[0]), int(c_t[1])
                    cx_pred, cy_pred = int(c_p[0]), int(c_p[1])
                    vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                    if np.amax(c_p) == 0 and np.amax(c_t) == 0:
                        # 都无球 -> TN
                        pred_dict['Type'].append(pred_types_map['TN'])
                    elif np.amax(c_p) > 0 and np.amax(c_t) == 0:
                        # 预测有球但真实无球 -> FP2（误检）
                        pred_dict['Type'].append(pred_types_map['FP2'])
                    elif np.amax(c_p) == 0 and np.amax(c_t) > 0:
                        # 预测无球但真实有球 -> FN（漏检）
                        pred_dict['Type'].append(pred_types_map['FN'])
                    elif np.amax(c_p) > 0 and np.amax(c_t) > 0:
                        # 都有球，计算距离判断准确性
                        dist = math.sqrt(pow(cx_pred - cx_true, 2) + pow(cy_pred - cy_true, 2))
                        if dist > tolerance:
                            pred_dict['Type'].append(pred_types_map['FP1'])  # 位置偏差大
                        else:
                            pred_dict['Type'].append(pred_types_map['TP'])  # 正确检测
                    else:
                        raise ValueError(f'Invalid input: {c_p}, {c_t}')
                elif y_true is not None and y_pred is not None:
                    # 热力图模式评估逻辑
                    y_t = y_true[n][f]
                    y_p = y_pred[n][f]
                    h_p = h_pred[n][f]
                    # 从热力图提取坐标（连通区域中心）
                    bbox_true = predict_location(to_img(y_t))
                    cx_true, cy_true = int(bbox_true[0] + bbox_true[2] / 2), int(bbox_true[1] + bbox_true[3] / 2)
                    bbox_pred = predict_location(to_img(h_p))
                    cx_pred, cy_pred = int(bbox_pred[0] + bbox_pred[2] / 2), int(bbox_pred[1] + bbox_pred[3] / 2)
                    # 计算置信度（预测热力图在检测框区域内的最大值）
                    if np.amax(bbox_pred) > 0:
                        conf = np.amax(
                            y_p[bbox_pred[1]:bbox_pred[1] + bbox_pred[3], bbox_pred[0]:bbox_pred[0] + bbox_pred[2]])
                    else:
                        conf = 0.
                    vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                    if np.amax(h_p) == 0 and np.amax(y_t) == 0:
                        # 都无球 -> TN
                        pred_dict['Type'].append(pred_types_map['TN'])
                    elif np.amax(h_p) > 0 and np.amax(y_t) == 0:
                        # 预测有球但真实无球 -> FP2（误检）
                        pred_dict['Type'].append(pred_types_map['FP2'])
                    elif np.amax(h_p) == 0 and np.amax(y_t) > 0:
                        # 预测无球但真实有球 -> FN（漏检）
                        pred_dict['Type'].append(pred_types_map['FN'])
                    elif np.amax(h_p) > 0 and np.amax(y_t) > 0:
                        # 都有球，计算距离判断准确性
                        dist = math.sqrt(pow(cx_pred - cx_true, 2) + pow(cy_pred - cy_true, 2))
                        if dist > tolerance:
                            pred_dict['Type'].append(pred_types_map['FP1'])  # 位置偏差大
                        else:
                            pred_dict['Type'].append(pred_types_map['TP'])  # 正确检测
                    else:
                        raise ValueError('Invalid input')
                else:
                    raise ValueError('Invalid input')

                # 记录预测结果（坐标反缩放到原图尺寸）
                pred_dict['Frame'].append(int(d_i[1]))
                pred_dict['X'].append(int(cx_pred * img_scaler[0]))
                pred_dict['Y'].append(int(cy_pred * img_scaler[1]))
                pred_dict['Visibility'].append(vis_pred)

                if output_bbox:
                    pred_dict['BBox'].append([
                        int(bbox_pred[0] * img_scaler[0]),
                        int(bbox_pred[1] * img_scaler[1]),
                        int(bbox_pred[2] * img_scaler[0]),
                        int(bbox_pred[3] * img_scaler[1])
                    ])
                    pred_dict['Confidence'].append(float(conf))

                if output_gt:
                    vis_gt = 0 if cx_true == 0 and cy_true == 0 else 1
                    pred_dict['X_GT'].append(int(cx_true * img_scaler[0]))
                    pred_dict['Y_GT'].append(int(cy_true * img_scaler[1]))
                    pred_dict['Visibility_GT'].append(vis_gt)

                prev_d_i = d_i
            else:
                break  # 遇到重复帧立即跳出（batch内后续帧都会重复）

    # 清理未使用的字段（根据配置）
    if not output_bbox:
        del pred_dict['BBox']
        del pred_dict['Confidence']

    if not output_gt:
        del pred_dict['X_GT']
        del pred_dict['Y_GT']
        del pred_dict['Visibility_GT']

    return pred_dict


def generate_inpaint_mask(pred_dict, th_h=30):
    """ 根据TrackNet预测结果生成修复掩码（标记需要InpaintNet修复的帧）

    修复策略（针对羽毛球被遮挡或快速运动导致的轨迹缺失）：
    1. 遍历轨迹，找到所有连续不可见的片段（vis=0的连续帧）
    2. 判断该片段前后是否有有效的轨迹点（y > th_h，确保球在场内而非出界）
    3. 如果前后都有有效点，则标记该片段为需要修复（mask=1）
    4. 如果片段在开头或结尾，或前后有点在地面以下（出界），则不修复（可能是球出画面）

    Args:
        pred_dict (dict): TrackNet预测结果，包含'Y'坐标和'Visibility'
        th_h (float): Y坐标阈值（像素），低于此值视为出界或无效

    Returns:
        inpaint_mask (list): 修复掩码列表（1表示需要修复，0表示无需修复）
    """
    y = np.array(pred_dict['Y'])
    vis_pred = np.array(pred_dict['Visibility'])
    inpaint_mask = np.zeros_like(y)
    i = 0  # 不可见片段起始索引
    j = 0  # 不可见片段结束索引
    threshold = th_h
    while j < len(vis_pred):
        # 寻找可见->不可见的转换点（消失起点）
        while i < len(vis_pred) - 1 and vis_pred[i] == 1:
            i += 1
        j = i
        # 寻找不可见->可见的转换点（重新出现点）
        while j < len(vis_pred) - 1 and vis_pred[j] == 0:
            j += 1
        if j == i:
            break  # 没有更多不可见片段
        elif i == 0 and y[j] > threshold:
            # 特殊情况：从第一帧就开始消失，但重新出现时有效（可能是视频截断）
            inpaint_mask[:j] = 1
        elif (i > 1 and y[i - 1] > threshold) and (j < len(vis_pred) and y[j] > threshold):
            # 标准情况：前后都有有效轨迹点，中间消失段需要修复（遮挡/模糊）
            inpaint_mask[i:j] = 1
        else:
            # 球不在画面内（出界），无需修复（保持不可见状态）
            pass
        i = j  # 继续寻找下一个片段

    return inpaint_mask.tolist()


def linear_interp(target, inpaint_mask):
    """ 线性插值修复：对标记为需要修复的帧进行线性插值填充

    算法：
    对于每个标记为1（inpaint_mask=1）的连续片段，用片段起点前一帧和终点后一帧的值进行线性插值填充

    Args:
        target (list): 目标值序列（如X或Y坐标）
        inpaint_mask (list): 修复掩码（1表示需要插值填充）

    Returns:
        target (numpy.ndarray): 修复后的目标序列
    """
    assert len(target) == len(inpaint_mask), 'Length of target and inpaint_mask should be the same'
    target = np.array(target)
    inpaint_mask = np.array(inpaint_mask)
    i = 0  # 待修复片段起始
    j = 0  # 待修复片段结束

    while j < len(inpaint_mask):
        # 跳过无需修复的帧
        while i < len(inpaint_mask) - 1 and inpaint_mask[i] == 0:
            i += 1
        j = i
        # 寻找修复片段终点
        while j < len(inpaint_mask) - 1 and inpaint_mask[j] == 1:
            j += 1
        if j == i:
            break  # 没有更多修复片段
        else:
            # 线性插值参数
            x = np.linspace(0, 1, len(inpaint_mask[i:j]))  # 插值位置比例
            xp = [0, 1]  # 已知点位置

            # 边界情况处理：确定已知点值
            if i == 0:
                # 片段在开头，用终点值填充（无起点参考）
                fp = [target[j], target[j]]
            elif j == len(inpaint_mask) - 1:
                # 片段在结尾，用起点值填充（无终点参考）
                fp = [target[i - 1], target[i - 1]]
            else:
                # 标准情况：用前后帧值插值
                fp = [target[i - 1], target[j]]

            # 执行线性插值
            target[i:j] = np.interp(x, xp, fp)
        i = j  # 继续处理下一个片段

    return target


# 仅用于训练评估，不保存结果
def get_eval_res(pred_dict):
    """ 从预测结果字典解析混淆矩阵统计量

    Args:
        pred_dict (dict): 包含'Type'字段的预测结果

    Returns:
        res (numpy.ndarray): 5维统计向量 [TP, TN, FP1, FP2, FN]
    """

    type_res = np.array(pred_dict['Type'])
    res = np.zeros(5)
    for pred_type in pred_types:
        res[pred_types_map[pred_type]] += int((type_res == pred_types_map[pred_type]).sum())

    return res


def eval_tracknet(model, data_loader, param_dict):
    """ TrackNet模型评估（验证集/测试集）

    流程：
    1. 遍历数据加载器，前向传播获取预测热力图
    2. 计算WBCE损失
    3. 调用evaluate计算混淆矩阵（TP/TN/FP/FN）
    4. 汇总统计并计算准确率、精确率、召回率、F1、漏检率

    Args:
        model (nn.Module): TrackNet模型
        data_loader (DataLoader): 评估数据加载器
        param_dict (dict): 参数字典，包含verbose（是否显示进度）、tolerance（容差阈值）

    Returns:
        avg_loss (float): 平均WBCE损失
        res_dict (dict): 包含TP/TN/FP1/FP2/FN及派生指标的字典
    """

    model.eval()
    losses = []
    confusion_matrix = np.zeros(5)  # TP, TN, FP1, FP2, FN
    if param_dict['verbose']:
        data_prob = tqdm(data_loader)
    else:
        data_prob = data_loader

    for step, (i, x, y, _, _) in enumerate(data_prob):
        x, y = x.float().to(DEVICE), y.float().to(DEVICE)
        with torch.no_grad():
            y_pred = model(x)

        loss = WBCELoss(y_pred, y)
        losses.append(loss.item())

        # 评估当前batch，累加混淆矩阵
        pred_dict = evaluate(i, y_true=y, y_pred=y_pred, tolerance=param_dict['tolerance'])
        confusion_matrix += get_eval_res(pred_dict)

        if param_dict['verbose']:
            TP, TN, FP1, FP2, FN = confusion_matrix
            data_prob.set_description(f'Evaluation')
            data_prob.set_postfix(TP=TP, TN=TN, FP1=FP1, FP2=FP2, FN=FN)

    # 计算评估指标
    TP, TN, FP1, FP2, FN = confusion_matrix
    accuracy, precision, recall, f1, miss_rate = get_metric(TP, TN, FP1, FP2, FN)
    res_dict = {'TP': TP, 'TN': TN,
                'FP1': FP1, 'FP2': FP2, 'FN': FN,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'miss_rate': miss_rate}

    return float(np.mean(losses)), res_dict


def eval_inpaintnet(model, data_loader, param_dict):
    """ InpaintNet模型评估

    支持三种评估模式：
    1. 'inpaint': 与真实值对比（标准评估）
    2. 'reconstruct': 与TrackNet预测值对比（验证修复是否过度偏离原预测）
    3. 'baseline': TrackNet原预测与真实值对比（基线对比）

    Args:
        model (nn.Module): InpaintNet模型
        data_loader (DataLoader): 数据加载器（coordinate模式）
        param_dict (dict): 参数字典

    Returns:
        avg_loss (float): 平均MSE损失
        res_dict (dict): 三种评估模式的指标字典
    """

    model.eval()
    losses = []
    # 分别为三种模式维护混淆矩阵
    confusion_matrix = {eval_type: np.zeros(5) for eval_type in inpaintnet_eval_types}  # TP, TN, FP1, FP2, FN
    if param_dict['verbose']:
        data_prob = tqdm(data_loader)
    else:
        data_prob = data_loader

    for step, (i, coor_pred, coor, _, _, inpaint_mask) in enumerate(data_prob):
        coor_pred, coor, inpaint_mask = (
            coor_pred.float().to(DEVICE),
            coor.float().to(DEVICE),
            inpaint_mask.float().to(DEVICE)
        )

        with torch.no_grad():
            # InpaintNet修复：输入TrackNet预测+掩码，输出修复后坐标
            coor_inpaint = model(coor_pred, inpaint_mask)
            # 掩码融合：只修改标记为需要修复的帧
            coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1 - inpaint_mask)

            loss = nn.MSELoss()(coor_inpaint * inpaint_mask, coor * inpaint_mask)
            losses.append(loss.item())

            # 阈值处理：过小坐标视为0（不可见）
            th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
            coor_inpaint[th_mask] = 0.

        # 三种模式评估
        for eval_type in inpaintnet_eval_types:
            if eval_type == 'inpaint':
                # 修复结果 vs 真实值（最终评估标准）
                pred_dict = evaluate(i, c_true=coor, c_pred=coor_inpaint, tolerance=param_dict['tolerance'])
            elif eval_type == 'reconstruct':
                # 修复结果 vs TrackNet原预测（检查修复是否偏离太远）
                pred_dict = evaluate(i, c_true=coor_pred, c_pred=coor_inpaint, tolerance=param_dict['tolerance'])
            elif eval_type == 'baseline':
                # TrackNet原预测 vs 真实值（基线对比，看修复是否有提升）
                pred_dict = evaluate(i, c_true=coor, c_pred=coor_pred, tolerance=param_dict['tolerance'])
            else:
                raise ValueError('Invalid eval_type')
            confusion_matrix[eval_type] += get_eval_res(pred_dict)

        if param_dict['verbose']:
            TP, TN, FP1, FP2, FN = confusion_matrix['inpaint']
            data_prob.set_description(f'Evaluation')
            data_prob.set_postfix(TP=TP, TN=TN, FP1=FP1, FP2=FP2, FN=FN)

    # 整理三种模式的指标
    res_dict = {}
    for eval_type in inpaintnet_eval_types:
        TP, TN, FP1, FP2, FN = confusion_matrix[eval_type]
        accuracy, precision, recall, f1, miss_rate = get_metric(TP, TN, FP1, FP2, FN)
        res_dict[eval_type] = {'TP': TP, 'TN': TN,
                               'FP1': FP1, 'FP2': FP2, 'FN': FN,
                               'accuracy': accuracy,
                               'precision': precision,
                               'recall': recall,
                               'f1': f1,
                               'miss_rate': miss_rate}

    return float(np.mean(losses)), res_dict


# 用于测试集评估（COCO格式和标准指标）
def get_coco_res(pred_dict, drop=False):
    """ 将预测结果转换为COCO格式（用于标准目标检测评估mAP）

    COCO格式要求：
    - image_id: 图像ID（全局唯一）
    - category_id: 类别ID（羽毛球=1）
    - bbox: [x, y, w, h] 边界框（左上角坐标+宽高）
    - score: 置信度（用于计算PR曲线）
    - area: 边界框面积

    Args:
        pred_dict (dict): 多回合预测结果字典，键为'{match_id}_{rally_id}'
        drop (bool): 是否丢弃drop_frame.json指定的帧范围（用于处理测试集特殊分段）

    Returns:
        res_list (list): COCO格式的检测结果列表
    """
    sample_count = 0
    res_list = []
    for rally_key, pred in pred_dict.items():
        # 如需裁剪帧范围（测试集特殊处理）
        if drop:
            drop_frame_dict = json.load(open(os.path.join(data_dir, 'drop_frame.json')))
            start_f, end_f = drop_frame_dict['start'], drop_frame_dict['end']
            for key in pred.keys():
                pred[key] = pred[key][start_f[rally_key]:end_f[rally_key]]

        # 逐帧转换为COCO格式（只包含有球的帧，Visibility>0）
        for i in range(len(pred['Frame'])):
            if pred['Visibility'][i] > 0:
                res_list.append({
                    'id': sample_count,
                    'image_id': sample_count,
                    'category_id': 1,  # 羽毛球类别
                    'bbox': pred['BBox'][i],  # [x, y, w, h]
                    'score': pred['Confidence'][i],  # 检测置信度
                    'ignore': 0,
                    'area': pred['BBox'][i][2] * pred['BBox'][i][3],  # w*h
                    'segmentation': [],
                    'iscrowd': 0
                })
            sample_count += 1

    return res_list


def get_test_res(pred_dict, drop=False):
    """ 计算测试集的整体评估指标（汇总多回合结果）

    Args:
        pred_dict (dict): 多回合预测结果
        drop (bool): 是否应用drop_frame裁剪

    Returns:
        res_dict (dict): 整体指标（accuracy, precision, recall, f1, miss_rate）
    """

    res_dict = {pred_type: 0 for pred_type in pred_types}
    for rally_key, pred in pred_dict.items():
        if drop:
            drop_frame_dict = json.load(open(os.path.join(data_dir, 'drop_frame.json')))
            start_f, end_f = drop_frame_dict['start'], drop_frame_dict['end']
            type_res = np.array(pred['Type'])[start_f[rally_key]:end_f[rally_key]]
        else:
            type_res = np.array(pred['Type'])

        # 累加各类别的计数
        for pred_type in pred_types:
            res_dict[pred_type] += int((type_res == pred_types_map[pred_type]).sum())

    # 计算综合指标
    TP, TN, FP1, FP2, FN = res_dict['TP'], res_dict['TN'], res_dict['FP1'], res_dict['FP2'], res_dict['FN']
    accuracy, precision, recall, f1, miss_rate = get_metric(TP, TN, FP1, FP2, FN)
    res_dict = {'TP': TP, 'TN': TN,
                'FP1': FP1, 'FP2': FP2, 'FN': FN,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'miss_rate': miss_rate}

    return res_dict


def test(model, split, param_dict, save_inpaint_mask=False, linear_interp=False):
    """ 在指定数据划分（train/val/test）上测试模型

    遍历该split下的所有回合（rally），逐个调用test_rally进行测试

    Args:
        model (tuple): (TrackNet, InpaintNet)模型对，InpaintNet可为None
        split (str): 数据划分
        param_dict (dict): 参数字典
        save_inpaint_mask (bool): 是否保存修复掩码到CSV（用于InpaintNet训练数据准备）
        linear_interp (bool): 是否使用线性插值而非InpaintNet进行修复

    Returns:
        pred_dict (dict): 各回合的预测结果字典，键为'{match_id}_{rally_id}'
    """

    # 基于回合的测试
    pred_dict = {}
    rally_dirs = get_rally_dirs(data_dir, split)
    rally_dirs = [os.path.join(data_dir, rally_dir) for rally_dir in rally_dirs]
    if param_dict['debug']:
        rally_dirs = rally_dirs[:1]  # 调试模式只测第一个回合

    for rally_dir in rally_dirs:
        # 解析路径获取match_id和rally_id作为键
        file_format_str = os.path.join('{}', 'frame', '{}')
        match_dir, rally_id = parse.parse(file_format_str, rally_dir)
        match_id = match_dir.split('match')[-1]
        rally_key = f'{match_id}_{rally_id}'

        # 执行单回合测试
        if linear_interp:
            tmp_pred = test_rally_linear(model, rally_dir, param_dict)
        else:
            tmp_pred = test_rally(model, rally_dir, param_dict, save_inpaint_mask=save_inpaint_mask)
        pred_dict[rally_key] = tmp_pred

        # 如需保存修复掩码（用于后续InpaintNet训练）
        if save_inpaint_mask:
            pred_csv_dir = os.path.join(match_dir, 'predicted_csv')
            if not os.path.exists(pred_csv_dir):
                os.makedirs(pred_csv_dir)
            csv_file = os.path.join(pred_csv_dir, f'{rally_id}_ball.csv')
            write_pred_csv(tmp_pred, save_file=csv_file, save_inpaint_mask=save_inpaint_mask)

    return pred_dict


def test_rally(model, rally_dir, param_dict, save_inpaint_mask=False):
    """ 在单个回合（rally）上测试模型

    支持四种组合模式：
    1. TrackNet only（无InpaintNet）：直接输出TrackNet结果
    2. TrackNet + InpaintNet：TrackNet检测 + InpaintNet修复
    3. nonoverlap模式：快速但精度较低
    4. overlap+时序集成模式：高精度但较慢

    Args:
        model (tuple): (TrackNet, InpaintNet)
        rally_dir (str): 回合帧目录路径
        param_dict (dict): 包含seq_len, eval_mode, tolerance等参数
        save_inpaint_mask (bool): 是否返回修复掩码（用于训练数据生成）

    Returns:
        pred_dict (dict): 该回合的预测结果（包含坐标、可见性、类型、可选的检测框等）
    """

    tracknet, inpaintnet = model
    w, h = Image.open(os.path.join(rally_dir, '0.png')).size

    # 根据模式确定缩放方式
    if save_inpaint_mask:
        # 保存训练数据时保持原始尺寸（不缩放）
        w_scaler, h_scaler = 1., 1.
    else:
        w_scaler, h_scaler = w / WIDTH, h / HEIGHT

    # 模式A：仅TrackNet（无修复网络）
    if inpaintnet is None:
        tracknet.eval()
        seq_len = param_dict['tracknet_seq_len']
        # 初始化结果字典
        tracknet_pred_dict = {
            'Frame': [], 'X': [], 'Y': [], 'Visibility': [], 'Type': [],
            'BBox': [], 'Confidence': [], 'X_GT': [], 'Y_GT': [], 'Visibility_GT': []
        }

        if param_dict['eval_mode'] == 'nonoverlap':
            # A1: 非重叠滑动窗口（快速模式）
            # sliding_step = seq_len，相邻窗口无重叠，每帧仅被预测一次
            dataset = Shuttlecock_Trajectory_Dataset(
                seq_len=seq_len, sliding_step=seq_len, data_mode='heatmap',
                bg_mode=param_dict['bg_mode'], rally_dir=rally_dir, padding=True
            )
            data_loader = DataLoader(
                dataset, batch_size=param_dict['batch_size'],
                shuffle=False, num_workers=param_dict['num_workers'], drop_last=False
            )

            data_prob = tqdm(data_loader) if param_dict['verbose'] else data_loader
            for step, (i, x, y, _, _) in enumerate(data_prob):
                x = x.float().to(DEVICE)
                with torch.no_grad():
                    y_pred = tracknet(x).detach().cpu()

                # 评估并累加结果
                tmp_pred = evaluate(i, y_true=y, y_pred=y_pred,
                                    tolerance=param_dict['tolerance'],
                                    img_scaler=(w_scaler, h_scaler),
                                    output_bbox=param_dict['output_bbox'],
                                    output_gt=param_dict['output_gt'])
                for key in tmp_pred.keys():
                    tracknet_pred_dict[key].extend(tmp_pred[key])
        else:
            # A2: 重叠滑动窗口 + 时序集成（高精度模式）
            # sliding_step = 1，每帧被多个窗口覆盖，通过加权平均提高稳定性
            dataset = Shuttlecock_Trajectory_Dataset(
                seq_len=seq_len, sliding_step=1, data_mode='heatmap',
                bg_mode=param_dict['bg_mode'], rally_dir=rally_dir
            )
            data_loader = DataLoader(
                dataset, batch_size=param_dict['batch_size'],
                shuffle=False, num_workers=param_dict['num_workers'], drop_last=False
            )
            weight = get_ensemble_weight(seq_len, param_dict['eval_mode'])

            # 时序集成缓冲区初始化
            num_sample, sample_count = len(dataset), 0
            buffer_size = seq_len - 1
            batch_i = torch.arange(seq_len)  # [0,1,2,3,4,5,6,7]
            frame_i = torch.arange(seq_len - 1, -1, -1)  # [7,6,5,4,3,2,1,0] 逆序索引，用于对齐时序

            y_pred_buffer = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)

            data_prob = tqdm(data_loader) if param_dict['verbose'] else data_loader
            for step, (i, x, y, _, _) in enumerate(data_prob):
                x = x.float().to(DEVICE)
                b_size = i.shape[0]
                with torch.no_grad():
                    y_pred = tracknet(x).detach().cpu()

                # 将新预测加入缓冲区
                y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)

                # 集成容器
                ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
                ensemble_y = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)
                ensemble_y_pred = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)

                # 处理当前batch中的每个样本
                for b in range(b_size):
                    if sample_count < buffer_size:
                        # 缓冲区未满：简单平均（样本数较少时）
                        y_pred_avg = y_pred_buffer[batch_i + b, frame_i].sum(0) / (sample_count + 1)
                    else:
                        # 缓冲区已满：加权时序集成
                        y_pred_avg = (y_pred_buffer[batch_i + b, frame_i] * weight[:, None, None]).sum(0)

                    # 收集集成结果
                    ensemble_i = torch.cat((ensemble_i, i[b][0].reshape(1, 1, 2)), dim=0)
                    ensemble_y = torch.cat((ensemble_y, y[b][0].reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                    ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred_avg.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                    sample_count += 1

                    # 处理最后一批的尾部帧（边界处理）
                    if sample_count == num_sample:
                        y_zero_pad = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
                        y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)

                        for f in range(1, seq_len):
                            y_pred_avg = y_pred_buffer[batch_i + b + f, frame_i].sum(0) / (seq_len - f)
                            ensemble_i = torch.cat((ensemble_i, i[-1][f].reshape(1, 1, 2)), dim=0)
                            ensemble_y = torch.cat((ensemble_y, y[-1][f].reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                            ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred_avg.reshape(1, 1, HEIGHT, WIDTH)),
                                                        dim=0)

                # 评估集成后的结果
                tmp_pred = evaluate(ensemble_i, y_true=ensemble_y, y_pred=ensemble_y_pred,
                                    tolerance=param_dict['tolerance'],
                                    img_scaler=(w_scaler, h_scaler),
                                    output_bbox=param_dict['output_bbox'],
                                    output_gt=param_dict['output_gt'])
                for key in tmp_pred.keys():
                    tracknet_pred_dict[key].extend(tmp_pred[key])

                # 滑动缓冲区：保留最后buffer_size个预测用于下一次集成
                y_pred_buffer = y_pred_buffer[-buffer_size:]

        # 生成修复掩码（标记TrackNet预测失败的帧，供后续InpaintNet使用）
        tracknet_pred_dict['Inpaint_Mask'] = generate_inpaint_mask(tracknet_pred_dict, th_h=30)
        return tracknet_pred_dict

    # 模式B：TrackNetV3（TrackNet + InpaintNet）
    else:
        inpaintnet.eval()
        seq_len = param_dict['inpaintnet_seq_len']
        inpaintnet_pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': [], 'Type': []}

        if param_dict['eval_mode'] == 'nonoverlap':
            # B1: 非重叠模式（快速）
            dataset = Shuttlecock_Trajectory_Dataset(
                seq_len=seq_len, sliding_step=seq_len, data_mode='coordinate',
                rally_dir=rally_dir, padding=True
            )
            data_loader = DataLoader(
                dataset, batch_size=param_dict['batch_size'],
                shuffle=False, num_workers=param_dict['num_workers'], drop_last=False
            )

            data_prob = tqdm(data_loader) if param_dict['verbose'] else data_loader
            for step, (i, coor_pred, coor, _, _, inpaint_mask) in enumerate(data_prob):
                coor_pred, coor, inpaint_mask = coor_pred.float(), coor.float(), inpaint_mask.float()

                with torch.no_grad():
                    # InpaintNet修复：输入预测坐标+掩码，输出修复后坐标
                    coor_inpaint = inpaintnet(
                        coor_pred.to(DEVICE), inpaint_mask.to(DEVICE)
                    ).detach().cpu()
                    # 掩码融合：只修改需要修复的帧
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1 - inpaint_mask)

                # 阈值处理：过小坐标视为无效（不可见）
                th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
                coor_inpaint[th_mask] = 0.

                # 评估修复后的结果
                tmp_pred = evaluate(i, c_true=coor, c_pred=coor_inpaint,
                                    tolerance=param_dict['tolerance'], img_scaler=(w_scaler, h_scaler))
                for key in tmp_pred.keys():
                    inpaintnet_pred_dict[key].extend(tmp_pred[key])
        else:
            # B2: 重叠模式 + 时序集成（高精度）
            dataset = Shuttlecock_Trajectory_Dataset(
                seq_len=seq_len, sliding_step=1, data_mode='coordinate', rally_dir=rally_dir
            )
            data_loader = DataLoader(
                dataset, batch_size=param_dict['batch_size'],
                shuffle=False, num_workers=param_dict['num_workers'], drop_last=False
            )
            weight = get_ensemble_weight(seq_len, param_dict['eval_mode'])

            # 坐标缓冲区（不同于TrackNet的热力图缓冲区）
            num_sample, sample_count = len(dataset), 0
            buffer_size = seq_len - 1
            batch_i = torch.arange(seq_len)
            frame_i = torch.arange(seq_len - 1, -1, -1)
            coor_inpaint_buffer = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)

            data_prob = tqdm(data_loader) if param_dict['verbose'] else data_loader
            for step, (i, coor_pred, coor, _, _, inpaint_mask) in enumerate(data_prob):
                coor_pred, coor, inpaint_mask = coor_pred.float(), coor.float(), inpaint_mask.float()
                b_size = i.shape[0]

                with torch.no_grad():
                    coor_inpaint = inpaintnet(
                        coor_pred.to(DEVICE), inpaint_mask.to(DEVICE)
                    ).detach().cpu()
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1 - inpaint_mask)

                # 阈值处理
                th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
                coor_inpaint[th_mask] = 0.

                # 时序集成（与TrackNet类似，但对坐标直接平均）
                coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_inpaint), dim=0)
                ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
                ensemble_coor = torch.empty((0, 1, 2), dtype=torch.float32)
                ensemble_coor_inpaint = torch.empty((0, 1, 2), dtype=torch.float32)

                for b in range(b_size):
                    if sample_count < buffer_size:
                        coor_inpaint_avg = coor_inpaint_buffer[batch_i + b, frame_i].sum(0) / (sample_count + 1)
                    else:
                        coor_inpaint_avg = (coor_inpaint_buffer[batch_i + b, frame_i] * weight[:, None]).sum(0)

                    ensemble_i = torch.cat((ensemble_i, i[b][0].view(1, 1, 2)), dim=0)
                    ensemble_coor = torch.cat((ensemble_coor, coor[b][0].view(1, 1, 2)), dim=0)
                    ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint_avg.view(1, 1, 2)), dim=0)
                    sample_count += 1

                    if sample_count == num_sample:
                        # 尾部处理
                        coor_zero_pad = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
                        coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_zero_pad), dim=0)

                        for f in range(1, seq_len):
                            coor_inpaint_avg = coor_inpaint_buffer[batch_i + b + f, frame_i].sum(0) / (seq_len - f)
                            ensemble_i = torch.cat((ensemble_i, i[b][f].view(1, 1, 2)), dim=0)
                            ensemble_coor = torch.cat((ensemble_coor, coor[b][f].view(1, 1, 2)), dim=0)
                            ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint_avg.view(1, 1, 2)),
                                                              dim=0)

                # 阈值处理和评估
                th_mask = ((ensemble_coor_inpaint[:, :, 0] < COOR_TH) & (ensemble_coor_inpaint[:, :, 1] < COOR_TH))
                ensemble_coor_inpaint[th_mask] = 0.

                tmp_pred = evaluate(ensemble_i, c_true=ensemble_coor, c_pred=ensemble_coor_inpaint,
                                    tolerance=param_dict['tolerance'],
                                    img_scaler=(w_scaler, h_scaler))
                for key in tmp_pred.keys():
                    inpaintnet_pred_dict[key].extend(tmp_pred[key])

                # 滑动缓冲区
                coor_inpaint_buffer = coor_inpaint_buffer[-buffer_size:]

        return inpaintnet_pred_dict


def test_rally_linear(model, rally_dir, param_dict):
    """ 使用线性插值代替InpaintNet进行修复（基线对比方法）

    流程：
    1. 先用TrackNet预测轨迹（可能包含遮挡导致的缺失）
    2. 生成修复掩码（标记缺失段）
    3. 对X和Y坐标分别进行线性插值填充缺失段
    4. 评估插值修复后的轨迹

    用于对比InpaintNet相对于简单插值的优势
    """
    tracknet, _ = model
    w, h = Image.open(os.path.join(rally_dir, '0.png')).size
    w_scaler, h_scaler = w / WIDTH, h / HEIGHT

    # 第一步：TrackNet预测（与test_rally相同）
    tracknet.eval()
    seq_len = param_dict['tracknet_seq_len']
    tracknet_pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': [], 'Type': []}

    if param_dict['eval_mode'] == 'nonoverlap':
        # 非重叠滑动窗口
        dataset = Shuttlecock_Trajectory_Dataset(
            seq_len=seq_len, sliding_step=seq_len,
            data_mode='heatmap', bg_mode=param_dict['bg_mode'],
            rally_dir=rally_dir, padding=True
        )
        data_loader = DataLoader(
            dataset, batch_size=param_dict['batch_size'],
            shuffle=False, num_workers=param_dict['num_workers'], drop_last=False
        )

        data_prob = tqdm(data_loader) if param_dict['verbose'] else data_loader
        for step, (i, x, y, _, _) in enumerate(data_prob):
            x = x.float().to(DEVICE)
            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()

            tmp_pred = evaluate(i, y_true=y, y_pred=y_pred, tolerance=param_dict['tolerance'])
            for key in tmp_pred.keys():
                tracknet_pred_dict[key].extend(tmp_pred[key])
    else:
        # 重叠滑动窗口 + 时序集成
        dataset = Shuttlecock_Trajectory_Dataset(
            seq_len=seq_len, sliding_step=1,
            data_mode='heatmap', bg_mode=param_dict['bg_mode'],
            rally_dir=rally_dir
        )
        data_loader = DataLoader(
            dataset, batch_size=param_dict['batch_size'],
            shuffle=False, num_workers=param_dict['num_workers'], drop_last=False
        )
        weight = get_ensemble_weight(seq_len, param_dict['eval_mode'])

        # 缓冲区初始化
        num_sample, sample_count = len(dataset), 0
        buffer_size = seq_len - 1
        batch_i = torch.arange(seq_len)
        frame_i = torch.arange(seq_len - 1, -1, -1)
        y_pred_buffer = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)

        data_prob = tqdm(data_loader) if param_dict['verbose'] else data_loader
        for step, (i, x, y, _, _) in enumerate(data_prob):
            x = x.float().to(DEVICE)
            b_size, seq_len = i.shape[0], i.shape[1]
            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()

            y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)
            ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
            ensemble_y = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)
            ensemble_y_pred = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)

            for b in range(b_size):
                if sample_count < buffer_size:
                    # 缓冲区未满
                    y_pred_avg = y_pred_buffer[batch_i + b, frame_i].sum(0) / (sample_count + 1)
                else:
                    # 加权集成
                    y_pred_avg = (y_pred_buffer[batch_i + b, frame_i] * weight[:, None, None]).sum(0)

                ensemble_i = torch.cat((ensemble_i, i[b][0].reshape(1, 1, 2)), dim=0)
                ensemble_y = torch.cat((ensemble_y, y[b][0].reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred_avg.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                sample_count += 1

                if sample_count == num_sample:
                    # 最后一批处理
                    y_zero_pad = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
                    y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)

                    for f in range(1, seq_len):
                        y_pred_avg = y_pred_buffer[batch_i + b + f, frame_i].sum(0) / (seq_len - f)
                        ensemble_i = torch.cat((ensemble_i, i[-1][f].reshape(1, 1, 2)), dim=0)
                        ensemble_y = torch.cat((ensemble_y, y[-1][f].reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                        ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred_avg.reshape(1, 1, HEIGHT, WIDTH)), dim=0)

            tmp_pred = evaluate(ensemble_i, y_true=ensemble_y, y_pred=ensemble_y_pred,
                                tolerance=param_dict['tolerance'])
            for key in tmp_pred.keys():
                tracknet_pred_dict[key].extend(tmp_pred[key])

            # 滑动缓冲区更新
            y_pred_buffer = y_pred_buffer[-buffer_size:]

    # 第二步：生成修复掩码
    tracknet_pred_dict['Inpaint_Mask'] = generate_inpaint_mask(tracknet_pred_dict, th_h=30)

    # 第三步：加载真实标签（用于评估对比）
    file_format_str = os.path.join('{}', 'frame', '{}')
    match_dir, rally_id = parse.parse(file_format_str, rally_dir)
    csv_file = os.path.join(match_dir, 'corrected_csv', f'{rally_id}_ball.csv')
    label_df = pd.read_csv(csv_file, encoding='utf-8')
    x_gt, y_gt = label_df['X'].values / w, label_df['Y'].values / h  # 归一化到网络尺寸

    # 第四步：线性插值修复
    x_pred = linear_interp(tracknet_pred_dict['X'], tracknet_pred_dict['Inpaint_Mask']) / WIDTH
    y_pred = linear_interp(tracknet_pred_dict['Y'], tracknet_pred_dict['Inpaint_Mask']) / HEIGHT

    # 第五步：构建评估输入并评估
    d_i = torch.empty((0, 1, 2), dtype=torch.float32)
    coor = torch.empty((0, 1, 2), dtype=torch.float32)
    coor_inpaint = torch.empty((0, 1, 2), dtype=torch.float32)

    for i in range(len(label_df)):
        d_i = torch.cat((d_i, torch.tensor([[[0, i]]], dtype=torch.float32)), dim=0)
        coor = torch.cat((coor, torch.tensor([[[x_gt[i], y_gt[i]]]], dtype=torch.float32)), dim=0)
        coor_inpaint = torch.cat((coor_inpaint, torch.tensor([[[x_pred[i], y_pred[i]]]], dtype=torch.float32)), dim=0)

    inpaintnet_pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': [], 'Type': []}
    tmp_pred = evaluate(d_i, c_true=coor, c_pred=coor_inpaint,
                        tolerance=param_dict['tolerance'], img_scaler=(w_scaler, h_scaler))
    for key in tmp_pred.keys():
        inpaintnet_pred_dict[key].extend(tmp_pred[key])

    return inpaintnet_pred_dict


if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracknet_file', type=str, help='TrackNet模型检查点路径')
    parser.add_argument('--inpaintnet_file', type=str, default='', help='InpaintNet模型检查点路径（可选）')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='测试数据划分')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--tolerance', type=float, default=4, help='坐标误差容差（像素，输入尺寸下）')
    parser.add_argument('--eval_mode', type=str, default='weight',
                        choices=['nonoverlap', 'average', 'weight'],
                        help='评估模式：nonoverlap（快速）、average（均匀集成）、weight（加权集成）')
    parser.add_argument('--video_file', type=str, default='',
                        help='单个视频文件路径（如果在数据集目录中）')
    parser.add_argument('--output_pred', action='store_true', default=False,
                        help='是否输出详细预测结果用于错误分析')
    parser.add_argument('--output_bbox', action='store_true', default=False,
                        help='是否输出COCO格式边界框用于mAP评估')
    parser.add_argument('--save_dir', type=str, default='output', help='结果保存目录')
    parser.add_argument('--verbose', action='store_true', default=False, help='显示详细进度')
    parser.add_argument('--debug', action='store_true', default=False, help='调试模式（只测第一个回合）')
    parser.add_argument('--linear_interp', action='store_true', default=False,
                        help='使用线性插值而非InpaintNet进行修复（基线对比）')
    args = parser.parse_args()

    param_dict = vars(args)
    param_dict['num_workers'] = args.batch_size if args.batch_size <= 16 else 16
    param_dict['output_bbox'] = args.output_bbox
    param_dict['output_gt'] = False

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 加载模型
    print(f'Loading checkpoint...')
    if args.tracknet_file:
        tracknet_ckpt = torch.load(args.tracknet_file, map_location=DEVICE)
        param_dict['tracknet_seq_len'] = tracknet_ckpt['param_dict']['seq_len']
        param_dict['bg_mode'] = tracknet_ckpt['param_dict']['bg_mode']
        tracknet = get_model('TrackNet', seq_len=param_dict['tracknet_seq_len'],
                             bg_mode=param_dict['bg_mode']).to(DEVICE)
        tracknet.load_state_dict(tracknet_ckpt['model'])
        model = (tracknet, None)
    else:
        tracknet = None

    if args.inpaintnet_file:
        inpaintnet_ckpt = torch.load(args.inpaintnet_file, map_location=DEVICE)
        param_dict['inpaintnet_seq_len'] = inpaintnet_ckpt['param_dict']['seq_len']
        inpaintnet = get_model('InpaintNet').to(DEVICE)
        inpaintnet.load_state_dict(inpaintnet_ckpt['model'])
        model = (tracknet, inpaintnet)

    if args.video_file:
        # 单视频评估模式（用于快速测试）
        print(f'Test on video {args.video_file} ...')
        file_format_str = os.path.join('{}', 'video', '{}.mp4')
        match_dir, rally_id = parse.parse(file_format_str, args.video_file)
        rally_dir = os.path.join(match_dir, 'frame', rally_id)

        # 加载标签
        csv_file = os.path.join(match_dir, 'corrected_csv',
                                f'{rally_id}_ball.csv') if 'test' in rally_dir else os.path.join(match_dir, 'csv',
                                                                                                 f'{rally_id}_ball.csv')
        assert os.path.exists(csv_file), f'{csv_file} does not exist.'
        label_df = pd.read_csv(csv_file, encoding='utf8').sort_values(by='Frame').fillna(0)

        # 预测标签
        pred_dict = test_rally(model, rally_dir, param_dict)

        # 写入结果
        out_video_file = os.path.join(args.save_dir, f'{rally_id}.mp4')
        out_csv_file = os.path.join(args.save_dir, f'{rally_id}_ball.csv')
        frame_list, fps, (w, h) = generate_frames(args.video_file)
        write_pred_video(frame_list, dict(fps=fps, shape=(w, h)), pred_dict, label_df=label_df,
                         save_file=out_video_file)
        write_pred_csv(pred_dict, save_file=out_csv_file)
    else:
        # 完整数据集评估模式
        eval_analysis_file = os.path.join(args.save_dir, f'{args.split}_eval_analysis_{args.eval_mode}.json')
        eval_res_file = os.path.join(args.save_dir, f'{args.split}_eval_res_{args.eval_mode}.json')

        start_time = time.time()
        print(f'Split: {args.split}')
        print(f'Evaluation mode: {args.eval_mode}')
        print(f'Tolerance Value: {args.tolerance}')

        # 执行测试
        pred_dict = test(model, args.split, param_dict, linear_interp=args.linear_interp)

        # 计算指标（test集可应用drop_frame裁剪）
        if args.split == 'test':
            res_dict = get_test_res(pred_dict, drop=True)
        else:
            res_dict = get_test_res(pred_dict, drop=False)

        # 保存结果
        with open(eval_res_file, 'w') as f:
            json.dump(res_dict, f, indent=2)

        if args.output_pred:
            eval_dict = dict(param_dict=param_dict, pred_dict=pred_dict)
            with open(eval_analysis_file, 'w') as f:
                json.dump(eval_dict, f, indent=2)

        # COCO mAP评估（如果启用output_bbox）
        if args.output_bbox:
            coco_file = os.path.join(args.save_dir, f'{args.split}_coco_res_{args.eval_mode}.json')
            if args.split == 'test':
                dect_list = get_coco_res(pred_dict, drop=True)
            else:
                dect_list = get_coco_res(pred_dict, drop=False)

            # 计算不同IoU阈值下的mAP
            mAP = {0.25: 0, 0.5: 0}
            coco_gt = COCO(os.path.join(data_dir, 'coco_format_gt.json'))
            coco_dt = coco_gt.loadRes(dect_list)
            for iou_th in [0.25, 0.5]:
                coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
                coco_eval.params.iouThrs = [iou_th]
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                mAP[iou_th] = coco_eval.stats[0]  # stats[0]是mAP

            coco_res_dict = dict(AP_25=mAP, detection=dect_list)
            with open(coco_file, 'w') as f:
                json.dump(coco_res_dict, f, indent=2)