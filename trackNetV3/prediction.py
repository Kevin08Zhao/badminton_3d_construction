import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from typing import Optional, Tuple, List, Dict, Union, Any
from .test_tmp import predict_location, get_ensemble_weight, generate_inpaint_mask
from .dataset_tmp import Shuttlecock_Trajectory_Dataset, Video_IterableDataset
from .general_tmp import get_model_tmp, generate_frames, to_img, write_pred_csv, write_pred_video, to_img_format
import math

# ==================== 全局常量配置 ====================
WIDTH, HEIGHT = 512, 288  # 网络输入图像尺寸（宽512，高288，保持16:9比例）
DELTA_T = 1 / math.sqrt(HEIGHT ** 2 + WIDTH ** 2)  # 归一化距离单位（对角线倒数）
COOR_TH = DELTA_T * 50  # 坐标有效性阈值（50像素对应的归一化距离，小于此值视为无效/缺失）


def predict(indices, y_pred=None, c_pred=None, img_scaler=(1, 1)):
    """
    从模型输出（热力图或坐标）提取最终预测坐标

    功能：
    1. 热力图模式：通过predict_location从热力图提取峰值位置（质心）
    2. 坐标模式：直接将归一化坐标反缩放回原图尺寸
    3. 去重：确保同一帧不会被重复输出（通过prev_f_i跟踪）

    Args:
        indices (torch.Tensor): 输入序列的帧索引，形状(N, L, 2)
            格式：[[[batch_id, frame_id], ...], ...]，用于确定输出对应的帧号
        y_pred (torch.Tensor, optional): TrackNet预测的热力图序列，形状(N, L, H, W)
            值范围[0,1]，通过>0.5二值化后提取连通区域中心
        c_pred (torch.Tensor, optional): InpaintNet预测的坐标序列，形状(N, L, 2)
            值范围[0,1]（归一化坐标），需乘以img_scaler反缩放
        img_scaler (Tuple[float, float]): 图像缩放因子 (w_scaler, h_scaler)
            用于将网络输出坐标映射回原视频分辨率（原图尺寸/网络输入尺寸）

    Returns:
        pred_dict (Dict): 预测结果字典
            {
                'Frame': []   # 帧号列表（int）
                'X': []       # X坐标列表（int，原图分辨率）
                'Y': []       # Y坐标列表（int，原图分辨率）
                'Visibility': []  # 可见性列表（0或1，0表示该帧无球）
            }

    Raises:
        ValueError: 如果y_pred和c_pred同时未提供或同时提供（互斥）
    """

    pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}

    batch_size, seq_len = indices.shape[0], indices.shape[1]
    # 确保indices在CPU上并转为numpy
    indices = indices.detach().cpu().numpy() if torch.is_tensor(indices) else indices.numpy()

    # 热力图预处理：阈值化为0/1，并转换为图像格式(N, L, H, W)
    if y_pred is not None:
        y_pred = y_pred > 0.5  # 二值化：>0.5为1（球区域），否则为0
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_pred = to_img_format(y_pred)  # 转换格式以便逐帧处理

    # 坐标预处理：转为numpy
    if c_pred is not None:
        c_pred = c_pred.detach().cpu().numpy() if torch.is_tensor(c_pred) else c_pred

    prev_f_i = -1  # 上一帧号，用于去重（滑动窗口会导致同一帧被多次预测）

    for n in range(batch_size):
        for f in range(seq_len):
            f_i = indices[n][f][1]  # 提取帧号（索引的第二维度）

            # 跳过重复帧（overlap模式下，相邻batch会包含相同帧）
            if f_i != prev_f_i:
                if c_pred is not None:
                    # InpaintNet模式：从归一化坐标反缩放
                    c_p = c_pred[n][f]
                    cx_pred = int(c_p[0] * WIDTH * img_scaler[0])  # x * 网络宽 * 缩放因子
                    cy_pred = int(c_p[1] * HEIGHT * img_scaler[1])  # y * 网络高 * 缩放因子

                elif y_pred is not None:
                    # TrackNet模式：从热力图提取坐标
                    y_p = y_pred[n][f]
                    # 从二值图提取边界框，计算中心点
                    bbox_pred = predict_location(to_img(y_p))
                    cx_pred = int(bbox_pred[0] + bbox_pred[2] / 2)  # x_center = left + width/2
                    cy_pred = int(bbox_pred[1] + bbox_pred[3] / 2)  # y_center = top + height/2
                    # 反缩放到原图分辨率
                    cx_pred = int(cx_pred * img_scaler[0])
                    cy_pred = int(cy_pred * img_scaler[1])
                else:
                    raise ValueError('必须提供y_pred（热力图）或c_pred（坐标）之一')

                # 判断可见性：坐标为(0,0)视为不可见（无球）
                vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1

                pred_dict['Frame'].append(int(f_i))
                pred_dict['X'].append(cx_pred)
                pred_dict['Y'].append(cy_pred)
                pred_dict['Visibility'].append(vis_pred)
                prev_f_i = f_i
            else:
                # 遇到重复帧立即跳出（batch内的帧是连续的，后面的都会重复）
                break

    return pred_dict


def predict_trajectory(
        video_file=r'/data/video/test0.mp4',
        tracknet_file=r'/data/weights/ckpts/TrackNet_best.pt',
        inpaintnet_file=None,  # 可选，不提供则只运行TrackNet
        batch_size=16,
        eval_mode='nonoverlap',  # 'nonoverlap'（快速）或'overlap'（高精度，时序集成）
        large_video=True,  # 是否使用流式加载（大视频）或一次性加载（小视频）
        output_video=False,  # 是否生成可视化视频
        save_dir='data/tmp',
        out_csv_file: Optional[str] = None,
        video_range=None,  # 大视频时生成中值背景的时间范围（秒）
        max_sample_num=1800,  # 生成中值背景的最大采样帧数
        traj_len=8,  # 可视化时轨迹长度（历史轨迹点数量）
        return_dict=False  # 是否返回结果字典（当前未实现，仅保存文件）
):
    """
    TrackNetV3轨迹预测主函数
    支持双阶段推理：TrackNet（检测）+ InpaintNet（修复）
    支持两种评估模式：非重叠（快速）和重叠（时序集成，高精度）

    流程：
    1. 加载TrackNet模型，提取帧序列特征，预测热力图
    2. （可选）加载InpaintNet，修复TrackNet的漏检/遮挡帧
    3. 时序集成（overlap模式）或多帧独立预测（nonoverlap模式）
    4. 保存CSV结果和可视化视频（可选）
    """

    # ==================== 初始化配置 ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = batch_size if batch_size <= 16 else 16  # 数据加载进程数限制
    video_name = video_file.split('/')[-1][:-4]  # 提取视频名（去路径和扩展名）

    # 输出路径配置
    if out_csv_file is None:
        out_csv_file = os.path.join(save_dir, 'tmp1.csv')  # 临时CSV（会被覆盖）
    out_video_file = os.path.join(save_dir, f'{video_name}.mp4')

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_csv_file) or '.', exist_ok=True)

    # ==================== 模型加载 ====================
    # 加载TrackNet检查点，提取配置参数
    tracknet_ckpt = torch.load(tracknet_file, map_location=device)
    tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']  # TrackNet输入序列长度（通常8）
    bg_mode = tracknet_ckpt['param_dict']['bg_mode']  # 背景处理模式

    # 初始化TrackNet并加载权重，移至当前可用设备（GPU/CPU）
    tracknet = get_model_tmp('TrackNet', tracknet_seq_len, bg_mode).to(device)
    tracknet.load_state_dict(tracknet_ckpt['model'])
    tracknet.eval()  # 切换到评估模式（关闭BN和Dropout）

    # 可选：加载InpaintNet（TrackNetV3的两阶段架构）
    if inpaintnet_file:
        inpaintnet_ckpt = torch.load(inpaintnet_file, map_location=device)
        inpaintnet_seq_len = inpaintnet_ckpt['param_dict']['seq_len']  # InpaintNet序列长度（通常8）
        inpaintnet = get_model_tmp('InpaintNet').to(device)
        inpaintnet.load_state_dict(inpaintnet_ckpt['model'])
        inpaintnet.eval()
    else:
        inpaintnet = None

    # ==================== 视频参数获取 ====================
    cap = cv2.VideoCapture(video_file)
    w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    w_scaler, h_scaler = w / WIDTH, h / HEIGHT  # 计算缩放因子（原图/网络输入）
    img_scaler = (w_scaler, h_scaler)
    cap.release()  # 仅获取参数，后续由Dataset重新打开

    # TrackNet预测结果容器（供InpaintNet使用或作为最终输出）
    tracknet_pred_dict = {
        'Frame': [], 'X': [], 'Y': [], 'Visibility': [],
        'Inpaint_Mask': [],  # 需要修复的帧标记（InpaintNet使用）
        'Img_scaler': (w_scaler, h_scaler),
        'Img_shape': (w, h)
    }

    # ==================== TrackNet推理阶段 ====================
    seq_len = tracknet_seq_len

    if eval_mode == 'nonoverlap':
        # 模式A：非重叠滑动窗口（快速推理，无时序集成）
        # sliding_step = seq_len，相邻窗口无重叠，每帧仅被预测一次
        if large_video:
            # 大视频：使用IterableDataset流式加载，避免内存溢出
            dataset = Video_IterableDataset(
                video_file, seq_len=seq_len, sliding_step=seq_len,
                bg_mode=bg_mode, max_sample_num=max_sample_num, video_range=video_range
            )
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
            print(f'Video length: {dataset.video_len}')
        else:
            # 小视频：一次性加载所有帧到内存
            frame_list = generate_frames(video_file)
            dataset = Shuttlecock_Trajectory_Dataset(
                seq_len=seq_len, sliding_step=seq_len, data_mode='heatmap',
                bg_mode=bg_mode, frame_arr=np.array(frame_list)[:, :, :, ::-1],  # BGR转RGB
                padding=True  # 最后一帧填充，确保整除
            )
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, drop_last=False)

        # 批量推理
        for step, (i, x) in enumerate(tqdm(data_loader)):
            x = x.float().to(device)
            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()  # 预测热力图序列

            # 提取坐标并累加结果
            tmp_pred = predict(i, y_pred=y_pred, img_scaler=img_scaler)
            for key in tmp_pred.keys():
                tracknet_pred_dict[key].extend(tmp_pred[key])

    else:
        # 模式B：重叠滑动窗口 + 时序集成（高精度，TrackNetV2/V3标准做法）
        # sliding_step = 1，每帧被多个窗口覆盖，通过加权平均多窗口预测提高稳定性
        if large_video:
            dataset = Video_IterableDataset(
                video_file, seq_len=seq_len, sliding_step=1,  # 关键：步长=1，重叠度最高
                bg_mode=bg_mode, max_sample_num=max_sample_num, video_range=video_range
            )
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
            video_len = dataset.video_len
            print(f'Video length: {video_len}')
        else:
            frame_list = generate_frames(video_file)
            video_len = len(frame_list)
            dataset = Shuttlecock_Trajectory_Dataset(
                seq_len=seq_len, sliding_step=1, data_mode='heatmap',
                bg_mode=bg_mode, frame_arr=np.array(frame_list)[:, :, :, ::-1]
            )
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, drop_last=False)

        # 时序集成缓冲区初始化
        num_sample = video_len - seq_len + 1  # 有效样本数（滑动窗口数量）
        sample_count = 0  # 当前已处理样本计数
        buffer_size = seq_len - 1  # 缓冲区大小（需保留的历史预测数）

        # 批次和帧索引辅助张量
        batch_i = torch.arange(seq_len)  # [0, 1, 2, ..., seq_len-1]
        frame_i = torch.arange(seq_len - 1, -1, -1)  # [seq_len-1, ..., 1, 0]（逆序，用于对齐时序）

        # 热力图缓冲区：存储最近的buffer_size个batch的预测结果，用于时序集成
        y_pred_buffer = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)

        # 获取时序集成权重（通常高斯分布或线性衰减，中心帧权重最高）
        weight = get_ensemble_weight(seq_len, eval_mode)

        for step, (i, x) in enumerate(tqdm(data_loader)):
            x = x.float().to(device)
            b_size = i.shape[0]  # 实际batch size（最后一批可能不足）

            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()

            # 将新预测加入缓冲区（沿着batch维度拼接）
            y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)

            # 集成结果容器
            ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)  # 帧索引
            ensemble_y_pred = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)  # 集成后的热力图

            for b in range(b_size):
                if sample_count < buffer_size:
                    # 缓冲区未满（前seq_len-1个样本）：简单平均
                    # 从缓冲区提取对应batch的seq_len个预测，按逆序frame_i对齐，求平均
                    y_pred_avg = y_pred_buffer[batch_i + b, frame_i].sum(0) / (sample_count + 1)
                else:
                    # 缓冲区已满：加权时序集成
                    # 提取batch_i+b位置的seq_len个历史预测，按frame_i逆序对齐，乘以weight后求和
                    y_pred_avg = (y_pred_buffer[batch_i + b, frame_i] * weight[:, None, None]).sum(0)

                # 收集集成结果（只取每组的第一个帧索引作为代表）
                ensemble_i = torch.cat((ensemble_i, i[b][0].reshape(1, 1, 2)), dim=0)
                ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred_avg.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                sample_count += 1

                # 处理最后一批的尾部帧（不足seq_len的剩余帧）
                if sample_count == num_sample:
                    # 用零填充缓冲区，确保尾部帧有足够的历史上下文
                    y_zero_pad = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
                    y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)

                    # 处理剩余的seq_len-1帧，逐步减少平均数量（边界处理）
                    for f in range(1, seq_len):
                        # 对第f个后续帧，可用历史预测数为seq_len-f，求平均
                        y_pred_avg = y_pred_buffer[batch_i + b + f, frame_i].sum(0) / (seq_len - f)
                        ensemble_i = torch.cat((ensemble_i, i[-1][f].reshape(1, 1, 2)), dim=0)
                        ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred_avg.reshape(1, 1, HEIGHT, WIDTH)), dim=0)

            # 从集成后的热力图提取坐标
            tmp_pred = predict(ensemble_i, y_pred=ensemble_y_pred, img_scaler=img_scaler)
            for key in tmp_pred.keys():
                tracknet_pred_dict[key].extend(tmp_pred[key])

            # 滑动缓冲区：保留最后buffer_size个预测，用于下一批的时序集成
            y_pred_buffer = y_pred_buffer[-buffer_size:]

    # ==================== InpaintNet推理阶段（可选） ====================
    if inpaintnet is not None:
        inpaintnet.eval()
        seq_len = inpaintnet_seq_len

        # 生成修复掩码：标记TrackNet预测失败的帧（遮挡、运动模糊、出界等）
        # th_h=h*0.05：y坐标小于图像高度5%视为出界（羽毛球落地后弹起前）
        tracknet_pred_dict['Inpaint_Mask'] = generate_inpaint_mask(tracknet_pred_dict, th_h=h * 0.05)
        inpaint_pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}

        if eval_mode == 'nonoverlap':
            # InpaintNet非重叠模式
            dataset = Shuttlecock_Trajectory_Dataset(
                seq_len=seq_len, sliding_step=seq_len, data_mode='coordinate',
                pred_dict=tracknet_pred_dict, padding=True
            )
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, drop_last=False)

            for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_loader)):
                coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()

                with torch.no_grad():
                    # InpaintNet输入：预测坐标+修复掩码，输出修复后的坐标
                    coor_inpaint = inpaintnet(
                        coor_pred.to(device), inpaint_mask.to(device)
                    ).detach().cpu()

                # 掩码融合：掩码=1的位置用InpaintNet输出，掩码=0的位置保持TrackNet原预测
                coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1 - inpaint_mask)

                # 阈值处理：坐标小于COOR_TH（接近0）视为无效，置0表示不可见
                th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
                coor_inpaint[th_mask] = 0.

                # 提取并累加结果
                tmp_pred = predict(i, c_pred=coor_inpaint, img_scaler=img_scaler)
                for key in tmp_pred.keys():
                    inpaint_pred_dict[key].extend(tmp_pred[key])

        else:
            # InpaintNet重叠模式 + 时序集成（逻辑与TrackNet类似）
            dataset = Shuttlecock_Trajectory_Dataset(
                seq_len=seq_len, sliding_step=1, data_mode='coordinate',
                pred_dict=tracknet_pred_dict
            )
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, drop_last=False)
            weight = get_ensemble_weight(seq_len, eval_mode)

            # 缓冲区初始化（坐标缓冲区，而非热力图）
            num_sample, sample_count = len(dataset), 0
            buffer_size = seq_len - 1
            batch_i = torch.arange(seq_len)
            frame_i = torch.arange(seq_len - 1, -1, -1)
            coor_inpaint_buffer = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)

            for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_loader)):
                coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()
                b_size = i.shape[0]

                with torch.no_grad():
                    coor_inpaint = inpaintnet(
                        coor_pred.to(device), inpaint_mask.to(device)
                    ).detach().cpu()
                coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1 - inpaint_mask)

                # 阈值处理
                th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
                coor_inpaint[th_mask] = 0.

                # 时序集成（与TrackNet类似，但对坐标直接平均而非热力图）
                coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_inpaint), dim=0)
                ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
                ensemble_coor_inpaint = torch.empty((0, 1, 2), dtype=torch.float32)

                for b in range(b_size):
                    if sample_count < buffer_size:
                        coor_inpaint_avg = coor_inpaint_buffer[batch_i + b, frame_i].sum(0)
                        coor_inpaint_avg /= (sample_count + 1)
                    else:
                        coor_inpaint_avg = (coor_inpaint_buffer[batch_i + b, frame_i] * weight[:, None]).sum(0)

                    ensemble_i = torch.cat((ensemble_i, i[b][0].view(1, 1, 2)), dim=0)
                    ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint_avg.view(1, 1, 2)), dim=0)
                    sample_count += 1

                    # 尾部帧处理
                    if sample_count == num_sample:
                        coor_zero_pad = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
                        coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_zero_pad), dim=0)

                        for f in range(1, seq_len):
                            coor_inpaint_avg = coor_inpaint_buffer[batch_i + b + f, frame_i].sum(0)
                            coor_inpaint_avg /= (seq_len - f)
                            ensemble_i = torch.cat((ensemble_i, i[-1][f].view(1, 1, 2)), dim=0)
                            ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint_avg.view(1, 1, 2)),
                                                              dim=0)

                # 阈值处理（集成后再次过滤）
                th_mask = ((ensemble_coor_inpaint[:, :, 0] < COOR_TH) & (ensemble_coor_inpaint[:, :, 1] < COOR_TH))
                ensemble_coor_inpaint[th_mask] = 0.

                tmp_pred = predict(ensemble_i, c_pred=ensemble_coor_inpaint, img_scaler=img_scaler)
                for key in tmp_pred.keys():
                    inpaint_pred_dict[key].extend(tmp_pred[key])

                # 滑动缓冲区
                coor_inpaint_buffer = coor_inpaint_buffer[-buffer_size:]

    # ==================== 结果输出 ====================
    # 选择最终输出：如果有InpaintNet则使用其修复结果，否则使用TrackNet原始结果
    pred_dict = inpaint_pred_dict if inpaintnet is not None else tracknet_pred_dict

    # 写入CSV文件（包含帧号、XY坐标、可见性）
    write_pred_csv(pred_dict, save_file=out_csv_file)

    # 可选：生成带轨迹线的可视化视频
    if output_video:
        write_pred_video(video_file, pred_dict, save_file=out_video_file, traj_len=traj_len)

    print('Done.')


# 使用示例
if __name__ == '__main__':
    result = predict_trajectory(
        video_file=r'/data/video/input_video.mp4',
        tracknet_file=r'/data/weights/ckpts/TrackNet_best.pt',
        inpaintnet_file=None,  # 不提供则仅运行TrackNet（TrackNetV2模式）
        batch_size=16,
        eval_mode='nonoverlap',  # 'overlap'为高精度模式，带时序集成
        large_video=True,  # 大视频流式处理
        output_video=False,
        save_dir='data/tmp',
        return_dict=False
    )
    print("Prediction completed!")