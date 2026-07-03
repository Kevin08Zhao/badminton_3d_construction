import torch


def WBCELoss(y_pred, y, reduce=True):
    """
    加权二元交叉熵损失函数（Weighted Binary Cross Entropy）
    出自TrackNetV2论文，用于解决羽毛球检测中的极端类别不平衡问题

    背景：羽毛球在画面中仅占极少像素（<0.1%），标准BCE会导致模型倾向于预测全背景

    加权策略：
    - 正样本（y=1，羽毛球位置）：权重为 (1 - y_pred)^2
      * 当预测接近1（正确）时，权重小，损失轻微
      * 当预测接近0（错误）时，权重大，损失被显著放大（平方效应）
    - 负样本（y=0，背景）：权重为 y_pred^2
      * 当预测接近0（正确）时，权重小，损失轻微
      * 当预测接近1（误检）时，权重大，损失被显著放大

    数学公式：
    Loss = -[(1-ŷ)² · y · log(ŷ) + ŷ² · (1-y) · log(1-ŷ)]

    Args:
        y_pred (torch.Tensor): 模型预测值，经Sigmoid激活后的概率图
            形状：(N, 1, H, W)，N为batch size，H/W为特征图高/宽
        y (torch.Tensor): 真实标签（Ground Truth）
            形状：(N, 1, H, W)，值为0（背景）或1（羽毛球位置）
        reduce (bool): 损失归约方式
            - True: 返回所有像素的平均损失（标量，用于反向传播）
            - False: 返回每个样本的平均损失（形状(N,)，用于分析各样本难度）

    Returns:
        torch.Tensor:
            - reduce=True时，形状(1,)，标量损失值
            - reduce=False时，形状(N,)，每个样本的独立损失

    Note:
        使用torch.clamp限制y_pred范围在[1e-7, 1]，防止log(0)产生NaN
    """

    # 计算加权BCE损失
    # 正样本项：(1-y_pred)^2 * y * log(y_pred)
    # 负样本项：y_pred^2 * (1-y) * log(1-y_pred)
    loss = (-1) * (
            torch.square(1 - y_pred) * y * torch.log(torch.clamp(y_pred, 1e-7, 1)) +
            torch.square(y_pred) * (1 - y) * torch.log(torch.clamp(1 - y_pred, 1e-7, 1))
    )

    if reduce:
        # 全局平均：所有batch所有像素的平均损失（标准训练模式）
        return torch.mean(loss)
    else:
        # 样本级平均：先展平H,W维度，再对空间维度求平均，保留batch维度
        # 用于困难样本挖掘（Hard Negative Mining）或样本权重调整
        return torch.mean(torch.flatten(loss, start_dim=1), dim=1)


def get_metric(TP, TN, FP1, FP2, FN):
    """
    计算羽毛球检测/跟踪任务的多维度评估指标

    混淆矩阵定义（基于TrackNetV2的评估协议）：
    - TP (True Positive): 真正例，正确检测到羽毛球且位置准确（通常<5像素误差）
    - TN (True Negative): 真负例，正确识别为无球帧（背景）
    - FP1 (False Positive Type 1): I类假阳性，检测到球但位置偏差过大（>阈值像素）
    - FP2 (False Positive Type 2): II类假阳性，背景区域被误检为球（鬼影/噪声）
    - FN (False Negative): 假负例，漏检（实际有球但模型未检测出或置信度<阈值）

    Args:
        TP (int): 真正例数量
        TN (int): 真负例数量
        FP1 (int): I类假阳性数量（定位不准）
        FP2 (int): II类假阳性数量（误检背景）
        FN (int): 假负例数量（漏检）

    Returns:
        tuple: 包含5个评估指标的元组
            - accuracy (float): 准确率，(TP+TN)/Total，整体正确率
            - precision (float): 精确率，TP/(TP+FP1+FP2)，预测为球的结果中确实正确的比例
            - recall (float): 召回率，TP/(TP+FN)，实际有球的帧中被成功检出的比例
            - f1 (float): F1分数，2·P·R/(P+R)，精确率和召回率的调和平均
            - miss_rate (float): 漏检率，FN/(TP+FN)，与召回率互补（1-recall）

    Note:
        所有指标均包含除零保护（分母为0时返回0），避免训练初期或特殊情况下崩溃
    """

    # 总样本数（所有类别之和）
    total = TP + TN + FP1 + FP2 + FN

    # 准确率：所有预测正确的比例（包括正例和负例）
    # 在极度不平衡数据中，TN占比极高，此指标可能虚高（需结合F1判断）
    accuracy = (TP + TN) / total if total > 0 else 0

    # 精确率：预测为正的样本中实际为正的比例
    # 高精确率意味着较少的误检（FP1+FP2少），但可能伴随漏检（FN多）
    precision = TP / (TP + FP1 + FP2) if (TP + FP1 + FP2) > 0 else 0

    # 召回率（敏感度）：实际为正的样本中被检出的比例
    # 高召回率意味着较少的漏检，但可能伴随误检增多
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # F1分数：精确率和召回率的调和平均，综合评价指标
    # 当精确率和召回率都高时，F1才高；单方面高无法提升F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 漏检率：与召回率互斥，直接反映漏检情况（比赛中漏检会导致轨迹断裂）
    miss_rate = FN / (TP + FN) if (TP + FN) > 0 else 0

    return accuracy, precision, recall, f1, miss_rate