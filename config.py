"""
配置文件
包含所有训练、数据、模型参数
"""

import torch
import os


class Config:
    """配置类"""

    # ==================== 数据相关 ====================
    # 数据目录
    DATA_DIR = '../../data/carla_round_all'

    # 轨迹长度
    OBSERVATION_LEN = 25  # 观察帧数：25帧 = 1秒 @ 25fps
    PREDICTION_LEN = 25  # 预测帧数：25帧 = 1秒

    # 环岛类型映射
    ROUNDABOUT_TYPES = {
        0: [0],  # Type 0: recording 0
        1: [1],  # Type 1: recording 1
        2: [2, 3, 4, 5, 6, 7, 8],  # Type 2: recordings 2-8
        9: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # Type 9: recordings 9-23
    }

    # 排除的车辆类型（VRUs - Vulnerable Road Users）
    EXCLUDED_CLASSES = ['pedestrian', 'bicycle', 'motorcyclist']

    # 数据集划分
    VAL_SPLIT = 0.2  # 验证集比例

    # ==================== 模型相关 ====================
    # Transformer参数
    HIDDEN_DIM = 64  # 隐藏层维度
    N_HEADS = 8  # 注意力头数
    N_LAYERS = 4  # Transformer层数
    DROPOUT = 0.1  # Dropout比率
    DIM_FEEDFORWARD = 256  # 前馈网络维度（通常是HIDDEN_DIM的4倍）

    # 输入输出维度
    INPUT_DIM = 2  # 输入维度 (x, y)
    OUTPUT_DIM = 2  # 输出维度 (x, y)

    # ==================== GCN相关 ====================
    USE_GCN = True                # 是否启用GCN空间交互建模
    NEIGHBOR_RADIUS = 30.0        # 邻居搜索半径（米），环岛外半径~25m
    MAX_NEIGHBORS = 8             # 每个时间步最大邻居数
    GCN_LAYERS = 2                # GCN层数（论文：two-layer GCN）
    GCN_INPUT_DIM = 2             # GCN节点输入维度 (relative_x, relative_y)

    # ==================== 训练相关 ====================
    # 基本参数
    BATCH_SIZE = 256  # 批次大小
    EPOCHS = 50  # 训练轮数
    LEARNING_RATE = 5e-5  # 初始学习率（配合小初始化）
    WEIGHT_DECAY = 1e-4  # 权重衰减

    # Early Stopping
    PATIENCE = 10  # 早停耐心值

    # 优化器参数
    ADAM_BETAS = (0.9, 0.999)
    ADAM_EPS = 1e-8

    # 学习率调度器参数
    LR_FACTOR = 0.5  # 学习率衰减因子
    LR_PATIENCE = 5  # 学习率调度器耐心值
    LR_MIN = 1e-6  # 最小学习率
    LR_WARMUP_EPOCHS = 3  # 学习率预热轮数

    # 梯度裁剪
    GRAD_CLIP = 0.5  # 梯度裁剪阈值（降低以增强稳定性）

    # ==================== 系统相关 ====================
    # 设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 随机种子
    SEED = 42

    # DataLoader参数
    NUM_WORKERS = 4  # 数据加载线程数（Windows下建议设为0）
    PIN_MEMORY = True  # 是否将数据固定到GPU内存

    # ==================== 路径相关 ====================
    # 检查点目录
    CHECKPOINT_DIR = 'checkpoints'

    # 日志目录
    LOG_DIR = 'logs'

    # 结果目录
    RESULT_DIR = 'results'

    # 可视化目录
    VIS_DIR = 'visualizations'

    def __init__(self):
        """初始化配置，创建必要的目录"""
        # 创建所有必要的目录
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.RESULT_DIR, exist_ok=True)
        os.makedirs(self.VIS_DIR, exist_ok=True)

        # Windows系统下NUM_WORKERS建议设为0
        import platform
        if platform.system() == 'Windows':
            self.NUM_WORKERS = 0

    def __repr__(self):
        """打印配置信息"""
        info = "\n" + "=" * 60 + "\n"
        info += "Configuration Settings\n"
        info += "=" * 60 + "\n"

        info += "\n[Data Settings]\n"
        info += f"  Data directory: {self.DATA_DIR}\n"
        info += f"  Observation length: {self.OBSERVATION_LEN} frames\n"
        info += f"  Prediction length: {self.PREDICTION_LEN} frames\n"
        info += f"  Validation split: {self.VAL_SPLIT * 100:.0f}%\n"
        info += f"  Excluded classes: {self.EXCLUDED_CLASSES}\n"

        info += "\n[Model Settings]\n"
        info += f"  Hidden dimension: {self.HIDDEN_DIM}\n"
        info += f"  Number of heads: {self.N_HEADS}\n"
        info += f"  Number of layers: {self.N_LAYERS}\n"
        info += f"  Dropout: {self.DROPOUT}\n"

        info += "\n[Training Settings]\n"
        info += f"  Batch size: {self.BATCH_SIZE}\n"
        info += f"  Epochs: {self.EPOCHS}\n"
        info += f"  Learning rate: {self.LEARNING_RATE}\n"
        info += f"  Weight decay: {self.WEIGHT_DECAY}\n"
        info += f"  Patience: {self.PATIENCE}\n"
        info += f"  Gradient clip: {self.GRAD_CLIP}\n"

        info += "\n[System Settings]\n"
        info += f"  Device: {self.DEVICE}\n"
        info += f"  Random seed: {self.SEED}\n"
        info += f"  Num workers: {self.NUM_WORKERS}\n"

        info += "\n[Directory Settings]\n"
        info += f"  Checkpoints: {self.CHECKPOINT_DIR}\n"
        info += f"  Logs: {self.LOG_DIR}\n"
        info += f"  Results: {self.RESULT_DIR}\n"
        info += f"  Visualizations: {self.VIS_DIR}\n"

        info += "=" * 60 + "\n"

        return info

    def to_dict(self):
        """转换为字典"""
        return {
            # 数据相关
            'DATA_DIR': self.DATA_DIR,
            'OBSERVATION_LEN': self.OBSERVATION_LEN,
            'PREDICTION_LEN': self.PREDICTION_LEN,
            'VAL_SPLIT': self.VAL_SPLIT,
            'EXCLUDED_CLASSES': self.EXCLUDED_CLASSES,

            # 模型相关
            'HIDDEN_DIM': self.HIDDEN_DIM,
            'N_HEADS': self.N_HEADS,
            'N_LAYERS': self.N_LAYERS,
            'DROPOUT': self.DROPOUT,
            'INPUT_DIM': self.INPUT_DIM,
            'OUTPUT_DIM': self.OUTPUT_DIM,

            # GCN相关
            'USE_GCN': self.USE_GCN,
            'NEIGHBOR_RADIUS': self.NEIGHBOR_RADIUS,
            'MAX_NEIGHBORS': self.MAX_NEIGHBORS,
            'GCN_LAYERS': self.GCN_LAYERS,
            'GCN_INPUT_DIM': self.GCN_INPUT_DIM,

            # 训练相关
            'BATCH_SIZE': self.BATCH_SIZE,
            'EPOCHS': self.EPOCHS,
            'LEARNING_RATE': self.LEARNING_RATE,
            'WEIGHT_DECAY': self.WEIGHT_DECAY,
            'PATIENCE': self.PATIENCE,
            'GRAD_CLIP': self.GRAD_CLIP,

            # 系统相关
            'DEVICE': str(self.DEVICE),
            'SEED': self.SEED,
            'NUM_WORKERS': self.NUM_WORKERS
        }


# 测试代码
if __name__ == '__main__':
    """测试配置"""
    config = Config()

    print(config)

    print("\nRoundabout Types:")
    for rt, recordings in config.ROUNDABOUT_TYPES.items():
        print(f"  Type {rt}: {recordings} ({len(recordings)} recordings)")

    print("\nDirectory Check:")
    print(f"  ✓ {config.CHECKPOINT_DIR} exists: {os.path.exists(config.CHECKPOINT_DIR)}")
    print(f"  ✓ {config.LOG_DIR} exists: {os.path.exists(config.LOG_DIR)}")
    print(f"  ✓ {config.RESULT_DIR} exists: {os.path.exists(config.RESULT_DIR)}")
    print(f"  ✓ {config.VIS_DIR} exists: {os.path.exists(config.VIS_DIR)}")

    print("\nConfig as dict:")
    config_dict = config.to_dict()
    for key, value in config_dict.items():
        print(f"  {key}: {value}")