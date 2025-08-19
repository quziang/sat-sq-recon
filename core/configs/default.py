
# 版权所有 (c) 2023 SLAB Group
# 作者: Tae Ha "Jeff" Park (tpark94@stanford.edu)
#
# 本文件为项目默认配置，包含训练、数据、模型等所有主要参数。


from os.path import join
from yacs.config import CfgNode as CN


# 创建配置节点对象
_C = CN()


# ------------------------------------------------------------------------------ #
# 基础设置
# ------------------------------------------------------------------------------ #
_C.ROOT       = '/root/sat-sq-recon'                   # 项目根目录
_C.OUTPUT_DIR = 'output'                                            # 训练输出保存文件夹
_C.LOG_DIR    = 'log'                                               # 日志保存文件夹
_C.EXP_NAME   = 'expNameTemp'                                       # 当前实验名称

# 基础参数
_C.LOG_TENSORBOARD = False      # 是否记录 tensorboard 日志
_C.CUDA            = False      # 是否使用 GPU
_C.AMP             = False      # 是否使用混合精度训练
_C.AMP_DTYPE       = "float16" # 混合精度类型
_C.AUTO_RESUME     = True       # 是否自动恢复上次训练
_C.PIN_MEMORY      = True       # 是否使用内存 pinning
_C.SEED            = None       # 随机种子，None 时自动生成
_C.VERBOSE         = False      # 是否打印详细日志

# cudNN 相关参数
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK     = True      # cudNN benchmark 模式
_C.CUDNN.DETERMINISTIC = False     # cudNN 是否使用确定性算法
_C.CUDNN.ENABLED       = True      # cudNN 是否启用

# 分布式训练相关参数
_C.DIST = CN()
_C.DIST.RANK = 0                          # 当前节点编号
_C.DIST.BACKEND = 'nccl'                  # 通信后端
_C.DIST.MULTIPROCESSING_DISTRIBUTED = False # 是否多进程分布式


# ------------------------------------------------------------------------------ #
# 数据集相关参数
# ------------------------------------------------------------------------------ #
_C.DATASET = CN()

# 数据集目录与文件
#   ROOT/DATANAME
#   - CAMERA
#   - IMAGE_DIR
_C.DATASET.ROOT         = '/teams/microsate_1687685838/qza/dataset/SPE3R'   # 数据集根目录
_C.DATASET.DATANAME     = 'spe3r'                        # 数据集名称
_C.DATASET.CAMERA       = 'camera.json'                   # 相机参数文件
_C.DATASET.IMAGE_DIR    = 'images'                        # 图像文件夹
_C.DATASET.MASK_DIR     = 'masks'                         # 掩码文件夹

# I/O 参数
_C.DATASET.IMAGE_SIZE         = [400, 300]                # 图像尺寸
_C.DATASET.NUM_POINTS_ON_MESH = 2000                      # 网格表面采样点数
_C.DATASET.NUM_POINTS_IN_MESH = 10000                     # 网格内部采样点数

# 文件相关
_C.DATASET.SPLIT_CSV = 'splits.csv'                       # 数据集划分文件


# ------------------------------------------------------------------------------ #
# 模型相关参数
# ------------------------------------------------------------------------------ #
_C.MODEL = CN()
_C.MODEL.PRETRAIN_FILE = None                # 预训练模型路径

# 模型结构参数
_C.MODEL.LATENT_DIM         = 128            # 潜在空间维度
_C.MODEL.HIDDEN_DIM         = 256            # 隐藏层维度
_C.MODEL.NUM_MAX_PRIMITIVES = 5              # 最大原语数量
_C.MODEL.APPLY_TAPER        = True           # 是否应用 Taper 变换

# 渲染相关参数
_C.MODEL.ICOSPHERE_LEVEL = 3                 # 网格球细分等级
_C.MODEL.RENDER_SIGMA    = 1e-4              # 渲染高斯参数

# 消融实验参数
_C.MODEL.USE_TRUE_POSE_RENDER = False        # 是否使用真实姿态渲染
_C.MODEL.USE_DUAL_SQ          = True         # 是否使用双超二次体


# ------------------------------------------------------------------------------ #
# 损失函数相关参数
# ------------------------------------------------------------------------------ #
_C.LOSS = CN()
_C.LOSS.RECON_TYPE   = []        # 重建损失类型
_C.LOSS.RECON_WEIGHT = []        # 重建损失权重
_C.LOSS.POSE_TYPE    = []        # 姿态损失类型
_C.LOSS.POSE_WEIGHT  = []        # 姿态损失权重
_C.LOSS.REG_TYPE     = []        # 正则项类型
_C.LOSS.REG_WEIGHT   = []        # 正则项权重
_C.LOSS.SHARPNESS    = 10        # 锐化参数
_C.LOSS.BETA_OVERLAP = 2.0       # 原语重叠正则参数


# ------------------------------------------------------------------------------ #
# 训练相关参数
# ------------------------------------------------------------------------------ #
_C.TRAIN = CN()

# 学习率与调度器
_C.TRAIN.LR        = 0.001           # 初始学习率
_C.TRAIN.SCHEDULER = 'step'          # 学习率调度器类型
_C.TRAIN.LR_FACTOR = 0.1             # 学习率衰减因子
_C.TRAIN.LR_STEP   = [90, 110]       # 学习率衰减步数

# 优化器相关
_C.TRAIN.OPTIMIZER = 'SGD'           # 优化器名称（需与 PyTorch 优化器一致）
_C.TRAIN.WD        = 0.0001          # 权重衰减
_C.TRAIN.EPS       = 1e-5            # 优化器 epsilon
_C.TRAIN.GAMMA1    = 0.9             # 动量因子
_C.TRAIN.GAMMA2    = 0.999           # Adam 优化器二阶动量

# 训练轮数
_C.TRAIN.BEGIN_EPOCH     = 0         # 起始 epoch
_C.TRAIN.END_EPOCH       = 100       # 终止 epoch
# _C.TRAIN.STEPS_PER_EPOCH = 500     # 每轮步数（可选）
_C.TRAIN.VALID_FREQ      = 20        # 验证频率（每多少轮验证一次）
_C.TRAIN.VALID_FRACTION  = None      # 验证集采样比例

# 批次相关
_C.TRAIN.BATCH_SIZE_PER_GPU = 16     # 每个 GPU 的 batch size
_C.TRAIN.SHUFFLE            = True   # 是否打乱数据
_C.TRAIN.WORKERS            = 4      # 数据加载线程数


# ------------------------------------------------------------------------------ #
# 测试相关参数
# ------------------------------------------------------------------------------ #
_C.TEST = CN()

# 批次相关
_C.TEST.BATCH_SIZE_PER_GPU = 1       # 测试时每 GPU batch size
_C.TEST.NUM_REPEATS        = 1       # 测试重复次数



# 配置更新函数，根据命令行参数和配置文件合并配置
def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    # 可在此处添加自定义处理逻辑
    cfg.DATASET.CAMERA = join(cfg.DATASET.ROOT, cfg.DATASET.DATANAME, cfg.DATASET.CAMERA)

    # 仅支持 spe3r 相关数据集
    assert 'spe3r' in cfg.DATASET.DATANAME, 'only spe3r variants are supported for datasets'

    cfg.freeze()


# 支持命令行输出配置内容
if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)