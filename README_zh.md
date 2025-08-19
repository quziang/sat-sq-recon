
# 基于单幅二维图像的航天器三维结构快速抽象

本仓库由 Stanford University [Space Rendezvous Laboratory (SLAB)](https://slab.stanford.edu) 的 Tae Ha "Jeff" Park 开发。

## 项目文件结构与说明

```
sat-sq-recon/
├── LICENSE.md                # 项目许可证
├── README.md                 # 英文说明文档
├── README_zh.md              # 中文说明文档
├── pyproject.toml            # poetry 配置与依赖
├── poetry.lock               # poetry 依赖锁定
├── setup.py                  # Cython 扩展编译脚本
├── core/                     # 项目核心代码
│   ├── configs/              # 配置文件（如 default.py）
│   ├── dataset/              # 数据集相关（加载、构建、基类等）
│   ├── nets/                 # 网络结构与相关模块
│   │   ├── losses/           # 损失函数实现
│   │   ├── modules/          # 编码器、生成器等网络模块
│   │   ├── renderer/         # 渲染器与网格转换
│   │   └── utils.py          # 网络相关工具函数
│   ├── solver/               # 优化器与训练相关工具
│   ├── utils/                # 通用工具（断点、可视化、libmesh等）
│   │   ├── libmesh/          # 网格相关底层实现（如三角形哈希、点内外判定）
│   │   └── ...               # 其它工具
│   └── ...                   # 其它核心代码
├── experiments/              # 实验配置（如 config.yaml）
├── tools/                    # 各类脚本工具
│   ├── _init_paths.py        # 路径初始化
│   ├── evaluate.py           # 评估脚本，输出指标与可视化
│   ├── preprocess.py         # 数据预处理，生成点云与占据标签
│   ├── train.py              # 训练主脚本
│   └── get_spe3r.sh          # 数据集自动下载与解压脚本
└── ...
```

### 主要文件/文件夹功能简述

- `core/configs/`：存放项目配置文件，定义训练、数据等参数。
- `core/dataset/`：数据集加载、构建、基类定义，支持多种数据格式。
- `core/nets/`：神经网络结构，包括主模型、损失函数、编码器、生成器、渲染器等。
- `core/nets/losses/`：各类损失函数实现，如 Chamfer 距离、占据损失、姿态损失等。
- `core/nets/modules/`：网络模块，如编码器、生成器、基础层。
- `core/nets/renderer/`：渲染器与网格转换工具。
- `core/solver/`：优化器、学习率调整等训练相关工具。
- `core/utils/`：通用工具，包括断点保存/加载、可视化、网格相关底层实现（libmesh）。
- `core/utils/libmesh/`：网格相关底层算法，如三角形哈希、点内外判定。
- `experiments/`：实验配置文件（如 config.yaml），可自定义实验参数。
- `tools/`：各类脚本工具，包括训练、评估、预处理、路径初始化等。
- `setup.py`：Cython 扩展编译脚本，主要用于加速网格相关底层操作。
- `pyproject.toml`/`poetry.lock`：依赖管理与环境配置。

## 更新日志

- [2024/03/18] 更新版数据集（v1.1）已在 [Stanford Digital Repository](https://purl.stanford.edu/pk719hm4806) 发布。本版本修正了初始数据集中姿态标签的错误，建议用户下载新版。详细更新日志请见新数据集内的 `UPDATES.md`。
- [2024/09/22] 预训练模型下载链接已更新。

## 简介

本项目为论文 [Rapid Abstraction of Spacecraft 3D Structure from Single 2D Image](https://arc.aiaa.org/doi/10.2514/6.2024-2768) 的官方 PyTorch 实现。

### 摘要

本文提出了一种卷积神经网络（CNN），可从单幅二维图像同时抽象目标空间驻留物体的三维结构并估算其姿态。具体而言，CNN 能够从目标的单幅图像预测一组单位尺寸的超二次体（superquadric）原语，这些原语仅需少量参数即可描述多种简单三维形状（如长方体、椭球体）。所提出的训练流程在二维和三维空间中采用多种监督方式，以拟合超二次体集合到人造卫星结构。为避免在评估超二次体时出现数值不稳定，本文提出了一种基于双超二次体的全新数值稳定算法，可在所有形状参数下评估超二次体表面及内部点。此外，为训练 CNN，本文还引入了 SPE3R 数据集，包含 64 个不同卫星模型及每个模型 1,000 张图像、二值掩码和姿态标签。实验结果表明，所提出的 CNN 能够在已知模型的未见图像上重建准确的超二次体集合，并且在极小数据集训练下，大多数情况下能捕捉未知模型的高层结构。

## 安装

本仓库使用 [poetry](https://python-poetry.org) 管理虚拟环境和依赖库。开发与测试环境为 Ubuntu 22.04，训练使用 NVIDIA GeForce RTX 4090 24GB GPU。

1. 安装 [poetry](https://python-poetry.org/docs/#installation)，并为 `python 3.10` 创建虚拟环境（详见 `pyproject.toml`）。

2. 通过以下命令安装依赖：

    ```bash
    poetry install
    ```

3. 单独安装 `kaleido`（用于 [plotly](https://plotly.com/python/) 可视化）：

    ```bash
    pip install -U kaleido
    ```

    目前 poetry 尚不支持 kaleido。

4. 编译扩展模块：

    ```bash
    python setup.py build_ext --inplace
    ```

5. [可选] 下载预训练模型（[链接](https://1drv.ms/f/c/fa28139a835eeb46/Evpp5SltMNNFqX_W26jaCzAB_UF6knvqKmkF-143sSAMVw)），该模型使用 `M = 8` 个原语训练，支持 RGH 和灰度图像输入。

## SPE3R 数据集

首先从 [Stanford Digital Repository](https://purl.stanford.edu/pk719hm4806) 获取 SPE3R 数据集，并放置于 `ROOT` 目录。可使用 `tools/get_spe3r.sh` 解压所有文件：

```bash
sh tools/get_spe3r.sh /path/to/dataset
```

注意：本数据集与论文实验所用数据集略有不同，验证集的模型组成有细微差异。

## 脚本说明

### 数据预处理

`tools/preprocess.py` 会为每个模型生成 100,000 个占据标签（`occupancy_points.npz`）和表面点（`surface_points.npz`），并保存这些点的可视化图像。

```bash
python tools/preprocess.py --cfg experiments/config.yaml
```

### 训练

训练前需设置以下变量：

| 变量名                      | 说明                       |
|----------------------------|----------------------------|
| `ROOT`                     | 仓库位置                   |
| `DATASET.ROOT`             | 数据集位置                 |
| `EXP_NAME`                 | 本次训练名称               |
| `MODEL.NUM_MAX_PRIMITIVES` | 原语数量（固定）           |

可通过修改 `LOSS.RECON_TYPE`、`LOSS.POSE_TYPE`、`LOSS.REG_TYPE`，选择不同的主监督损失、姿态损失和正则项。

训练命令如下：

```bash
python tools/train.py --cfg experiments/config.yaml
```

### 评估

以下命令会将输出的超二次体网格和输入图像保存到 `SAVE_DIR`。若未指定 `--modelidx` 和 `--imageidx`，则会在指定数据集划分中随机选择一个模型和图像进行评估。

```bash
SPLIT=validation
SAVE_DIR=figures
NUM_PRIM=8
PRETRAIN=output/model_m8.pth.tar

python tools/evaluate.py --cfg experiments/config.yaml --split ${SPLIT} --save_dir ${SAVE_DIR} \
        MODEL.NUM_MAX_PRIMITIVES ${NUM_PRIM} MODEL.PRETRAIN_FILE ${PRETRAIN}
```

## 许可证

本仓库采用 MIT 许可证（详见 `LICENSE.md`）。

## 参考文献

```
@inbook{park_2024_scitech_spe3r,
    author = {Park, Tae Ha and D'Amico, Simone},
    title = {Rapid Abstraction of Spacecraft 3D Structure from Single 2D Image},
    booktitle = {AIAA SCITECH 2024 Forum},
    chapter = {},
    pages = {},
    doi = {10.2514/6.2024-2768},
}
```
