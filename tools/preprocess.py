
# 版权所有 (c) 2023 SLAB Group
# 作者: Tae Ha "Jeff" Park (tpark94@stanford.edu)
#
# 本脚本用于从三维网格生成点云和占据标签，并保存可视化结果。

import numpy as np
import argparse
import trimesh
from tqdm import tqdm

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Pointclouds

import _init_paths

from configs         import cfg, update_config
from dataset.build   import build_dataset
from utils.visualize import *
from utils.libmesh   import check_mesh_contains



# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='shapeExtractionNet 参数解析器')

    # 通用参数
    parser.add_argument('--cfg',
                        help='实验配置文件名',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="通过命令行修改配置项",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args



# 主程序入口
if __name__=='__main__':

    args = parse_args()  # 解析命令行参数
    update_config(cfg, args)  # 更新配置

    # 不使用 splits.csv，处理所有模型
    cfg.defrost()
    cfg.DATASET.SPLIT_CSV = None
    cfg.freeze()

    # ========== 创建数据集结构 ==========
    dataset = build_dataset(cfg, 'train')

    # ========== 从网格表面采样点云 ==========
    N = 100000  # 每个模型采样点数
    for idx in tqdm(range(dataset.num_models)):
        d = dataset.datasets[idx]

        # ---------- (1) 从网格表面采样点 ----------
        points = sample_points_from_meshes(d.mesh, num_samples=N) # [1 x N x 3]
        np.savez(str(d.path_to_surface_points), points=points.numpy())  # 保存点云

        # 可视化点云并保存图片
        plot_3dpoints(
            Pointclouds(points),
            savefn=str(d.path_to_surface_points).replace('npz', 'jpg')
        )

        # ---------- (2) 从网格内部采样点并判断占据 ----------
        mesh_gt = trimesh.load(d.path_to_mesh_file, force='mesh')  # 加载网格

        # 可选：检查网格是否封闭
        # assert mesh_gt.is_watertight, f'Model {d.tag} is not watertight'

        points = np.random.rand(N, 3) - 0.5  # 随机采样点
        occupancy = check_mesh_contains(mesh_gt, points)  # 判断点是否在网格内部
        np.savez(str(d.path_to_occupancy), points=points, labels=occupancy)  # 保存占据标签

        # 可视化占据标签并保存图片
        plot_occupancy_labels(
            points, occupancy,
            savefn=str(d.path_to_occupancy).replace('npz', 'jpg')
        )

        # # ---------- 可视化指定模型（示例）
        # if d.tag == 'chandra_v09':
        #     with np.load(d.path_to_surface_points) as data:
        #         points = data['points']
        #     plot_3dpoints(Pointclouds(torch.from_numpy(points)))


