
# 版权所有 (c) 2023 SLAB Group
# 作者: Tae Ha "Jeff" Park (tpark94@stanford.edu)
#
# 本脚本用于模型评估，包括推理、指标计算和结果可视化。

import argparse
import numpy as np
import random
from pathlib import Path
from scipy.io import savemat

import torch
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.transforms import so3_relative_angle
from pytorch3d.loss       import chamfer_distance

import _init_paths

from configs          import cfg, update_config
from dataset.build    import build_dataset
from nets             import Model
from nets.utils       import transform_world2primitive, inside_outside_function_dual
from utils.visualize  import plot_3dmesh, plot_3dpoints, plot_occupancy_labels, imshow
from utils.utils      import (
    set_seeds_cudnn,
    initialize_cuda,
    load_camera_intrinsics
)



# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='评估脚本参数解析')

    # 通用参数
    parser.add_argument('--cfg',
                        help='实验配置文件名',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="通过命令行修改配置项",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--rank', default=0, type=int,
                        help='分布式训练节点编号（默认0）')

    parser.add_argument('--split', default='test', const='test', nargs='?',
                        choices=['train', 'validation', 'test'],
                        help='评估用数据集划分')

    parser.add_argument('--save_dir', type=str)

    parser.add_argument('--modelidx', type=int)
    parser.add_argument('--imageidx', type=int)

    args = parser.parse_args()

    return args



# 推理函数，执行模型前向推理并计算各类评估指标
def inference(args, cfg, net, batch):

    with torch.no_grad():
        # 推理模式下，mesh 只包含高概率原语，点云包含全部原语
        params, mesh, pcls, trans, rot = net.forward_encoder_generator(batch["image"], train=False)

        # 占据函数，判断点是否在预测模型内部
        occupancy = net.occupancy_function(batch["points_in_mesh"], params) # Fbar
        occupancy_pr = occupancy.sigmoid().cpu() >= 0.5

        # 差分渲染，生成预测掩码
        R       = torch.bmm(rot.transpose(1, 2), net.Rz[0].unsqueeze(0))
        silh    = net.renderer(mesh, R=R, T=trans)
        mask_pr = silh[0, ..., 3].cpu()

        # 找出属于原语表面的点
        B, N, M, _ = pcls.shape
        pts_prim = transform_world2primitive(
            pcls.view(B, M * N, -1), params._translation, params._rotation, is_dcm=True
        ) # [B x (N x M) x M x 3]

        # 二值掩码，判断点是否在所有原语表面
        F = inside_outside_function_dual(pts_prim, params) # [B x (N x M) x M], < 1 = inside

        # F >= 1 表示点在表面或在所有表面外部
        F = F.view(B, N, M, M)
        on_surface = (F >= 0.9).all(-1) # [B x N x M]

        # 可视化用：为每个原语分别生成点云对象
        pcl_pr = []
        for i in range(cfg.MODEL.NUM_MAX_PRIMITIVES):
            pcl_pr.append(
                Pointclouds(
                    [pcls[0, on_surface[0,:,i], i].cpu()]
                )
            )

        # 可视化用：为每个原语分别生成 mesh 对象
        mesh_pr = []
        for i in range(cfg.MODEL.NUM_MAX_PRIMITIVES):
            mesh_pr.append(
                Meshes(
                    verts=[pcls[0, :, i]],
                    faces=[net.mesh_converter.faces]
                ).cpu()
            )

        # 评估指标
        metrics = {}
        with torch.no_grad():

            # 姿态误差
            metrics['eR'] = so3_relative_angle(rot, batch["rot"])[0].cpu().numpy() # [弧度]
            metrics['eT'] = torch.linalg.norm(trans[0] - batch["trans"][0]).cpu().numpy()

            # SPEED 分数
            metrics['speed'] = metrics['eR'] + metrics['eT'] / torch.linalg.norm(batch["trans"][0]).cpu().numpy()

            # Chamfer 距离
            chamfer_l2, _ = chamfer_distance(
                mesh.verts_padded(), batch["points_on_mesh"], point_reduction='mean', norm=2
            )

            chamfer_l1, _ = chamfer_distance(
                mesh.verts_padded(), batch["points_on_mesh"], point_reduction='mean', norm=1
            )

            metrics['chamfer_l2'] = chamfer_l2.cpu().numpy()
            metrics['chamfer_l1'] = chamfer_l1.cpu().numpy()

            # 3D 体素 IoU
            occ_pred = occupancy[0].any(dim=-1).sigmoid().cpu().numpy() >= 0.5
            occ_true = batch["occ_labels"].cpu().numpy() >= 0.5

            intersection = occ_pred & occ_true
            union        = occ_pred | occ_true
            metrics['iou_3d'] = intersection.sum() / float(union.sum())

            # 原语数量
            metrics['num_prims'] = sum(params._prob.squeeze().cpu().numpy() > 0.5)

            # 2D IoU
            y_pred = mask_pr.numpy() > 0.5
            y_true = batch['mask'][0].cpu().numpy() > 0.5

            intersection = y_pred & y_true
            union        = y_pred | y_true

            metrics['iou_2d'] = intersection.sum() / float(union.sum())

    return mesh_pr, pcl_pr, mask_pr, occupancy_pr, metrics



# 评估主流程
def evaluate(cfg):

    args = parse_args()  # 解析命令行参数
    update_config(cfg, args)  # 更新配置

    print(f"评估数据集划分: {args.split.upper()}")

    # 设置随机种子和 cudNN
    set_seeds_cudnn(cfg, seed=None)

    # 初始化 GPU 设备
    device = initialize_cuda(cfg, args.rank)

    # 结果保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ================= 构建模型 =================
    camera = load_camera_intrinsics(cfg.DATASET.CAMERA)
    net    = Model(cfg, fov=camera["horizontalFOV"], device=device)

    # 加载预训练模型参数
    load_dict = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location=device)
    net.load_state_dict(load_dict, strict=True)

    # ================= 构建数据集 =================
    dataset = build_dataset(cfg, split=args.split)

    # ================= 推理与结果获取 =================
    net.eval()

    # 随机选择模型和图像索引（如未指定）
    if args.modelidx is None:
        args.modelidx = random.randrange(dataset.num_models)

    if args.imageidx is None:
        args.imageidx = random.randrange(dataset.num_images_per_model)

    # 获取数据 batch
    batch = dataset._get_item(args.modelidx, imgidx=args.imageidx)

    # 数据转移到 GPU
    batch = {k: v.unsqueeze(0).to(device, non_blocking=True) for k, v in batch.items()}

    # 执行推理
    mesh_pr, pcl_pr, mask_pr, occupancy_pr, metrics = inference(args, cfg, net, batch)

    # ---------- 可视化与保存结果 ----------
    # 输入图像
    imshow(batch["image"][0], is_tensor=True, savefn=str(save_dir / "image.jpg"))

    # 预测 mesh
    plot_3dmesh(mesh_pr, markers_for_vertices=False, savefn=str(save_dir / "mesh.jpg"))

    # 保存评估指标
    savemat(str(save_dir / "metrics.mat"), metrics)

    print(f"E_T: {metrics['eT']:.2f} [m]    E_R: {np.rad2deg(metrics['eR']):.2f} [deg]    Chamfer-L1 (E-3): {metrics['chamfer_l1'] * 1000:.2f}    Num. prim.: {metrics['num_prims']}")



# 程序主入口
if __name__=="__main__":
    evaluate(cfg)