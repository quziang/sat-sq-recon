
# 版权所有 (c) 2023 SLAB Group
# 作者: Tae Ha "Jeff" Park (tpark94@stanford.edu)
#
# 本脚本为模型训练主入口，包含训练、验证、日志记录、模型保存等流程。

import argparse
import time
import os.path as osp
from copy import deepcopy

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import torchvision

from pytorch3d.structures import Meshes

import _init_paths

from configs          import cfg, update_config
from nets             import Model
from dataset          import get_dataloader
from solver           import get_optimizer, adjust_learning_rate_step
from utils.utils      import AverageMeter, ProgressMeter
from utils.visualize  import *
from utils.checkpoint import save_checkpoint, load_checkpoint

from utils.utils import (
    set_seeds_cudnn,
    initialize_cuda,
    create_logger_directories,
    load_camera_intrinsics
)

torch.autograd.set_detect_anomaly(False)



# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='训练脚本')

    # 通用参数
    parser.add_argument('--cfg',
                        help='实验配置文件名',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="通过命令行修改配置项",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--world-size', default=1, type=int,
                        help='分布式训练节点数（默认1）')

    parser.add_argument('--rank', default=0, type=int,
                        help='分布式训练节点编号（默认0）')

    args = parser.parse_args()

    return args



# 训练主流程
def train(cfg):

    args = parse_args()  # 解析命令行参数
    update_config(cfg, args)  # 根据参数更新配置

    # ================= 基础设置 =================
    # 创建输出和日志目录
    logger, output_dir, log_dir = create_logger_directories(
        cfg, args.rank, phase='train', write_cfg_to_file=True
    )

    # 设置随机种子和 cudNN
    set_seeds_cudnn(cfg, seed=cfg.SEED)

    # 初始化 GPU 设备
    device = initialize_cuda(cfg, args.rank)

    # Tensorboard 日志
    if cfg.LOG_TENSORBOARD:
        tb_writer = SummaryWriter(log_dir)

    # ================= 构建模型 =================
    camera = load_camera_intrinsics(cfg.DATASET.CAMERA)  # 加载相机参数
    net    = Model(cfg, fov=camera['horizontalFOV'], device=device)  # 初始化模型

    # ================= 构建数据加载器 =================
    train_loader = get_dataloader(cfg, split='train')      # 训练集
    val_loader   = get_dataloader(cfg, split='validation') # 验证集

    # ================= 构建优化器 =================
    optimizer = get_optimizer(cfg, net)

    # 是否自动恢复断点
    checkpoint_file = osp.join(output_dir, f'checkpoint.pth.tar')
    if cfg.AUTO_RESUME and osp.exists(checkpoint_file):
        last_epoch = load_checkpoint(
                        checkpoint_file,
                        net,
                        optimizer,
                        None,
                        device)
        begin_epoch = last_epoch
    else:
        begin_epoch = cfg.TRAIN.BEGIN_EPOCH
        last_epoch  = -1

    # ================= 主训练循环 =================
    # 冻结渲染器参数（通常不参与训练）
    for param in net.renderer.parameters():
        param.requires_grad = False

    # 按 epoch 迭代
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):

        batch_time = AverageMeter('', 'ms', ':3.0f')  # 批次计时器

        # --- 损失统计器
        loss_train_meters = {}
        for l_name in net.loss_names:
            loss_train_meters[l_name] = AverageMeter(l_name, '', ':.2e')

        loss_val_meters = deepcopy(loss_train_meters)

        # --- 进度条
        progress_train = ProgressMeter(
            len(train_loader),
            batch_time,
            list(loss_train_meters.values()),
            prefix="Epoch {:03d} ".format(epoch+1))

        progress_val = ProgressMeter(
            len(val_loader),
            batch_time,
            list(loss_val_meters.values()),
            prefix="Epoch {:03d} ".format(epoch+1))

        # ========== 训练循环 ========== #
        net.train()
        for step, batch in enumerate(train_loader):

            # 动态调整学习率
            adjust_learning_rate_step(optimizer, epoch, step, len(train_loader), cfg)

            start = time.time()

            # 数据转移到 GPU
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # 前向传播
            optimizer.zero_grad(set_to_none=True)
            loss, sm = net(batch)

            # 反向传播与参数更新
            loss.backward()
            clip_grad_norm_(net.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()

            # 记录批次耗时
            batch_time.update((time.time() - start) * 1000)

            # 记录损失
            for k, v in sm.items():
                loss_train_meters[k].update(float(v), cfg.TRAIN.BATCH_SIZE_PER_GPU)

            # 控制台进度条
            progress_train.display(step+1, lr=optimizer.param_groups[0]['lr'])

        progress_train.display_summary()

        # ========== 验证循环 ========== #
        net.eval()
        if (epoch + 1) % cfg.TRAIN.VALID_FREQ == 0:
            for step, batch_val in enumerate(val_loader):

                # 数据转移到 GPU
                batch_val = {k: v.to(device, non_blocking=True) for k, v in batch_val.items()}

                # 验证损失
                with torch.no_grad():
                    loss, sm = net(batch_val)

                # 记录损失
                for k, v in sm.items():
                    loss_val_meters[k].update(float(v), cfg.TEST.BATCH_SIZE_PER_GPU)

                # 控制台进度条
                progress_val.display(step+1, lr=optimizer.param_groups[0]['lr'])

            progress_val.display_summary()

        # ========== Tensorboard 日志记录 ========== #
        if cfg.LOG_TENSORBOARD:
            for meter in loss_train_meters.values():
                tb_writer.add_scalar('Train/' + meter.name, meter.avg, epoch+1)

            if (epoch + 1) % cfg.TRAIN.VALID_FREQ == 0:
                for meter in loss_val_meters.values():
                    tb_writer.add_scalar('Validation/' + meter.name, meter.avg, epoch+1)

            # 输入图像可视化
            imgs = []
            for b in range(4):
                imgs.append(denormalize(batch["image"][b]))

            imgs = torchvision.utils.make_grid(imgs, nrow=2)
            tb_writer.add_image('Images/Input Images', imgs, epoch+1)

            # ========== 推理与可视化 ========== #
            with torch.no_grad():
                _, mesh, pcl, trans, rot = net.forward_encoder_generator(
                    batch["image"][:4], train=False
                )

                # 使用预测姿态渲染
                R = torch.bmm(rot.transpose(1, 2), net.Rz.repeat(rot.shape[0], 1, 1))
                silh = net.renderer(mesh, R=R, T=trans)

                # 分解 mesh，按原语可视化
                mesh_batch = []
                for b in range(4):
                    meshes = []
                    for i in range(cfg.MODEL.NUM_MAX_PRIMITIVES):
                        meshes.append(
                            Meshes(
                                verts=[pcl[b, :, i].detach().cpu()],
                                faces=[net.mesh_converter.faces.cpu()]
                            )
                        )

                    mesh_batch.append(meshes)

            # 绘制 mesh
            imgs = []
            tmp_fn = osp.join(log_dir, 'tmp.jpeg')
            for b in range(4):
                plot_3dmesh(mesh_batch[b], markers_for_vertices=False, savefn=tmp_fn)
                imgs.append(torchvision.io.read_image(tmp_fn))

            imgs = torchvision.utils.make_grid(imgs, nrow=2)
            tb_writer.add_image('Images/Reconstructed Assemblies', imgs, epoch+1)

            # 绘制投影结果
            imgs = silh[:4, ..., 3].unsqueeze(1).mul(255).clamp(0,255).byte().cpu()
            imgs = torchvision.utils.make_grid(imgs, nrow=2)
            tb_writer.add_image('Images/Projection of Predictions', imgs, epoch+1)

            # GT 掩码可视化
            imgs = torchvision.utils.make_grid(batch["mask"][:4].unsqueeze(1), nrow=2)
            tb_writer.add_image('Images/Ground-Truth Masks', imgs, epoch+1)

        # --- 保存断点
        if (epoch + 1) % cfg.TRAIN.VALID_FREQ == 0:
            save_checkpoint({
                'epoch': epoch+1,
                'state_dict': net.state_dict(),
                'best_state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, True, epoch+1 == cfg.TRAIN.END_EPOCH, output_dir)



# 程序主入口
if __name__ == "__main__":
    train(cfg)
