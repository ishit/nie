import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from models import SDFModule
from render import Renderer
from utils import *

def gauss_kernel(size=5, device=torch.device('cuda:0'), channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel.cuda(), groups=img.shape[1])
    return out

def img_loss(imgs, target_imgs, multi_scale=True):
    loss = 0
    kernel = gauss_kernel()
    count = 0
    for i in range(imgs.shape[0]):
        count += 1
        loss = loss + (imgs[i] - target_imgs[i]).square().mean()
        if multi_scale:
            current_est = imgs[i].permute(2, 0, 1).unsqueeze(0)
            current_gt = target_imgs[i].permute(2, 0, 1).unsqueeze(0)
            for j in range(4):
                filtered_est = conv_gauss(current_est, kernel)
                filtered_gt = conv_gauss(current_gt, kernel)
                down_est = downsample(filtered_est)
                down_gt = downsample(filtered_gt)

                current_est = down_est
                current_gt = down_gt

                loss = loss + (current_est - current_gt).square().mean() / (j+1)

    loss = loss / count
    return loss

def main(config):
    model_cfg = Namespace(dim=3, out_dim=1,
            hidden_size=512,
            n_blocks=4, z_dim=1,
            const=60.)
    module = SDFModule(cfg=model_cfg, f=config.init_ckpt).cuda()
    logger = SummaryWriter(log_dir=config.expdir, flush_secs=5)

    with torch.no_grad():
        R = Renderer(config.num_views, config.res, fname=config.mesh,
                scale=config.scale)
        target_imgs = R.target_imgs

    logger.add_image('target', target_imgs[-1].permute(2, 0, 1).clamp(0, 1))
    optimizer = torch.optim.Adam(list(module.parameters()), lr=config.lr,
            weight_decay=config.weight_decay)

    gt_sdf = torch.zeros(config.max_v, 1).cuda()
    F = torch.zeros(config.max_v, 1).cuda()
    vertices = torch.zeros((config.max_v, 3)).cuda()
    normals = torch.zeros((config.max_v, 3)).cuda()
    faces = torch.empty((config.max_v, 3), dtype=torch.int32).cuda()
    vertices.requires_grad_()

    for e in range(config.epochs):
        laplace_lam = config.max_laplace_lam

        if e >= config.fine_e:
            laplace_lam = config.min_laplace_lam
            mesh_res = config.mesh_res_limit + np.random.randint(low=-3, high=3)

        else:
            laplace_lam = config.max_laplace_lam
            mesh_res = config.mesh_res_base + np.random.randint(low=-3, high=3)

        with torch.no_grad():
            vertices_np, faces_np = module.get_zero_points(mesh_res=mesh_res)

            v = vertices_np.shape[0]
            f = faces_np.shape[0]

            vertices.data[:v] = torch.from_numpy(vertices_np)
            faces.data[:f] = torch.from_numpy(np.ascontiguousarray(faces_np))

        vertices.grad = None

        edges = compute_edges(vertices[:v], faces[:f])
        L = laplacian_simple(vertices[:v], edges.long())
        laplacian_loss = torch.trace(((L @ vertices[:v]).T @ vertices[:v]))

        face_normals = compute_face_normals(vertices[:v], faces[:f])
        vertex_normals = compute_vertex_normals(vertices[:v], faces[:f], face_normals)
        imgs = R.render(vertices[:v], faces[:f], vertex_normals)
        loss = img_loss(imgs, target_imgs, multi_scale=True)
        loss = loss + laplace_lam * laplacian_loss

        loss.backward()
        logger.add_scalar('loss', loss.item(), global_step=e)

        with torch.no_grad():
            dE_dx = vertices.grad[:v].detach()

        idx = 0
        while idx < v:
            optimizer.zero_grad()
            min_i = idx
            max_i = min(min_i + config.batch_size, v)
            vertices_subset = vertices[min_i:max_i]
            vertices_subset.requires_grad_()
            pred_sdf = module.forward(vertices_subset.unsqueeze(0)).squeeze(0)
            normals[min_i:max_i] = gradient(pred_sdf, vertices_subset).detach()
            F[min_i:max_i] = torch.nan_to_num(torch.sum(normals[min_i:max_i] *\
                dE_dx[min_i:max_i], dim=-1, keepdim=True))
            gt_sdf[min_i:max_i] = (pred_sdf + config.eps * F[min_i:max_i]).detach()
            idx += config.batch_size

        n_batches = v // config.batch_size

        for j in range(config.iters):
            optimizer.zero_grad()

            idx = 0
            while idx < v:
                min_i = idx
                max_i = min(min_i + config.batch_size, v)
                vertices_subset = vertices[min_i:max_i].detach()
                pred_sdf = module.forward(vertices_subset.unsqueeze(0)).squeeze(0)
                loss = (gt_sdf[min_i:max_i] - pred_sdf).abs().mean() / n_batches
                loss.backward()
                idx += config.batch_size

            optimizer.step()

        if e % 1 == 0:
            print(f'Iter: {e}')

        if e % config.img_log_freq == 0:
            logger.add_image('est', imgs[-1].permute(2, 0, 1).clamp(0, 1), global_step=e)

        if e % config.mesh_log_freq == 0:
            with torch.no_grad():
                mse = (imgs-target_imgs).square().mean()
                psnr = -10.0*torch.log10(mse)
                logger.add_scalar('psnr', psnr, global_step=e)
            mesh = trimesh.Trimesh(vertices_np, faces_np)
            cd = compute_trimesh_chamfer(R.mesh, mesh)
            logger.add_scalar('cd', cd, global_step=e)
            mesh.export(f'{config.expdir}/mesh_{e:04d}.ply')

        if e % config.ckpt_log_freq == 0:
            torch.save(module.state_dict(), f'{config.expdir}/iter_{e:04d}.ckpt')

if __name__ == '__main__':
    config = parse_config(create_dir=False)
    main(config)
