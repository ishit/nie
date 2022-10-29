import numpy as np
import torch
import os
import sys
import trimesh
import torch
import torch.nn as nn

import nvdiffrast.torch as dr

from pytorch_lightning.utilities.seed import seed_everything
from PIL import Image
import utils

seed_everything(42)

class Renderer:
    def __init__(self, num_views, res, fname=None, scale=1.75):
        self.num_views = num_views
        self.num_views = num_views
        self.res = res
        self.r_mvp = []
        self.r_campos = []
        self.mvs = []
        self.lightdir = []

        self.glctx = dr.RasterizeGLContext()
        self.zero_tensor = torch.as_tensor(0.0, dtype=torch.float32, device='cuda')
        proj  = utils.projection(x=0.5, n=1.5, f=100.0)
        self.fov_x = np.rad2deg(2 * np.arctan(0.5))

        t = utils.translate(0, 0, 4.)
        e = 1.5/0.5
        focal_length = (res / 2) / (1 / e)

        self.intrinsics = np.array([[focal_length, 0., res/2],
                                [0., focal_length, res/2],
                                [0., 0., 1.]])

        self.albedo = 0.55
        self.scale = scale
        self.rots = []

        for i in range(num_views):
            r_rot = utils.random_rotation()
            r_mv  = np.matmul(utils.translate(0, 0, -4.), r_rot)
            self.mvs.append(r_mv)
            r_mvp = np.matmul(proj, r_mv).astype(np.float32)
            self.r_mvp.append(r_mvp)
            r_campos = torch.as_tensor(np.linalg.inv(r_mv)[:3, 3], dtype=torch.float32, device='cuda')
            lightdir = -r_campos / torch.norm(r_campos)
            self.lightdir.append(lightdir)

        proj = torch.as_tensor(proj, dtype=torch.float32, device='cuda')
        self.view_mats = torch.as_tensor(np.array(self.mvs), dtype=torch.float32, device='cuda')
        self.lightdir = torch.stack(self.lightdir)
        self.mvps = proj @ self.view_mats
        self.render_target(fname)

    def render_target(self, fname):
        # Load Mesh
        if fname is not None:
            mesh = trimesh.load_mesh(fname)
        else:
            mesh = trimesh.load_mesh('data/bunny.obj')

        mean = np.mean(mesh.vertices, axis=0, keepdims=True)
        mesh.vertices -= mean
        scale = self.scale / (np.max(mesh.vertices) - np.min(mesh.vertices))
        mesh.vertices *= scale
        self.mesh = mesh

        normals = mesh.vertex_normals
        normals = torch.as_tensor(normals, dtype=torch.float32,\
                device='cuda').contiguous()

        v = torch.as_tensor(mesh.vertices, dtype=torch.float32, device='cuda').contiguous()
        f = torch.as_tensor(mesh.faces, dtype=torch.int32,\
                device='cuda').contiguous()

        self.target_imgs = self.render(v, f, normals)

    def render_pointlight(self, pos, pos_idx, normals):
        v_hom = torch.nn.functional.pad(pos, (0,1), 'constant', 1.0)
        v_ndc = torch.matmul(v_hom, self.mvps.transpose(1,2))
        rast = dr.rasterize(self.glctx, v_ndc, pos_idx, [self.res,
            self.res])[0]
        v_cols = torch.zeros_like(pos)

        pixel_normals = dr.interpolate(normals[None, ...], rast, pos_idx)[0]
        diffuse = self.albedo * torch.sum(-self.lightdir.view(-1, 1, 1, 3) * pixel_normals, -1, keepdim=True)
        result = dr.antialias(torch.where(rast[..., -1:] != 0, diffuse,
            self.zero_tensor),
                    rast, v_ndc, pos_idx)

        return torch.nan_to_num(result)

    def render(self, pos, pos_idx, normals):
        return self.render_pointlight(pos, pos_idx, normals)

if __name__ == '__main__':
    R = Renderer(10, 1024)
    for i in range(R.target_imgs.shape[0]):
        utils.save_image(f'data/{i:06d}.png', R.target_imgs[i, :, :, :3].detach().cpu().numpy())
