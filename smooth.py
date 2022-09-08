from main import *
import pdb
import igl

def smooth(config):
    model_cfg = Namespace(dim=3, out_dim=1,
            hidden_size=512,
            n_blocks=5,
            const=30.)

    module = SDFModule(cfg=model_cfg, f=config.init_ckpt).cuda()

    gt_sdf = torch.zeros(config.max_v, 1).cuda()
    F = torch.zeros(config.max_v, 1).cuda()
    vertices = torch.zeros((config.max_v, 3)).cuda()
    normals = torch.zeros((config.max_v, 3)).cuda()
    faces = torch.empty((config.max_v, 3), dtype=torch.int32).cuda()
    vertices.requires_grad_()

    for e in range(config.epochs):
        mesh_res = config.mesh_res_limit + np.random.randint(low=-3, high=3)
        with torch.no_grad():
            try:
                vertices_np, faces_np = module.get_zero_points(mesh_res=mesh_res)
            except:
                pdb.set_trace()
            L = igl.cotmatrix(vertices_np, faces_np)
            vel = torch.FloatTensor(L.dot(vertices_np)).cuda()
            v = vertices_np.shape[0]
            f = faces_np.shape[0]
            vertices.data[:v] = torch.from_numpy(vertices_np)
            faces.data[:f] = torch.from_numpy(np.ascontiguousarray(faces_np))

        with torch.no_grad():
            dE_dx = -vel

        idx = 0
        while idx < v:
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

        optimizer = torch.optim.Adam(list(module.parameters()), lr=config.lr)
        n_batches = v // config.batch_size

        for j in range(config.iters):
            optimizer.zero_grad()
            idx = 0
            while idx < v:
                min_i = idx
                max_i = min(min_i + config.batch_size, v)
                vertices_subset = vertices[min_i:max_i].detach()
                vertices_subset.requires_grad_()
                pred_sdf = module.forward(vertices_subset.unsqueeze(0)).squeeze(0)
                normals_subset = gradient(pred_sdf, vertices_subset).detach()
                eik = (normals_subset.norm(dim=-1) - 1).square().mean()
                loss = (torch.nan_to_num(gt_sdf[min_i:max_i]) - pred_sdf).square().mean() / (n_batches + 1)
                loss = loss + config.eik_lam * eik
                loss.backward()
                idx += config.batch_size

            optimizer.step()

        if e % config.mesh_log_freq == 0:
            print(f'Iter: {e}')
            mesh = trimesh.Trimesh(vertices_np, faces_np)
            mesh.export(f'{config.expdir}/mesh_{e:03d}.ply')

if __name__ == '__main__':
    config = parse_config(create_dir=True)
    smooth(config)
