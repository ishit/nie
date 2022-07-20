import yaml
import os
import math
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import scipy.fftpack
import time
import plyfile
import skimage.measure
from scipy.spatial import cKDTree as KDTree
import trimesh
import torch.nn as nn
from torch.nn import functional as F
from sklearn.neighbors import NearestNeighbors
import redner
import matplotlib.pyplot as plt
import argparse

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def dir_counter(LOGDIR):
    return len([name for name in os.listdir(LOGDIR)])

def dict2namespace(config):
    if isinstance(config, argparse.Namespace):
        return config
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def parse_config(suffix='', create_dir=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Config file')
    parser.add_argument('--test_mesh', type=str, help='Test Mesh', default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = dict2namespace(config)
    dir_count = str(dir_counter(config.logdir)) + '_' + config.exp + suffix
    config.expdir = config.logdir + '/' + dir_count
    config.test_mesh = args.test_mesh
    if create_dir:
        os.makedirs(config.expdir, exist_ok=True)
    return config

def compute_trimesh_chamfer(gt_mesh, gen_mesh, num_mesh_samples=100000):
    gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]
    gt_points_np = trimesh.sample.sample_surface(gt_mesh, num_mesh_samples)[0]

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer

def compute_edges(vertices, faces, return_faces=False):
    v0, v1, v2 = faces.chunk(3, dim=1)
    e01 = torch.cat([v0, v1], dim=1)
    e12 = torch.cat([v1, v2], dim=1)
    e20 = torch.cat([v2, v0], dim=1)
    edges = torch.cat([e12, e20, e01], dim=0).long()
    edges, _ = edges.sort(dim=1)
    V = vertices.shape[0]
    edges_hash = V * edges[:, 0] + edges[:, 1]
    u, inverse_idxs = torch.unique(edges_hash,\
                                    return_inverse=True)
    sorted_hash, sort_idx = torch.sort(edges_hash, dim=0)
    unique_mask = torch.ones(
        edges_hash.shape[0], dtype=torch.bool, device=vertices.device
    )
    unique_mask[1:] = sorted_hash[1:] != sorted_hash[:-1]
    unique_idx = sort_idx[unique_mask]
    edges = torch.stack([u // V, u % V], dim=1)

    if return_faces:
        faces = inverse_idxs.reshape(3, faces.shape[0]).t()
        return edges.long(), faces.long()
    return edges.long()

def laplacian_simple(verts: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    V = verts.shape[0]
    e0, e1 = edges.unbind(1)
    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    deg0 = torch.where(deg0 > 0.0, -deg0*0 - 1, deg0)
    deg1 = deg[e1]
    deg1 = torch.where(deg1 > 0.0, -deg1*0 - 1, deg1)
    val = torch.cat([deg0, deg1])
    L = torch.sparse.FloatTensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = -1.
    idx = torch.arange(V, device=verts.device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device) *\
                deg[idx[0]]
    L += torch.sparse.FloatTensor(idx, ones, (V, V))

    return L

def remove_duplicates(v, f):
    """
    Generate a mesh representation with no duplicates and
    return it along with the mapping to the original mesh layout.
    """

    unique_verts, inverse = torch.unique(v, dim=0, return_inverse=True)
    new_faces = inverse[f.long()]
    return unique_verts, new_faces, inverse

def average_edge_length(verts, faces):
    """
    Compute the average length of all edges in a given mesh.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    return (A + B + C).sum() / faces.shape[0] / 3

def massmatrix_voronoi(verts, faces):
    """
    Compute the area of the Voronoi cell around each vertex in the mesh.
    http://www.alecjacobson.com/weblog/?p=863

    params
    ------

    v: vertex positions
    f: triangle indices
    """
    # Compute edge lengths
    l0 = (verts[faces[:,1]] - verts[faces[:,2]]).norm(dim=1)
    l1 = (verts[faces[:,2]] - verts[faces[:,0]]).norm(dim=1)
    l2 = (verts[faces[:,0]] - verts[faces[:,1]]).norm(dim=1)
    l = torch.stack((l0, l1, l2), dim=1)

    # Compute cosines of the corners of the triangles
    cos0 = (l[:,1].square() + l[:,2].square() - l[:,0].square())/(2*l[:,1]*l[:,2])
    cos1 = (l[:,2].square() + l[:,0].square() - l[:,1].square())/(2*l[:,2]*l[:,0])
    cos2 = (l[:,0].square() + l[:,1].square() - l[:,2].square())/(2*l[:,0]*l[:,1])
    cosines = torch.stack((cos0, cos1, cos2), dim=1)

    # Convert to barycentric coordinates
    barycentric = cosines * l
    barycentric = barycentric / torch.sum(barycentric, dim=1)[..., None]

    # Compute areas of the faces using Heron's formula
    areas = 0.25 * ((l0+l1+l2)*(l0+l1-l2)*(l0-l1+l2)*(-l0+l1+l2)).sqrt()

    # Compute the areas of the sub triangles
    tri_areas = areas[..., None] * barycentric

    # Compute the area of the quad
    cell0 = 0.5 * (tri_areas[:,1] + tri_areas[:, 2])
    cell1 = 0.5 * (tri_areas[:,2] + tri_areas[:, 0])
    cell2 = 0.5 * (tri_areas[:,0] + tri_areas[:, 1])
    cells = torch.stack((cell0, cell1, cell2), dim=1)

    # Different formulation for obtuse triangles
    # See http://www.alecjacobson.com/weblog/?p=874
    cells[:,0] = torch.where(cosines[:,0]<0, 0.5*areas, cells[:,0])
    cells[:,1] = torch.where(cosines[:,0]<0, 0.25*areas, cells[:,1])
    cells[:,2] = torch.where(cosines[:,0]<0, 0.25*areas, cells[:,2])

    cells[:,0] = torch.where(cosines[:,1]<0, 0.25*areas, cells[:,0])
    cells[:,1] = torch.where(cosines[:,1]<0, 0.5*areas, cells[:,1])
    cells[:,2] = torch.where(cosines[:,1]<0, 0.25*areas, cells[:,2])

    cells[:,0] = torch.where(cosines[:,2]<0, 0.25*areas, cells[:,0])
    cells[:,1] = torch.where(cosines[:,2]<0, 0.25*areas, cells[:,1])
    cells[:,2] = torch.where(cosines[:,2]<0, 0.5*areas, cells[:,2])

    # Sum the quad areas to get the voronoi cell
    return torch.zeros_like(verts).scatter_add_(0, faces, cells).sum(dim=1)

def compute_face_normals(verts, faces):
    """
    Compute per-face normals.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions
    faces : torch.Tensor
        Triangle faces
    """
    fi = torch.transpose(faces, 0, 1).long()
    verts = torch.transpose(verts, 0, 1)

    v = [verts.index_select(1, fi[0]),
                 verts.index_select(1, fi[1]),
                 verts.index_select(1, fi[2])]

    c = torch.cross(v[1] - v[0], v[2] - v[0])
    n = c / torch.norm(c, dim=0)
    return n

def safe_acos(x):
    return torch.acos(x.clamp(min=-1, max=1))

def compute_vertex_normals(verts, faces, face_normals):
    """
    Compute per-vertex normals from face normals.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions
    faces : torch.Tensor
        Triangle faces
    face_normals : torch.Tensor
        Per-face normals
    """
    fi = torch.transpose(faces, 0, 1).long()
    verts = torch.transpose(verts, 0, 1)
    normals = torch.zeros_like(verts)

    v = [verts.index_select(1, fi[0]),
             verts.index_select(1, fi[1]),
             verts.index_select(1, fi[2])]

    for i in range(3):
        d0 = v[(i + 1) % 3] - v[i]
        d0 = d0 / torch.norm(d0)
        d1 = v[(i + 2) % 3] - v[i]
        d1 = d1 / torch.norm(d1)
        d = torch.sum(d0*d1, 0)
        face_angle = safe_acos(torch.sum(d0*d1, 0))
        nn =  face_normals * face_angle
        for j in range(3):
            normals[j].index_add_(0, fi[i], nn[j])
    return (normals / torch.norm(normals, dim=0)).transpose(0, 1)

from scipy.spatial.transform import Rotation as R

#----------------------------------------------------------------------------
# Projection and transformation matrix helpers.
#----------------------------------------------------------------------------

def projection(x=0.1, n=1.0, f=50.0):
    return np.array([[n/x,    0,            0,              0],
                     [  0, n/-x,            0,              0],
                     [  0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                     [  0,    0,           -1,              0]]).astype(np.float32)

def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]]).astype(np.float32)

def rotate_x(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[1,  0, 0, 0],
                     [0,  c, s, 0],
                     [0, -s, c, 0],
                     [0,  0, 0, 1]]).astype(np.float32)

def rotate_y(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]]).astype(np.float32)

def random_rotation():
    r = R.random().as_matrix()
    r = np.hstack([r, np.zeros((3, 1))])
    r = np.vstack([r, np.zeros((1, 4))])
    r[3, 3] = 1
    return r

def random_rotation_translation(t):
    m = np.random.normal(size=[3, 3])
    m = np.identity(3)
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.random.uniform(-t, t, size=[3])
    return m

#----------------------------------------------------------------------------
# Bilinear downsample by 2x.
#----------------------------------------------------------------------------

def bilinear_downsample(x):
    w = torch.tensor([[1, 3, 3, 1], [3, 9, 9, 3], [3, 9, 9, 3], [1, 3, 3, 1]], dtype=torch.float32, device=x.device) / 64.0
    w = w.expand(x.shape[-1], 1, 4, 4) 
    x = torch.nn.functional.conv2d(x.permute(0, 3, 1, 2), w, padding=1, stride=2, groups=x.shape[-1])
    return x.permute(0, 2, 3, 1)


def save_image(fn, x):
    import imageio
    x = np.rint(x * 255.0)
    x = np.clip(x, 0, 255).astype(np.uint8)
    imageio.imsave(fn, x)
