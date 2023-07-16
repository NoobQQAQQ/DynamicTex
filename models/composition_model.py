import os
import torch
import torch.nn as nn
import numpy as np
import open3d
from PIL import Image

from .base_model import BaseModel
from .static_nerf import StaticNerf
from .dynamic_sdf import DynamicSdf
from .atlasnet import AtlasNet, InverseAtlasNet
from .density import LaplaceDensity
from .texture_net import TextureNet
from utils.network_helpers import positional_encoding
from utils.loss_helpers import NDC2world
from utils.cube_map import (
    generate_grid,
    convert_cube_uv_to_xyz,
    load_cube_from_single_texture,
    merge_cube_to_single_texture
)


class CompositionModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        self.input_ch = 3 + 6 * args.multires
        self.input_ch_time = 1 + 2 * args.multires_time
        self.input_ch_views = 3 + 6 * args.multires_views
        self.beta = args.beta_density

        self.use_viewdirs = args.use_viewdirs
        self.static_D = args.static_depth
        self.static_W = args.static_width
        self.static_skips = [4]

        self.n_eik_points = args.chunk * args.N_samples
        self.dynamic_D = args.dynamic_depth
        self.dynamic_W = args.dynamic_width
        self.dynamic_skips = [4]

        self.latent_code_dim = args.latent_code_dim
        self.uv_dim = args.uv_dim
        self.tex_input_ch = self.uv_dim * (1 + 2 * args.multires)

        self.atlas_D = args.atlas_depth
        self.atlas_W = args.atlas_width
        self.atlas_skips = [1]

        self.inverse_atlas_D = args.inverse_atlas_depth
        self.inverse_atlas_W = args.inverse_atlas_width
        self.inverse_atlas_skips = [1]

        self.texture_D = args.texture_depth
        self.texture_W = args.texture_width
        self.texture_skips = [2]

        self.resolution = args.resolution

        self.net_static = StaticNerf(D=self.static_D,
                                     W=self.static_W,
                                     input_ch=self.input_ch,
                                     input_ch_views=self.input_ch_views,
                                     skips=self.static_skips,
                                     use_viewdirs=self.use_viewdirs)

        self.net_dynamic = DynamicSdf(D=self.dynamic_D,
                                      W=self.dynamic_W,
                                      input_ch=self.input_ch,
                                      input_ch_time=self.input_ch_time,
                                      skips=self.dynamic_skips)

        self.net_density = LaplaceDensity(self.beta)

        self.net_time_embedding = nn.Linear(self.input_ch_time, self.latent_code_dim)

        # no positional encoding for smooth mapping (following the practice of NeuTex)
        self.net_atlas = AtlasNet(D=self.atlas_D,
                                  W=self.atlas_W,
                                  input_ch=3,
                                  skips=self.atlas_skips,
                                  latent_code_dim=self.latent_code_dim,
                                  uv_dim=self.uv_dim)

        self.net_inverse_atlas = InverseAtlasNet(D=self.inverse_atlas_D,
                                                 W=self.inverse_atlas_W,
                                                 input_ch=3,
                                                 skips=self.inverse_atlas_skips,
                                                 latent_code_dim=self.latent_code_dim)

        self.net_texture = TextureNet(D=self.texture_D,
                                      W=self.texture_W,
                                      uv_dim=self.uv_dim,
                                      input_ch=self.tex_input_ch,
                                      skips=self.texture_skips)

        self.network_names.append('static')
        self.network_names.append('dynamic')
        self.network_names.append('density')
        self.network_names.append('time_embedding')
        self.network_names.append('atlas')
        self.network_names.append('inverse_atlas')
        self.network_names.append('texture')

        if self.is_train:
            grad_vars = []
            for net in self.get_networks():
                grad_vars += net.parameters()
            # create optimizer
            self.optimizer = torch.optim.Adam(params=grad_vars, lr=args.lr, betas=(0.9, 0.999))
            if args.no_reload:
                # train from scratch
                self.global_step = 1  # index start from 1
            else:
                # resume from some point, load weight and optimizer status
                assert self.resume_dir
                self.load_model(args.resume_from)
        else:
            # only load weights
            assert self.resume_dir
            self.load_model(args.resume_from)
            for net in self.get_networks():
                for param in net.parameters():
                    param.requires_grad = False
            self.global_step = None
            self.optimizer = None

    def forward(self, pts, time, viewdirs=None):
        #  outputs shape:
        #    if train: N_pts x (4+4+1+6+1+3) (rgbd1 + rgbd2 + b + sf + sdf + inverse_pts + uv_pts)
        #        else: N_pts x (4+4+1+6) (rgbd1 + rgbd2 + b + sf)

        # do positional encoding
        pts = positional_encoding(pts.view(-1, 3), self.args.multires)
        time = positional_encoding(torch.Tensor([time]), self.args.multires_time).view(1, -1)
        if viewdirs is not None:
            viewdirs = viewdirs[:, None, :].expand(viewdirs.shape[0], self.args.N_samples, viewdirs.shape[-1])
            viewdirs = positional_encoding(viewdirs.reshape(-1, 3), self.args.multires_views)
        #      pts shape: N_pts x input_ch
        # viewdirs shape: N_pts x input_ch_views
        #     time shape: 1 x input_ch_time

        static_rgb, static_density = self.net_static(pts, viewdirs)
        dynamic_sdf, scene_flow, blending = self.net_dynamic(pts, time)
        dynamic_density = self.net_density(dynamic_sdf)

        latent_code = self.net_time_embedding(time)
        uv = self.net_atlas(pts[..., 0:3], latent_code)  # N_pts x (2 or 3)

        uv = positional_encoding(uv, self.args.multires)
        dynamic_rgb = self.net_texture(uv)

        if self.is_train:
            if self.uv_dim == 2:
                minus_1 = -torch.ones(uv.shape[0], 1)
                uv = torch.cat([uv[..., 0:2], minus_1], dim=-1)
            else:
                uv = uv[..., 0:3]
            inverse_pts = self.net_inverse_atlas(uv, latent_code)  # N_pts x 3
            return torch.cat([static_rgb, static_density, dynamic_rgb, dynamic_density, blending,
                              scene_flow, inverse_pts, uv[..., 0:self.uv_dim], dynamic_sdf], dim=-1)
            # return torch.cat([static_rgb, static_density, dynamic_rgb, dynamic_density, blending,
            #                   scene_flow, dynamic_sdf, inverse_pts, uv[..., 0:self.uv_dim]], dim=-1)

        return torch.cat([static_rgb, static_density, dynamic_rgb, dynamic_density, blending,
                          scene_flow], dim=-1)

    def update_lr(self):
        decay_rate = 0.1
        decay_steps = self.args.lr_decay
        new_lrate = self.args.lr * (decay_rate ** (self.global_step / decay_steps))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lrate

    def run_dynamic(self, pts, time):
        # this function will be called only during training
        pts = positional_encoding(pts.view(-1, 3), self.args.multires)
        time = positional_encoding(torch.Tensor([time]), self.args.multires_time).view(1, -1)
        latent_code = self.net_time_embedding(time)
        _, scene_flow, _ = self.net_dynamic(pts, time)
        uv = self.net_atlas(pts[..., 0:3], latent_code)
        if self.uv_dim == 2:
            minus_1 = -torch.ones(uv.shape[0], 1)
            uv = torch.cat([uv, minus_1], dim=-1)
        inverse_pts = self.net_inverse_atlas(uv, latent_code)

        return scene_flow, uv[..., 0:self.uv_dim], inverse_pts

    def visualize_mesh_3d(self, all_times, hwf, icosphere_division=6):
        import trimesh
        meshes = []
        textures = []

        for t in all_times:
            t = positional_encoding(torch.Tensor([t]), self.args.multires_time).view(1, -1)
            latent_code = self.net_time_embedding(t)
            mesh = trimesh.creation.icosphere(icosphere_division)
            grid = torch.Tensor(mesh.vertices)
            vertices = self.net_inverse_atlas(grid, latent_code)
            if self.args.no_ndc:
                mesh.vertices = vertices.data.cpu().numpy()
            else:
                mesh.vertices = NDC2world(vertices, *hwf).data.cpu().numpy()
            meshes.append(mesh)

            grid = positional_encoding(grid, self.args.multires)
            texture = self.net_texture(grid)
            textures.append(texture)

        return meshes, textures

    def visualize_cubemap_3d(self):
        save_dir = os.path.join(self.args.base_dir, self.args.exp_name, 'cubemaps')
        os.makedirs(save_dir, exist_ok=True)
        import trimesh
        mesh = trimesh.creation.icosphere(7)
        grid = torch.Tensor(mesh.vertices)  # pts on the sphere
        texture = self.net_texture(positional_encoding(grid, self.args.multires))
        texture = texture.data.cpu().numpy().reshape(-1, 3)

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(grid)
        pcd.colors = open3d.utility.Vector3dVector(texture)
        # open3d.visualization.draw_geometries([pcd])
        open3d.io.write_point_cloud(os.path.join(save_dir, "pcd.ply"), pcd)

    def get_rmm(self, t):
        # only support uv_dim=2 now
        time = positional_encoding(torch.Tensor([t]), self.args.multires_time).view(1, -1)
        latent_code = self.net_time_embedding(time)
        uv = torch.Tensor(generate_grid(2, self.resolution)).view(-1, 2)
        uv.requires_grad_(True)
        pts = self.net_inverse_atlas(uv, latent_code)
        d_output = torch.ones_like(pts, requires_grad=False, device=pts.device)
        grads = []
        i = 0
        while i < pts.shape[-1]:
            grad = torch.autograd.grad(
                outputs=pts[..., i],
                inputs=uv,
                grad_outputs=d_output[..., 0],
                create_graph=True)[0]
            grads.append(grad)
            i += 1
        J = torch.stack(grads, dim=1)
        g = torch.matmul(J.transpose(1, 2), J)
        return g

    def get_eikonal_gradient(self, t):
        eikonal_points = torch.empty(self.n_eik_points, 3).uniform_(-1, 1).cuda()  # make sense only in NDC space
        eikonal_points.requires_grad_(True)
        pts = positional_encoding(eikonal_points, self.args.multires)
        t = positional_encoding(torch.Tensor([t]), self.args.multires_time).view(1, -1)

        sdf, _, _ = self.net_dynamic(pts, t)
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=eikonal_points,
            grad_outputs=d_output,
            create_graph=True)[0]
        return gradients

    def import_texture(self, filename):
        if self.uv_dim == 3:
            texture = load_cube_from_single_texture(filename)
        else:
            texture = np.array(Image.open(filename)) / 255.0
        self.net_texture.uvmap = torch.Tensor(texture)

    def _export_cube(self):
        grid = torch.Tensor(generate_grid(2, self.resolution))
        textures = []
        for index in range(6):
            uvs = convert_cube_uv_to_xyz(index, grid)
            uvs = positional_encoding(uvs.view(-1, 3), self.args.multires)
            textures.append(self.net_texture(uvs))
        return torch.stack(textures, dim=0).view(6, self.resolution, self.resolution, 3)

    def export_texture(self):
        with torch.no_grad():
            if self.uv_dim == 3:
                texture = self._export_cube()
                texture = merge_cube_to_single_texture(texture)
                return texture
            else:
                grid = torch.Tensor(generate_grid(2, self.resolution))
                uv = positional_encoding(grid.view(-1, 2), self.args.multires)
                texture = self.net_texture(uv).view(self.resolution, self.resolution, 3)
            return texture
