import numpy as np
import torch

from .base_dataset import BaseDataset
from utils.load_llff import load_llff_data
from utils.ray_helpers import get_rays, ndc_rays


class DynamicSceneDataset(BaseDataset):
    def __init__(self, args):
        super().__init__(args)

        images, invdepths, masks, poses, bds,\
        render_poses, render_focals, flow_grids = load_llff_data(args, args.data_dir, args.factor,
                                                            frame2dolly=args.frame2dolly, recenter=True,
                                                            bd_factor=.9, spherify=args.spherify)
        assert len(poses) == len(images)

        self.images = images
        self.invdepths = invdepths
        self.masks = 1.0 - masks   # Static region mask
        hwf = poses[0, :3, -1]
        H, W, focal = hwf
        H, W = int(H), int(W)
        self.hwf = [H, W, focal]
        self.poses = poses[:, :3, :4]
        self.render_poses = render_poses
        self.render_focals = render_focals
        self.flow_grids = flow_grids

        # Use all views to train
        self.total_num = images.shape[0]
        print("# of training images:", self.total_num)
        self.i_train = np.array([i for i in np.arange(int(images.shape[0]))])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            raise NotImplementedError
            self.near = np.ndarray.min(bds) * .9
            self.far = np.ndarray.max(bds) * 1.
        else:
            self.near = 0.
            self.far = 1.
        print('NEAR FAR', self.near, self.far)

    def __len__(self):
        return self.total_num

    def __getitem__(self, idx):
        img_idx = self.i_train[idx]
        item = {}

        item["num_img"] = self.total_num
        item["img_idx"] = img_idx
        item["time"] = img_idx / self.total_num * 2. - 1.0  # time of the current frame, [0,N) -> [-1,1)
        item["far"] = self.far
        item["near"] = self.near
        item["pose"] = torch.Tensor(self.poses[img_idx])
        item["gt_image"] = torch.tensor(self.images[img_idx])
        item["gt_mask"] = torch.Tensor(self.masks[img_idx])
        item["gt_invdepth"] = torch.Tensor(self.invdepths[img_idx])
        item["gt_flow_grid"] = torch.Tensor(self.flow_grids[img_idx])

        rays_o, rays_d = get_rays(self.hwf, item["pose"])  # (H, W, 3), (H, W, 3)
        item["rays_d_world"] = rays_d.view(-1, 3)
        item["rays_o_world"] = rays_o.view(-1, 3)

        # return all rays
        if not self.is_train:
            if not self.args.no_ndc:
                # for forward facing scenes
                rays_o, rays_d = ndc_rays(self.hwf, 1., rays_o, rays_d)

            rays_o = torch.reshape(rays_o, [-1, 3])
            rays_d = torch.reshape(rays_d, [-1, 3])
            near, far = self.near * torch.ones_like(rays_d[..., :1]), self.far * torch.ones_like(rays_d[..., :1])

            # (ray origin, ray direction, min dist, max dist) for each ray
            rays = torch.cat([rays_o, rays_d, near, far], -1)
            item["rays"] = rays
        # return random selected rays
        else:
            coords_d = torch.stack((torch.where(item["gt_mask"] < 0.5)), -1)
            coords_s = torch.stack((torch.where(item["gt_mask"] >= 0.5)), -1)

            if self.pretrain:
                # sample 2/3 rays from static region and 1/3 from dynamic region
                N_dynamic = min(len(coords_d), self.args.N_rays // 3)
                N_static = self.args.N_rays - N_dynamic
            else:
                # sample 1/2 rays from dynamic region and 1/2 from static region
                N_dynamic = min(len(coords_d), self.args.N_rays // 2)
                N_static = self.args.N_rays - N_dynamic

            select_inds_d = np.random.choice(coords_d.shape[0], size=N_dynamic, replace=False)
            select_inds_s = np.random.choice(coords_s.shape[0], size=N_static, replace=False)
            select_coords = torch.cat([coords_s[select_inds_s], coords_d[select_inds_d]], 0)

            def select_batch(value, select_coords=select_coords):
                return value[select_coords[:, 0], select_coords[:, 1]]


            item['batch_color'] = select_batch(item["gt_image"])
            item["batch_flow_grid"] = select_batch(item["gt_flow_grid"])  # (N_rays, 8)
            item["batch_mask"] = select_batch(item["gt_mask"][..., None])
            item["batch_invdepth"] = select_batch(item["gt_invdepth"])

            rays_o = select_batch(rays_o)  # (N_rays, 3)
            rays_d = select_batch(rays_d)  # (N_rays, 3)
            item["rays_d_world"] = rays_d
            item["rays_o_world"] = rays_o
            if not self.args.no_ndc:
                # for forward facing scenes
                rays_o, rays_d = ndc_rays(self.hwf, 1., rays_o, rays_d)

            rays_o = torch.reshape(rays_o, [-1, 3])
            rays_d = torch.reshape(rays_d, [-1, 3])
            near, far = self.near * torch.ones_like(rays_d[..., :1]), self.far * torch.ones_like(rays_d[..., :1])

            # (ray origin, ray direction, min dist, max dist) for each ray
            rays = torch.cat([rays_o, rays_d, near, far], -1)
            item["rays"] = rays
        return item
