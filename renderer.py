import torch
import numpy as np
import torch.nn.functional as F
from utils.ray_helpers import uniform_stratified_sample


class Renderer:
    def __init__(self, args):
        self.args = args
        self.chunk = args.chunk
        self.use_viewdirs = args.use_viewdirs
        self.N_samples = args.N_samples
        self.lindisp = args.lindisp
        self.perturb = args.perturb
        self.raw_noise_std = args.raw_noise_std

        self.is_train = args.is_train
        self.pretrain = args.pretrain

    def render(self, model, t, rays, viewdirs, num_img):
        if self.use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)
        # Render rays in smaller minibatches to avoid OOM
        all_ret = {}
        for i in range(0, rays.shape[0], self.chunk):
            ret = self.render_rays(model, t, rays[i:i + self.chunk], num_img)
            for k in ret:
                if k not in all_ret and ret[k] is not None:
                    all_ret[k] = []
                if ret[k] is not None:
                    all_ret[k].append(ret[k])

        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret

    def render_edited(self, model, dataset):
        H, W, _ = dataset.hwf
        num_img = dataset.total_num
        results = {'rgb_s': [], 'rgb_d': [], 'rgb_full': []}
        for idx in dataset.i_train:
            data_dict = dataset.get_item(idx)
            rays = data_dict["rays"]
            viewdirs = data_dict["rays_d_world"]
            t = data_dict["time"]
            res = self.render(model, t, rays, viewdirs, num_img)
            res['rgb_s'] = res['rgb_s'].view(H, W, -1).squeeze()
            res['rgb_d'] = res['rgb_d'].view(H, W, -1).squeeze()
            res['rgb_full'] = res['rgb_full'].view(H, W, -1).squeeze()
            results['rgb_s'].append(res['rgb_s'].cpu().numpy())
            results['rgb_d'].append(res['rgb_d'].cpu().numpy())
            results['rgb_full'].append(res['rgb_full'].cpu().numpy())
        return results

    def render_test(self, model, testset):
        # random select two images
        H, W, _ = testset.hwf
        num_img = testset.total_num
        all_times = testset.i_train / float(num_img) * 2. - 1.0
        selected_idxes = np.random.choice(testset.i_train, 2, replace=False)
        results = []
        for i in selected_idxes:
            ret = {}
            ret['idx'] = i
            ret['view_fixed'] = {'rgb_s': [], 'rgb_d': [], 'rgb_full': [],
                                 'depth_full': [], 'dynamicness_map': []}
            ret['time_fixed'] = {'rgb_s': [], 'rgb_d': [], 'rgb_full': [],
                                 'depth_full': [], 'dynamicness_map': []}
            data_dict = testset.get_item(i)
            rays = data_dict["rays"]
            viewdirs = data_dict["rays_d_world"]
            t0 = data_dict["time"]
            # fix view, change time
            for t in all_times:
                res = self.render(model, t, rays, viewdirs, num_img)
                res['rgb_s'] = res['rgb_s'].view(H, W, -1).squeeze()
                res['rgb_d'] = res['rgb_d'].view(H, W, -1).squeeze()
                res['rgb_full'] = res['rgb_full'].view(H, W, -1).squeeze()
                res['depth_full'] = res['depth_full'].view(H, W, -1).squeeze()
                res['dynamicness_map'] = res['dynamicness_map'].view(H, W, -1).squeeze()

                ret['view_fixed']['rgb_s'].append(res['rgb_s'].cpu().numpy())
                ret['view_fixed']['rgb_d'].append(res['rgb_d'].cpu().numpy())
                ret['view_fixed']['rgb_full'].append(res['rgb_full'].cpu().numpy())
                ret['view_fixed']['depth_full'].append(res['depth_full'].cpu().numpy())
                ret['view_fixed']['dynamicness_map'].append(res['dynamicness_map'].cpu().numpy())
            # fix time, change view:
            for idx in testset.i_train:
                data_dict = testset.get_item(idx)
                rays = data_dict["rays"]
                viewdirs = data_dict["rays_d_world"]
                res = self.render(model, t0, rays, viewdirs, num_img)
                res['rgb_s'] = res['rgb_s'].view(H, W, -1).squeeze()
                res['rgb_d'] = res['rgb_d'].view(H, W, -1).squeeze()
                res['rgb_full'] = res['rgb_full'].view(H, W, -1).squeeze()
                res['depth_full'] = res['depth_full'].view(H, W, -1).squeeze()
                res['dynamicness_map'] = res['dynamicness_map'].view(H, W, -1).squeeze()
                ret['time_fixed']['rgb_s'].append(res['rgb_s'].cpu().numpy())
                ret['time_fixed']['rgb_d'].append(res['rgb_d'].cpu().numpy())
                ret['time_fixed']['rgb_full'].append(res['rgb_full'].cpu().numpy())
                ret['time_fixed']['depth_full'].append(res['depth_full'].cpu().numpy())
                ret['time_fixed']['dynamicness_map'].append(res['dynamicness_map'].cpu().numpy())
            results.append(ret)
        return results

    def render_test_novel(self, model, pose2render, time2render, hwf):
        raise NotImplementedError

    def render_rays(self, model, t, rays, num_img):
        # rays_o:    [N_rays, 0:3]
        # rays_d:    [N_rays, 3:6]
        # near:      [N_rays, 6:7]
        # far:       [N_rays, 7:8]
        # viewdirs:  [N_rays, 8:11]

        # Extract unit-normalized viewing direction in world space
        viewdirs = rays[:, -3:] if rays.shape[-1] > 8 else None

        pts, z_vals, dists = uniform_stratified_sample(rays[..., 0:-3], self.N_samples, self.lindisp, self.perturb)
        raw = model(pts, t, viewdirs)
        raw = raw.view(*dists.shape, -1)

        # static_rgbd:      [N_rays, N_samples, 0:4]
        # dynamic_rgbd:     [N_rays, N_samples, 4:8]
        # blending:         [N_rays, N_samples, 8]
        # sceneflow_b:      [N_rays, N_samples, 9:12]
        # sceneflow_f:      [N_rays, N_samples, 12:15]

        # inverse_pts:      [N_rays, N_samples, 15:18]
        # uv_pts:           [N_rays, N_samples, 18:20]
        # dynamic_sdf:      [N_rays, N_samples, 20]
        rgbd1, rgbd2 = raw[..., 0:4], raw[..., 4:8]
        blending = raw[..., 8]
        sceneflow_b = raw[..., 9:12]
        sceneflow_f = raw[..., 12:15]

        rgb_s, rgb_d, rgb_full, \
        depth_s, depth_d, depth_full, \
        weights_s, weights_d, weights_full, \
        dynamicness_map = self.blend2outputs(rgbd1, rgbd2, blending, dists, z_vals)

        ret = {'rgb_s': rgb_s, 'rgb_d': rgb_d, 'rgb_full': rgb_full,
               'depth_s': depth_s, 'depth_d': depth_d, 'depth_full': depth_full,
               'weights_s': weights_s, 'weights_d': weights_d, 'weights_full': weights_full,
               'dynamicness_map': dynamicness_map, 'sceneflow_b': sceneflow_b, 'sceneflow_f': sceneflow_f,
               'blending': blending}
        if not self.is_train:
            return ret

        # training
        pts_f = pts + sceneflow_f
        pts_b = pts + sceneflow_b
        ret['raw_pts'] = pts
        ret['dynamic_sdf'] = raw[..., 20]
        ret['raw_pts_inverse'] = raw[..., 15:18]
        ret['uv'] = raw[..., 18:20]
        ret['raw_pts_f'] = pts_f
        ret['raw_pts_b'] = pts_b

        t_interval = 1. / num_img * 2.
        # get the scene flow at time t - 1
        raw_d_b, uv_b, inverse_pts_b = model.run_dynamic(pts_b.view(-1, 3), t - t_interval)
        ret['sceneflow_b_f'] = raw_d_b[..., 3:6].view(*dists.shape, -1)
        ret['uv_b'] = uv_b.view(*dists.shape, -1)
        ret['inverse_pts_b'] = inverse_pts_b.view(*dists.shape, -1)

        # get the scene flow at time t + 1
        raw_d_f, uv_f, inverse_pts_f = model.run_dynamic(pts_f.view(-1, 3), t + t_interval)
        ret['sceneflow_f_b'] = raw_d_f[..., 0:3].view(*dists.shape, -1)
        ret['uv_f'] = uv_f.view(*dists.shape, -1)
        ret['inverse_pts_f'] = inverse_pts_f.view(*dists.shape, -1)

        # get gradients of sdf
        ret['sdf_grad'] = model.get_eikonal_gradient(t)

        # get riemannnian metric matrix
        # ret['rmm_b'] = model.get_rmm(t - t_interval)
        # ret['rmm'] = model.get_rmm(t)
        # ret['rmm_f'] = model.get_rmm(t + t_interval)

        return ret

    def blend2outputs(self, rgbd1, rgbd2, blending, dists, z_vals):
        # Function for computing density from model prediction. This value is
        # strictly between [0, 1].
        def density2alpha(density, dists): return 1.0 - torch.exp(-F.relu(density) * dists)

        # Extract RGB of each sample position along each ray.
        rgb_s, density_s = rgbd1[..., 0:3], rgbd1[..., 3]
        rgb_d, density_d = rgbd2[..., 0:3], rgbd2[..., 3]
        density_full = (density_d * blending + density_s * (1. - blending))
        rgb_full = (rgb_d * blending[..., None] + rgb_s * (1. - blending)[..., None])

        # Add noise to model's predictions for density. Can be used to
        # regularize network during training (prevents floater artifacts).
        noise = 0.
        if self.raw_noise_std > 0. and self.is_train:
            noise = torch.randn(density_s.shape) * self.raw_noise_std

        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point.
        alpha_s = density2alpha(density_s + noise, dists)  # [N_rays, N_samples]
        alpha_d = density2alpha(density_d + noise, dists)  # [N_rays, N_samples]
        alpha_full = density2alpha(density_full + noise, dists)

        T_s = torch.cumprod(torch.cat([torch.ones((alpha_s.shape[0], 1)), 1. - alpha_s + 1e-10], -1), -1)[:, :-1]
        T_d = torch.cumprod(torch.cat([torch.ones((alpha_d.shape[0], 1)), 1. - alpha_d + 1e-10], -1), -1)[:, :-1]
        T_full = torch.cumprod(torch.cat([torch.ones((alpha_full.shape[0], 1)), 1. - alpha_full + 1e-10], -1), -1)[:, :-1]

        # Compute weight for RGB of each sample along each ray.  A cumprod() is
        # used to express the idea of the ray not having reflected up to this
        # sample yet.
        weights_d = alpha_d * T_d
        weights_s = alpha_s * T_s
        weights_full = alpha_full * T_full

        # Computed weighted color of each sample along each ray.
        rgb_map_d = torch.sum(weights_d[..., None] * rgb_d, -2)
        rgb_map_s = torch.sum(weights_s[..., None] * rgb_s, -2)
        rgb_map_full = torch.sum(weights_full[..., None] * rgb_full, -2)

        # Estimated depth map is expected distance.
        depth_map_d = torch.sum(weights_d * z_vals, -1)
        depth_map_s = torch.sum(weights_s * z_vals, -1)
        depth_map_full = torch.sum(weights_full * z_vals, -1)

        # Computed dynamicness
        dynamicness_map = torch.sum(weights_full * blending, -1)

        return rgb_map_s, rgb_map_d, rgb_map_full, \
               depth_map_s, depth_map_d, depth_map_full, \
               weights_s, weights_d, weights_full, \
               dynamicness_map
