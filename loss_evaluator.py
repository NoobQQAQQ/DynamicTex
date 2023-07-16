from utils.loss_helpers import *
import torch.nn as nn


class LossEvaluator:
    def __init__(self, args):
        self.args = args
        self.pretrain = args.pretrain
        self.surface_threshold = args.surface_threshold

    def compute_loss(self, outputs, inputs, poses, hwf):
        batch_mask = inputs['batch_mask']
        batch_invdepth = inputs['batch_invdepth']
        batch_grid = inputs['batch_flow_grid']
        H, W, focal = hwf

        loss_dict = {}
        loss = 0

        # Reconstruction loss
        # Compute MAE loss between rgb_s and true RGB of static region.
        loss_dict['img_loss_s'] = img2mae(outputs['rgb_s'], inputs['batch_color'], batch_mask)
        loss += self.args.static_loss_lambda * loss_dict['img_loss_s']

        # Compute MAE loss between rgb_d and true RGB of dynamic region.
        loss_dict['img_loss_d'] = img2mae(outputs['rgb_d'], inputs['batch_color'], 1 - batch_mask)
        loss += self.args.dynamic_loss_lambda * loss_dict['img_loss_d']

        # Compute MAE loss between rgb_full and true RGB.
        loss_dict['img_loss_full'] = img2mae(outputs['rgb_full'], inputs['batch_color'])
        loss += self.args.full_loss_lambda * loss_dict['img_loss_full']

        # Depth loss
        # Depth in NDC space equals to negative disparity in Euclidean space.
        # # Compute depth loss of static region.
        # loss_dict['depth_loss_s'] = self.compute_depth_loss(outputs['depth_s'], -batch_invdepth, batch_mask)
        # loss += self.args.depth_loss_lambda * loss_dict['depth_loss_s']
        #
        # # Compute depth loss of dynamic region.
        # loss_dict['depth_loss_d'] = self.compute_depth_loss(outputs['depth_d'], -batch_invdepth, 1 - batch_mask)
        # loss += self.args.depth_loss_lambda * loss_dict['depth_loss_d']

        # Compute depth loss of whole region.
        loss_dict['depth_loss_full'] = self.compute_depth_loss(outputs['depth_full'], -batch_invdepth)
        loss += self.args.depth_loss_lambda * loss_dict['depth_loss_full']

        # Segmentation loss
        loss_dict['segmentation_loss'] = img2mae(outputs['dynamicness_map'], (1-batch_mask).squeeze())
        loss += self.args.segmentation_loss_lambda * loss_dict['segmentation_loss']

        # Region loss
        # all points of static region should have blending -> 0
        loss_dict['static_pts_loss'] = L1(outputs['blending'][batch_mask[:, 0].type(torch.bool)])
        loss += self.args.region_loss_lambda * loss_dict['static_pts_loss']

        # all points of dynamic region should have blending -> 1
        loss_dict['dynamic_pts_loss'] = L1(1 - outputs['blending'][(1-batch_mask)[:, 0].type(torch.bool)])
        loss += self.args.region_loss_lambda * loss_dict['dynamic_pts_loss']

        # # all points of static region should not be modeled by dynamic nerf
        # # therefore they should have zero dynamic weights
        # loss_dict['img_loss_d_s'] = L1(outputs['weights_d'], batch_mask)
        # loss += self.args.region_loss_lambda * loss_dict['img_loss_d_s']


        # # all points of static region should have zero scene flow
        # loss_dict['static_flow_loss'] = L1(outputs['sceneflow_b'][batch_mask[:, 0].type(torch.bool)]) + \
        #                                 L1(outputs['sceneflow_f'][batch_mask[:, 0].type(torch.bool)])
        # loss += self.args.static_region_loss_lambda * loss_dict['static_flow_loss']

        # # Sparsity loss. This may help surface learning.
        # sparse_loss = entropy(outputs['weights_d']) + entropy(outputs['blending'])
        # loss_dict['sparse_loss'] = sparse_loss
        # loss += self.args.sparse_loss_lambda * loss_dict['sparse_loss']

        # Pts cycle loss
        # ideally, only pts on dynamic surface are used for this loss
        # pts_mask = (torch.abs(outputs['dynamic_sdf']) < self.surface_threshold)
        pts_mask = (1-batch_mask).expand(outputs['raw_pts'].shape[0], outputs['raw_pts'].shape[1])
        pts_mask = pts_mask * (torch.abs(outputs['dynamic_sdf']) < self.surface_threshold).float()
        cycle_loss = pts2loss(outputs['raw_pts'], outputs['raw_pts_inverse'], 1, pts_mask)
        cycle_loss_f = pts2loss(outputs['raw_pts_f'], outputs['inverse_pts_f'], 1, pts_mask)
        cycle_loss_b = pts2loss(outputs['raw_pts_b'], outputs['inverse_pts_b'], 1, pts_mask)
        loss_dict['cycle_loss'] = (cycle_loss + cycle_loss_f + cycle_loss_b).squeeze()
        # loss_dict['cycle_loss'] = (cycle_loss).squeeze()
        loss += self.args.cycle_loss_lambda * loss_dict['cycle_loss']

        # time coherent loss
        coherent_loss = pts2loss(outputs['uv'], outputs['uv_b'], 1, pts_mask)
        coherent_loss += pts2loss(outputs['uv'], outputs['uv_f'], 1, pts_mask)
        loss_dict['coherent_loss'] = coherent_loss.squeeze()
        loss += self.args.region_loss_lambda * loss_dict['coherent_loss']

        # keep-shape loss
        shape_loss = pts2loss(outputs['raw_pts'][..., 0:2], outputs['uv'], 1, pts_mask)
        loss_dict['shape_loss'] = shape_loss.squeeze()
        loss += self.args.cycle_loss_lambda * loss_dict['shape_loss']

        # Consistency loss.
        loss_dict['consistency_loss'] = L1(outputs['sceneflow_f'] + outputs['sceneflow_f_b']) + \
                                        L1(outputs['sceneflow_b'] + outputs['sceneflow_b_f'])
        loss += self.args.consistency_loss_lambda * loss_dict['consistency_loss']

        # Motion loss.
        # Compuate EPE between induced flow and true flow (forward flow).
        # The last frame does not have forward flow.
        if inputs['img_idx'] < inputs['num_img'] - 1:
            pts_f = outputs['raw_pts_f']
            weight = outputs['weights_full']
            pose_f = poses[inputs['img_idx'] + 1, :3, :4]
            induced_flow_f = induce_flow(H, W, focal, pose_f, weight, pts_f, batch_grid[..., :2])
            flow_f_loss = img2mae(induced_flow_f, batch_grid[:, 2:4], batch_grid[:, 4:5])
            loss_dict['flow_f_loss'] = flow_f_loss
            loss += self.args.flow_loss_lambda * loss_dict['flow_f_loss']

        # Compuate EPE between induced flow and true flow (backward flow).
        # The first frame does not have backward flow.
        if inputs['img_idx'] > 0:
            pts_b = outputs['raw_pts_b']
            weight = outputs['weights_full']
            pose_b = poses[inputs['img_idx'] - 1, :3, :4]
            induced_flow_b = induce_flow(H, W, focal, pose_b, weight, pts_b, batch_grid[..., :2])
            flow_b_loss = img2mae(induced_flow_b, batch_grid[:, 5:7], batch_grid[:, 7:8])
            loss_dict['flow_b_loss'] = flow_b_loss
            loss += self.args.flow_loss_lambda * loss_dict['flow_b_loss']

        # Smooth scene flow. The summation of the forward and backward sceneflow should be small.
        smooth_loss = self.compute_sf_smooth_loss(outputs['raw_pts'],
                                                  outputs['raw_pts_f'],
                                                  outputs['raw_pts_b'],
                                                  H, W, focal)
        loss_dict['smooth_loss'] = smooth_loss
        loss += self.args.smooth_loss_lambda * loss_dict['smooth_loss']

        # Spatial smooth scene flow. (loss adapted from NSFF)
        sp_smooth_loss = self.compute_sf_smooth_s_loss(outputs['raw_pts'], outputs['raw_pts_f'], H, W, focal) + \
                         self.compute_sf_smooth_s_loss(outputs['raw_pts'], outputs['raw_pts_b'], H, W, focal)
        loss_dict['sp_smooth_loss'] = sp_smooth_loss
        loss += self.args.smooth_loss_lambda * loss_dict['sp_smooth_loss']

        # eikonal loss
        loss_dict['eikonal_loss'] = ((outputs['sdf_grad'].norm(2, dim=-1) - 1) ** 2).mean()
        loss += self.args.eikonal_loss_lambda * loss_dict['eikonal_loss']

        # Riemannnian metric loss. The texture should be time-coherent.
        # batch_loss = torch.norm(outputs['rmm'] - outputs['rmm_b'], dim=(1, 2)) + \
        #              torch.norm(outputs['rmm'] - outputs['rmm_f'], dim=(1, 2))
        # loss_dict['metric_loss'] = torch.sum(batch_loss)
        # loss += self.args.metric_loss_lambda * loss_dict['metric_loss']

        loss_dict['total'] = loss
        return loss_dict

    def compute_depth_loss(self, dyn_depth, gt_depth, mask=None):

        t_d = torch.median(dyn_depth)
        s_d = torch.mean(torch.abs(dyn_depth - t_d))
        dyn_depth_norm = (dyn_depth - t_d) / s_d

        t_gt = torch.median(gt_depth)
        s_gt = torch.mean(torch.abs(gt_depth - t_gt))
        gt_depth_norm = (gt_depth - t_gt) / s_gt

        if mask is None:
            return torch.mean((dyn_depth_norm - gt_depth_norm) ** 2)
        else:
            return torch.sum((dyn_depth_norm - gt_depth_norm) ** 2 * mask.squeeze()) / (torch.sum(mask) + 1e-8)

    # Spatial smoothness (adapted from NSFF)
    def compute_sf_smooth_s_loss(self, pts1, pts2, H, W, f):

        N_samples = pts1.shape[1]

        # NDC coordinate to world coordinate
        pts1_world = NDC2world(pts1[..., :int(N_samples * 0.95), :], H, W, f)
        pts2_world = NDC2world(pts2[..., :int(N_samples * 0.95), :], H, W, f)

        # scene flow in world coordinate
        scene_flow_world = pts1_world - pts2_world

        return L1(scene_flow_world[..., :-1, :] - scene_flow_world[..., 1:, :])

    # Temporal smoothness
    def compute_sf_smooth_loss(self, pts, pts_f, pts_b, H, W, f):

        N_samples = pts.shape[1]

        pts_world = NDC2world(pts[..., :int(N_samples * 0.9), :], H, W, f)
        pts_f_world = NDC2world(pts_f[..., :int(N_samples * 0.9), :], H, W, f)
        pts_b_world = NDC2world(pts_b[..., :int(N_samples * 0.9), :], H, W, f)

        # scene flow in world coordinate
        sceneflow_f = pts_f_world - pts_world
        sceneflow_b = pts_b_world - pts_world

        # For a 3D point, its forward and backward sceneflow should be opposite.
        return L2(sceneflow_f + sceneflow_b)
