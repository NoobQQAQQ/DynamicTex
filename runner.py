import torch
import os
import logging
import logging.handlers
import numpy as np
import copy
import imageio
from torch.utils.tensorboard import SummaryWriter

from renderer import Renderer
from loss_evaluator import LossEvaluator


def to8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def normalize_depth(depth):
    return torch.clamp(depth / percentile(depth, 97), 0., 1.)


def percentile(t, q):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """

    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result


def save_res(moviebase, ret, fps=None):
    if fps == None:
        if len(ret[0]['view_fixed']['rgb_s']) < 25:
            fps = 4
        else:
            fps = 24

    for item in ret:
        for k in item['view_fixed']:
            save_dir = os.path.join(moviebase, 'view{:03d}'.format(item['idx']))
            os.makedirs(save_dir, exist_ok=True)
            imageio.mimwrite(os.path.join(save_dir, k + '.mp4'),
                             to8b(item['view_fixed'][k]), fps=fps, quality=8, macro_block_size=1)
        for k in item['time_fixed']:
            save_dir = os.path.join(moviebase, 'time{:03d}'.format(item['idx']))
            os.makedirs(save_dir, exist_ok=True)
            imageio.mimwrite(os.path.join(save_dir, k + '.mp4'),
                             to8b(item['time_fixed'][k]), fps=fps, quality=8, macro_block_size=1)


class Runner:
    def __init__(self, args):
        self.args = args

        self.is_train = args.is_train
        self.pretrain = args.pretrain
        self.pretrain_niter = args.pretrain_niter
        self.full_model_niter = args.full_model_niter

        self.log_to_dir = os.path.join(args.base_dir, args.exp_name)
        self.test_save_dir = os.path.join(self.log_to_dir, 'test_results')
        self.cubemap_dir = os.path.join(self.log_to_dir, 'cubemaps')

        os.makedirs(self.log_to_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_to_dir, 'weights'), exist_ok=True)  # store model weights
        os.makedirs(self.test_save_dir, exist_ok=True)  # store test results
        os.makedirs(self.cubemap_dir, exist_ok=True)  # store cubemaps

        if not self.is_train:
            self.cubemap_name = args.cubemap_name
            self.edited_dir = os.path.join(self.log_to_dir, 'edited')
            os.makedirs(self.edited_dir, exist_ok=True)  # store edited results

        self.logger = self.create_logger()

        self.i_video = args.i_video
        self.i_test = args.i_test
        self.i_weights = args.i_weights
        self.i_print = args.i_print
        self.i_img = args.i_img
        # delay initialization to run
        self.writer = None
        self.renderer = None
        self.loss_evaluator = None

    def run(self, model, dataset):
        self.writer = SummaryWriter(self.log_to_dir)
        self.renderer = Renderer(self.args)
        self.loss_evaluator = LossEvaluator(self.args)

        if self.is_train:
            model.train_mode()
            self.train(model, dataset)
        else:
            model.eval_mode()
            with torch.no_grad():
                self.test(model, dataset)

    def train(self, model, dataset):
        # get global step (maybe resumed from some training point)
        global_step = model.get_global_step()

        test_dataset = copy.deepcopy(dataset)
        test_dataset.is_train = False

        poses = dataset.poses
        hwf = dataset.hwf

        while global_step <= self.args.full_model_niter:

            # get data from one random image
            data_dict = dataset.get_item(random=True)
            t = data_dict["time"]
            rays = data_dict["rays"]
            viewdirs = data_dict["rays_d_world"]
            num_img = data_dict["num_img"]
            # render rays to get raw outputs
            outputs = self.renderer.render(model, t, rays, viewdirs, num_img)
            loss_dict = self.loss_evaluator.compute_loss(outputs, data_dict, poses, hwf)

            model.optimize_parameters(loss_dict['total'])
            model.update_lr()

            if global_step % self.i_print == 0:
                self.logging(f"Step: {global_step}, Loss: {loss_dict['total']}")
                self.writer.add_scalar("lr", model.optimizer.param_groups[0]['lr'], global_step)
                for loss_key in loss_dict:
                    self.writer.add_scalar(loss_key, loss_dict[loss_key].item(), global_step)

            if global_step % self.i_img == 0:
                with torch.no_grad():
                    test_data_dict = test_dataset.get_item(random=True)
                    test_t = test_data_dict["time"]
                    all_rays = test_data_dict["rays"]
                    all_viewdirs = test_data_dict["rays_d_world"]
                    self.renderer.is_train = False
                    model.is_train = False
                    cubemap = model.export_texture()
                    vis_out = self.renderer.render(model, test_t, all_rays, all_viewdirs, num_img)
                    for k in vis_out:
                        vis_out[k] = vis_out[k].view(hwf[0], hwf[1], -1).squeeze()
                    model.is_train = True
                    self.renderer.is_train = True

                imageio.imwrite(os.path.join(self.cubemap_dir, 'step{:06d}.png'.format(global_step)),
                                to8b(cubemap.cpu().numpy()))

                self.writer.add_image("texture", cubemap,
                                      global_step=global_step, dataformats='HWC')
                self.writer.add_image("dynamicness_map", vis_out['dynamicness_map'],
                                      global_step=global_step, dataformats='HW')

                self.writer.add_image("gt_image", test_data_dict['gt_image'],
                                      global_step=global_step, dataformats='HWC')
                self.writer.add_image("gt_mask", test_data_dict['gt_mask'],
                                      global_step=global_step, dataformats='HW')

                self.writer.add_image("rgb_s", torch.clamp(vis_out['rgb_s'], 0., 1.),
                                      global_step=global_step, dataformats='HWC')
                self.writer.add_image("rgb_d", torch.clamp(vis_out['rgb_d'], 0., 1.),
                                      global_step=global_step, dataformats='HWC')
                self.writer.add_image("rgb_full", torch.clamp(vis_out['rgb_full'], 0., 1.),
                                      global_step=global_step, dataformats='HWC')

                self.writer.add_image("depth_s", normalize_depth(vis_out['depth_s']),
                                      global_step=global_step, dataformats='HW')
                self.writer.add_image("depth_d", normalize_depth(vis_out['depth_d']),
                                      global_step=global_step, dataformats='HW')
                self.writer.add_image("depth_full", normalize_depth(vis_out['depth_full']),
                                      global_step=global_step, dataformats='HW')

            if global_step % self.i_weights == 0:
                model.save_model()

            if global_step % self.i_test == 0:
                self.renderer.is_train = False
                model.is_train = False
                with torch.no_grad():
                    vis_out = self.renderer.render_test(model, test_dataset)
                    moviebase = os.path.join(self.test_save_dir, 'step{:06d}'.format(global_step))
                    save_res(moviebase, vis_out)
                model.is_train = True
                self.renderer.is_train = True

            global_step += 1
            model.set_global_step(global_step)

    def test(self, model, dataset):
        raise NotImplementedError

    def create_logger(self):
        logger = logging.getLogger("logger")
        logger.setLevel(logging.DEBUG)
        # logger.info(msg) output to both file and console
        # logger.debug(msg) only output to console
        fh = logging.FileHandler(os.path.join(self.log_to_dir, 'log.txt'))
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # set format, we don't care msg level here
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def logging(self, msg, to_file=True):
        if to_file:
            self.logger.info(msg)
        else:
            self.logger.debug(msg)

    def render_with_cubemap(self, model, dataset):
        self.renderer = Renderer(self.args)
        assert self.renderer.is_train is False
        assert model.is_train is False
        with torch.no_grad():
            cubemap_path = os.path.join(self.cubemap_dir, self.cubemap_name + ".png")
            model.import_texture(cubemap_path)
            vis_out = self.renderer.render_edited(model, dataset)
            for k in vis_out:
                imageio.mimwrite(os.path.join(self.edited_dir, self.cubemap_name + k + '.mp4'),
                                 to8b(vis_out[k]), fps=4, quality=8, macro_block_size=1)

    def visualize_object_3d(self, model, dataset):
        with torch.no_grad():
            all_times = dataset.i_train / float(dataset.total_num) * 2. - 1.0
            hwf = dataset.hwf

            model.visualize_cubemap_3d()
            meshes, textures = model.visualize_mesh_3d(all_times, hwf, icosphere_division=7)
            for i, (mesh, texture) in enumerate(zip(meshes, textures)):
                color = to8b(texture.data.cpu().numpy())
                c = np.ones((len(color), 4)) * 255
                c[:, :3] = color

                import trimesh
                save_dir = os.path.join(self.log_to_dir, 'meshes')
                os.makedirs(save_dir, exist_ok=True)
                # white color visualization
                mesh.visual.vertex_colors = np.ones_like(c)
                trimesh.repair.fix_inversion(mesh)
                trimesh.repair.fix_normals(mesh)
                # mesh.show(viewer="gl", smooth=True)
                mesh.export(os.path.join(save_dir, f"white_time{i}.ply"))

                # color visualization
                mesh.visual.vertex_colors = c
                trimesh.repair.fix_inversion(mesh)
                trimesh.repair.fix_normals(mesh)
                # mesh.show(viewer="gl", smooth=True)
                mesh.export(os.path.join(save_dir, f"color_time{i}.ply"))
