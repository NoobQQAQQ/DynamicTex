import configargparse
import os


def get_parameters():
    parser = configargparse.ArgumentParser()

    # global options
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--exp_name", type=str,
                        help='experiment name')
    parser.add_argument("--base_dir", type=str, default='./runs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--data_dir", type=str, default='./data/DynamicScene/Balloon1',
                        help='input data directory')

    # running options
    parser.add_argument("--model_name", type=str, default='dynamic_tex',
                        help='model name')
    parser.add_argument("--dataset_name", type=str, default='dynamic_scene',
                        help='dataset name')
    parser.add_argument("--resume_dir", type=str, default='./runs/weights/',
                        help='directory of saved checkpoint')
    parser.add_argument("--resume_from", type=int, default=300000,
                        help="which iteration to resume from")
    parser.add_argument("--random_seed", type=int, default=1,
                        help='fix random seed for repeatability')
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--is_train", action='store_true',
                        help='train model')
    parser.add_argument("--pretrain", action='store_true',
                        help='pretrain the StaticNeRF')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')

    # cubemap options
    parser.add_argument("--resolution", type=int, default=512,
                        help='uvmap resolution when sampling')
    parser.add_argument("--uvmap_name", type=str, default='test',
                        help='use which uvmap to render video')

    # training options
    parser.add_argument("--pretrain_niter", type=int, default=300000,
                        help='number of pretrain iterations')
    parser.add_argument("--full_model_niter", type=int, default=300000,
                        help='number of full model iterations')
    parser.add_argument("--N_imgs", type=int, default=1,
                        help='number of images')
    parser.add_argument("--N_rays", type=int, default=32*32*4,
                        help='number of random rays per image per gradient step')
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lr_decay", type=int, default=300000,
                        help='exponential learning rate decay')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_time", type=int, default=10,
                        help='log2 of max freq for positional encoding (1D time)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (3D direction)')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')

    # StaticNerf options
    parser.add_argument("--static_depth", type=int, default=8,
                        help='layers in static nerf')
    parser.add_argument("--static_width", type=int, default=256,
                        help='channels per layer in static nerf')

    # DynamicSdF options
    parser.add_argument("--dynamic_depth", type=int, default=8,
                        help='layers in dynamic nerf')
    parser.add_argument("--dynamic_width", type=int, default=256,
                        help='channels per layer in dynamic nerf')

    # sdf2density and latent code options
    parser.add_argument("--beta_density", type=float, default=0.1,
                        help='initial beta value in sdf to density')
    parser.add_argument("--latent_code_dim", type=int, default=256,
                        help='geometry latent code dim for uv mapping')

    # UV mapping options
    parser.add_argument("--uv_dim", type=int, default=2,
                        help='dimension of uv map')
    parser.add_argument("--atlas_depth", type=int, default=3,
                        help='layers in atlas network')
    parser.add_argument("--atlas_width", type=int, default=256,
                        help='channels per layer in atlas network')
    parser.add_argument("--inverse_atlas_depth", type=int, default=3,
                        help='layers in inverse atlas network')
    parser.add_argument("--inverse_atlas_width", type=int, default=256,
                        help='channels per layer in inverse atlas network')

    # texture net options
    parser.add_argument("--texture_depth", type=int, default=4,
                        help='layers in texture network')
    parser.add_argument("--texture_width", type=int, default=256,
                        help='channels per layer in texture network')

    # rendering options
    parser.add_argument("--surface_threshold", type=float, default=0.1,
                        help='control whether a point belong to dynamic surface')
    parser.add_argument("--chunk", type=int, default=1024,
                        help='number of rays to be sampled in parallel, decrease if running out of memory')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')

    # llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric logging')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_test", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')


    # Loss lambdas
    parser.add_argument("--dynamic_loss_lambda", type=float, default=1.,
                        help='lambda of dynamic loss')
    parser.add_argument("--static_loss_lambda", type=float, default=1.,
                        help='lambda of static loss')
    parser.add_argument("--full_loss_lambda", type=float, default=3.,
                        help='lambda of full loss')
    parser.add_argument("--depth_loss_lambda", type=float, default=0.04,
                        help='lambda of depth loss')
    parser.add_argument("--region_loss_lambda", type=float, default=0.04,
                        help='lambda of region loss')
    parser.add_argument("--flow_loss_lambda", type=float, default=0.02,
                        help='lambda of optical flow loss')
    parser.add_argument("--smooth_loss_lambda", type=float, default=0.1,
                        help='lambda of sf smooth regularization')
    parser.add_argument("--consistency_loss_lambda", type=float, default=1,
                        help='lambda of sf cycle consistency regularization')
    parser.add_argument("--sparse_loss_lambda", type=float, default=0.1,
                        help='lambda of sparse loss')
    parser.add_argument("--cycle_loss_lambda", type=float, default=1,
                        help='lambda of uv mapping cycle loss')
    parser.add_argument("--segmentation_loss_lambda", type=float, default=1,
                        help='lambda of segmentation loss')
    parser.add_argument("--metric_loss_lambda", type=float, default=0.1,
                        help='lambda of riemannian metric loss')
    parser.add_argument("--eikonal_loss_lambda", type=float, default=1,
                        help='lambda of eikonal loss')

    # For rendering teasers
    parser.add_argument("--frame2dolly", type=int, default=-1,
                        help='choose frame to perform dolly zoom')
    parser.add_argument("--x_trans_multiplier", type=float, default=1.,
                        help='x_trans_multiplier')
    parser.add_argument("--y_trans_multiplier", type=float, default=0.33,
                        help='y_trans_multiplier')
    parser.add_argument("--z_trans_multiplier", type=float, default=5.,
                        help='z_trans_multiplier')
    parser.add_argument("--num_novelviews", type=int, default=60,
                        help='num_novelviews')
    parser.add_argument("--focal_decrease", type=float, default=200,
                        help='focal_decrease')

    return print_and_get_args(parser)


def print_and_get_args(parser):
    args = parser.parse_args()
    message = ""
    message += "------------------------ Arguments ------------------------\n"
    for k, v in sorted(vars(args).items()):
        comment = ""
        default = parser.get_default(k)
        if v != default:
            comment = "\t[default: {}]".format(str(default))
        message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
    message += "-------------------------- End ----------------------------"
    print(message)

    expr_dir = os.path.join(args.base_dir, args.exp_name)
    os.makedirs(expr_dir, exist_ok=True)
    file_name = os.path.join(expr_dir, "args.txt")
    with open(file_name, "wt") as opt_file:
        opt_file.write(message)
        opt_file.write("\n")
    return args
