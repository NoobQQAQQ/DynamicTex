import torch
import numpy as np

from parameter import get_parameters
from runner import Runner
from datasets import create_dataset
from models import create_model


def main():
    args = get_parameters()

    # fix random seed if given
    if args.random_seed is not None:
        print('Fixing random seed: ', args.random_seed)
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)

    # create dataset & model
    dataset = create_dataset(args)
    model = create_model(args)  # resume or from scratch according to args
    model.print_networks()

    # model.load_pretrain()
    # start training or testing according to args
    runner = Runner(args)
    runner.run(model, dataset)
    # runner.visualize_object_3d(model, dataset)
    # runner.render_with_cubemap(model, dataset)


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')  # move to gpu automatically
    torch.backends.cudnn.benchmark = True  # for fast training
    main()
