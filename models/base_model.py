import torch
from torch import nn
import os


class BaseModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.global_step = None
        self.args = args
        self.is_train = args.is_train
        self.pretrain = args.pretrain
        self.resume_dir = args.resume_dir
        self.save_dir = os.path.join(args.base_dir, args.exp_name, 'weights')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_ids = list(range(torch.cuda.device_count()))
        self.optimizer = None
        self.network_names = []  # store names of all component networks

    def name(self):
        return self.__class__.__name__

    def forward(self):
        raise NotImplementedError()

    def optimize_parameters(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_lr(self):
        raise NotImplementedError()

    def set_global_step(self, step=None):
        self.global_step = step

    def get_global_step(self):
        return self.global_step

    def get_networks(self) -> [nn.Module]:
        ret = []
        for name in self.network_names:
            assert isinstance(name, str)
            net = getattr(self, "net_{}".format(name))
            assert isinstance(net, nn.Module)
            ret.append(net)
        return ret

    def print_networks(self):
        num_params = 0
        print("----------------------- Networks -----------------------")
        for name, net in zip(self.network_names, self.get_networks()):
            for param in net.parameters():
                num_params += param.numel()
            print(net)
        print("Total number of parameters: {:.3f}M".format(num_params / 1e6))
        print("--------------------------------------------------------")

    def eval_mode(self):
        """turn on eval mode"""
        for net in self.get_networks():
            net.eval()

    def train_mode(self):
        """turn on train mode"""
        for net in self.get_networks():
            net.train()

    def save_model(self):
        save_filename = '{:06d}.pth'.format(self.global_step)
        save_path = os.path.join(self.save_dir, save_filename)
        save_dict = {}
        save_dict['global_step'] = self.global_step
        save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        for name, net in zip(self.network_names, self.get_networks()):
            if isinstance(net, nn.DataParallel):
                net = net.module
            save_dict[name] = net.state_dict()
        torch.save(save_dict, save_path)

    def load_model(self, resume_from):
        ckpt_filename = '{:06d}.pth'.format(resume_from)
        ckpt_path = os.path.join(self.resume_dir, ckpt_filename)
        print('reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.global_step = ckpt['global_step'] + 1
        if self.is_train:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print(f'resuming training from step {self.global_step - 1}')
        for name, net in zip(self.network_names, self.get_networks()):
            if isinstance(net, nn.DataParallel):
                net = net.module
            net.load_state_dict(ckpt[name], strict=False)

