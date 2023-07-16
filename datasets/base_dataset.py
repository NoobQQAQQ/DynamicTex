import torch.utils.data.dataset as data
import numpy as np


class BaseDataset(data.Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.data_dir = args.data_dir
        self.is_train = args.is_train
        self.pretrain = args.pretrain

    def name(self):
        return self.__class__.__name__

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()

    def get_item(self, idx=None, random=False):
        if random:
            idx = np.random.randint(self.__len__())
        return self.__getitem__(idx)
