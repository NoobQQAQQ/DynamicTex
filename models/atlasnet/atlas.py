import torch
from torch import nn
import torch.nn.functional as F


class AtlasNet(nn.Module):
    def __init__(self, D=3, W=256, input_ch=3, skips=[1], latent_code_dim=64, uv_dim=2):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.latent_code_dim = latent_code_dim
        self.uv_dim = uv_dim

        self.linear1 = nn.Sequential(nn.Linear(input_ch, W), nn.ReLU(inplace=True))
        self.linear_latent = nn.Sequential(nn.Linear(latent_code_dim, W), nn.ReLU(inplace=True))

        self.linear_list = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(W, W), nn.ReLU(inplace=True)) if i not in skips
                else nn.Sequential(nn.Linear(W + input_ch, W), nn.ReLU(inplace=True))
                for i in range(D)
            ]
        )
        self.last_linear = nn.Linear(self.W, uv_dim)

    def forward(self, xyz, latent_code):
        #   xyz shape : N_pts x 3
        # latent code : 1 x code_dim, control unit sphere to object morphism
        #      output : uvs(N_pts x 3) : coords on 3D unit sphere
        #             : uv (N_pts x 2) : coords on 2D square [-1,1]^2
        h = self.linear1(xyz) + self.linear_latent(latent_code)
        for i, l in enumerate(self.linear_list):
            if i in self.skips:
                h = torch.cat([h, xyz], -1)
            h = self.linear_list[i](h) + self.linear_latent(latent_code)
        uv = self.last_linear(h)
        if self.uv_dim == 3:
            return F.normalize(uv, dim=-1)
        else:
            return F.tanh(uv)




