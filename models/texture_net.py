import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.cube_map import sample_cubemap


class TextureNet(nn.Module):
    def __init__(self, D=4, W=256, uv_dim=2, input_ch=3, skips=[2]):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.uv_dim = uv_dim
        self.uvmap = None

        self.linear1 = nn.Sequential(nn.Linear(input_ch, W), nn.ReLU(inplace=True))

        self.linear_list = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(W, W), nn.ReLU(inplace=True)) if i not in skips
                else nn.Sequential(nn.Linear(input_ch + W, W), nn.ReLU(inplace=True))
                for i in range(D)
            ]
        )
        self.color_linear = nn.Sequential(nn.Linear(input_ch + W, W),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(W, W // 2),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(W // 2, 3),
                                          nn.Sigmoid())

    def forward(self, uv):
        #       uv shape: N_pts x input_ch
        #   output shape: N_pts x 3 (color)

        if self.uvmap is None:
            h = self.linear1(uv)
            for i, l in enumerate(self.linear_list):
                if i in self.skips:
                    h = torch.cat([uv, h], -1)
                h = self.linear_list[i](h)

            rgb = self.color_linear(torch.cat([uv, h], -1))
            return rgb
        else:
            rgb = self.sample_uvmap(uv[..., 0:self.uv_dim])
            return rgb
            # rgb = (rgb * 3).clamp(min=0, max=1)
            # return cubemap_color * rgb.mean(dim=-1, keepdim=True)

    def sample_uvmap(self, uv):
        if self.uv_dim == 3:
            return sample_cubemap(self.uvmap, uv)
        else:
            texture = self.uvmap.permute(2, 0, 1)
            sampled_color = F.grid_sample(
                texture[None],
                uv.view((1, -1, 1, 2)),
                padding_mode="border",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            return sampled_color.view(-1, 3)
