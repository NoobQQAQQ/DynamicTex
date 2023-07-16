import torch
from torch import nn


class StaticNerf(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, skips=[4], use_viewdirs=True):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.linear1 = nn.Sequential(nn.Linear(input_ch, W), nn.ReLU(inplace=True))

        self.linear_list = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(W, W), nn.ReLU(inplace=True)) if i not in skips
                else nn.Sequential(nn.Linear(input_ch + W, W), nn.ReLU(inplace=True))
                for i in range(D)
            ]
        )
        self.density_linear = nn.Sequential(nn.Linear(W, 1), nn.ReLU(inplace=True))

        if self.use_viewdirs:
            self.color_linear = nn.Sequential(nn.Linear(input_ch + input_ch_views + W, W),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(W, W // 2),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(W // 2, 3),
                                              nn.Sigmoid())
        else:
            self.color_linear = nn.Sequential(nn.Linear(input_ch + W, 3), nn.Sigmoid())

    def forward(self, pts, viewdirs=None):
        #      pts shape: N_pts x input_ch
        # viewdirs shape: N_pts x input_ch_views
        #   output shape: N_pts x 4 (rgb + density)

        h = self.linear1(pts)
        for i, l in enumerate(self.linear_list):
            if i in self.skips:
                h = torch.cat([pts, h], -1)
            h = self.linear_list[i](h)

        density = self.density_linear(h)
        if self.use_viewdirs:
            rgb = self.color_linear(torch.cat([pts, viewdirs, h], -1))
        else:
            rgb = self.color_linear(torch.cat([pts, h], -1))

        return rgb, density
