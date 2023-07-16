import torch
from torch import nn


class DynamicSdf(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_time=1, skips=[4]):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips

        self.linear1 = nn.Sequential(nn.Linear(input_ch + input_ch_time, W), nn.ReLU(inplace=True))

        self.linear_list = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(W, W), nn.ReLU(inplace=True)) if i not in skips
                else nn.Sequential(nn.Linear(input_ch + input_ch_time + W, W), nn.ReLU(inplace=True))
                for i in range(D)
            ]
        )
        self.sdf_linear = nn.Sequential(nn.Linear(W, 1), nn.Hardtanh(min_val=-1, max_val=2))  # make sense only in NDC space
        # self.sdf_linear = nn.Sequential(nn.Linear(W, 1), nn.ReLU(inplace=True))
        self.blending_liner = nn.Sequential(nn.Linear(W, 1), nn.Sigmoid())
        self.sf_linear = nn.Sequential(nn.Linear(W, 6), nn.Tanh())  # make sense only in NDC space

    def forward(self, pts, time):
        #      pts shape: N_pts x input_ch
        #     time shape: 1 x input_ch_views
        #   output shape: N_pts x (1+6+1) (sdf + sf + blending)

        xyzt = torch.cat([pts, torch.ones(pts.shape[0], self.input_ch_time) * time], -1)
        h = self.linear1(xyzt)
        for i, l in enumerate(self.linear_list):
            if i in self.skips:
                h = torch.cat([xyzt, h], -1)
            h = self.linear_list[i](h)

        sdf = self.sdf_linear(h)
        # the probability of a point belonging to the dynamic region
        blending = self.blending_liner(h)
        sf = self.sf_linear(h)

        return sdf, sf, blending
