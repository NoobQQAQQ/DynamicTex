import torch
from torch import nn
import torch.nn.functional as F


class SphereTemplate:
    def get_random_points(self, npoints, device):
        """
        Get random points on a Sphere
        Return Tensor of Size [x, 3, x ... x]
        """
        with torch.no_grad():
            points = torch.randn((npoints, 3)).to(device).float() * 2 - 1
            points = F.normalize(points, dim=-1)
        return points

    def get_regular_points(self, npoints, device):
        """
        Get regular points on a Sphere
        Return Tensor of Size [x, 3]
        """
        import trimesh
        mesh = trimesh.creation.icosphere(6)
        return torch.tensor(mesh.vertices).to(device).float()


class InverseAtlasNet(nn.Module):
    def __init__(self, D=3, W=256, input_ch=3, skips=[1], latent_code_dim=64):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.latent_code_dim = latent_code_dim

        self.linear1 = nn.Sequential(nn.Linear(input_ch, W), nn.ReLU(inplace=True))
        self.linear_latent = nn.Sequential(nn.Linear(latent_code_dim, W), nn.ReLU(inplace=True))

        self.linear_list = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(W, W), nn.ReLU(inplace=True)) if i not in skips
                else nn.Sequential(nn.Linear(W + input_ch, W), nn.ReLU(inplace=True))
                for i in range(D)
            ]
        )
        self.last_linear = nn.Linear(self.W, 3)

    def forward(self, uvs, latent_code):
        #   uvs shape : N_pts x 3, coords on 3D unit sphere or 3D plane
        # latent code : 1 x code_dim, control unit sphere to object morphism
        #      output : xyz(N_pts x 3)

        h = self.linear1(uvs) + self.linear_latent(latent_code)
        for i, l in enumerate(self.linear_list):
            if i in self.skips:
                h = torch.cat([h, uvs], -1)
            h = self.linear_list[i](h) + self.linear_latent(latent_code)
        xyz = self.last_linear(h)
        return xyz
