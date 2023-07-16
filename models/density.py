import torch.nn as nn
import torch


class LaplaceDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, beta, beta_min=0.0001):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))
        self.beta_min = torch.tensor(beta_min).cuda()

    def forward(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta
