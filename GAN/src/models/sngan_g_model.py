import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

import src


class SNGANGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.process = nn.Sequential(
            spectral_norm(nn.Linear(src.models.z_size, 512, bias=False)),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(512, 128, bias=False)),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(128, 32, bias=False)),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(32, src.models.x_size))
        )

        self.apply(src.utils.init_weights)

    def forward(self, x: torch.Tensor):
        x_hat = self.process(x)
        return x_hat
