"""The published PRnet network with project-specific input scaling."""

import torch

from torch import nn

from prnet_core import PGM


class ControlPRnet(nn.Module):
    def __init__(self, n_genes=978):
        super().__init__()
        self.register_buffer("control_center", torch.zeros(n_genes))
        self.register_buffer("control_scale", torch.ones(n_genes))
        self.network = PGM(
            x_dim=n_genes,
            c_dim=64,
            n_dim=10,
            hidden_layer_sizes=[128],
            z_dimension=64,
            adaptor_layer_sizes=[128],
            comb_adapt_dim=1024,
            dr_rate=0.05,
        )

    def forward(self, chemical, cell, control, noise=None):
        control = (control - self.control_center) / self.control_scale
        if noise is None:
            noise = torch.randn(len(control), 10, device=control.device)
        output = self.network(control, chemical * 1.041392685158225, noise)
        mean, raw_variance = output.chunk(2, dim=1)
        return mean, nn.functional.softplus(raw_variance).clamp_min(1e-6)
