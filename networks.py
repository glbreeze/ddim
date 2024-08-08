import torch
import torch.nn as nn


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=5000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = torch.outer(x, freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ==================== Define Unet ====================

class DiffusionBlock(nn.Module):
    def __init__(self, in_dim, out_dim, embed_dim, act, norm='none'):
        super(DiffusionBlock, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.dense = nn.Linear(embed_dim, out_dim)
        if norm == 'gnorm':
            self.norm = nn.GroupNorm(4, out_dim)
        else:
            self.norm = None
        self.act = act

    def forward(self, x, emb):
        h = self.layer(x) + self.dense(emb)
        if self.norm is not None:
            h = self.norm(h)
        h = self.act(h)
        return h


class UNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, in_dim, embed_dim=32, channels=[128, 128, 256]):
        """Initialize a time-dependent score-based network.
        """
        super().__init__()
        self.dim = in_dim
        self.embed = nn.Sequential(PositionalEmbedding(num_channels=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))

        self.act = lambda x: x * torch.sigmoid(x)
        enc_blocks = []
        for idx, out_dim in enumerate(channels):
            enc_blocks.append(
                DiffusionBlock(in_dim, out_dim, embed_dim, act=self.act, norm='none')
            )
            in_dim = out_dim
        self.enc_blocks = nn.Sequential(*enc_blocks)

        dec_blocks = []
        for idx, out_dim in enumerate(channels[::-1][1:]):
            dec_blocks.append(
                DiffusionBlock(in_dim if idx == 0 else in_dim + in_dim, out_dim, embed_dim, act=self.act, norm='none')
            )
            in_dim = out_dim
        self.dec_blocks = nn.Sequential(*dec_blocks)

        self.out = nn.Linear(in_dim + in_dim, self.dim)
        # self.marginal_prob_std = marginal_prob_std

    def forward(self, x, sigma):
        embed = self.act(self.embed(sigma.squeeze()))

        encoded = []
        for idx, blk in enumerate(self.enc_blocks):
            x = blk(x, embed)
            encoded.append(x)

        for idx, blk in enumerate(self.dec_blocks):
            if idx == 0:
                input = x
                encoded.pop()
            else:
                input = torch.cat([x, encoded.pop()], dim=1)
            x = blk(input, embed)

        x = self.out(torch.cat([x, encoded.pop()], dim=1))

        return x


class PrecondUnet(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 embed_dim=32,
                 channels=[128, 128, 256],
                 sigma_min=0,  # Minimum supported noise level.
                 sigma_max=float('inf'),  # Maximum supported noise level.
                 sigma_data=0.5,  # Expected standard deviation of the training data.
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = UNet(in_dim=in_dim, embed_dim=embed_dim, channels=channels)

    def forward(self, x, sigma):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x), c_noise.flatten())
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)