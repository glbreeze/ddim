
import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
device = "cuda" if torch.cuda.is_available() else 'cpu'
#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

def edm_sampler(
        net, latents, num_steps=18, randn_like=torch.randn_like,
        sigma_min=0.002, sigma_max=80, rho=7,
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    net.eval()
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    with torch.no_grad():
        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.
            denoised = net(x_hat, t_hat * torch.ones(x_hat.shape[0])).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = net(x_next, t_next * torch.ones(x_hat.shape[0])).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

# =========================== generate  ===========================


def sampling(net, seeds, batch_size, device=device, **sampler_kwargs):

    gen_samples = []
    num_batches = int( np.ceil(len(seeds) / batch_size) )
    for idx in range(num_batches):
        batch_seeds = seeds[batch_size*idx: min((batch_size*idx + batch_size), len(seeds))]

        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.in_dim], device=device)

        # Generate images.
        sampler_fn = edm_sampler
        samples = sampler_fn(net, latents, randn_like=rnd.randn_like, **sampler_kwargs)
        gen_samples.append(samples)

    gen_samples = torch.cat(gen_samples, dim=0)
    return gen_samples


