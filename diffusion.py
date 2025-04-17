# diffusion.py
import torch
import torch.nn.functional as F
import numpy as np

class Diffusion:
    def __init__(self, timesteps=100, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.timesteps = timesteps
        self.device = device

        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_image(self, x, t):
        """
        Forward process: Add noise to image x at timestep t
        """
        sqrt_alpha_hat = self.alpha_hat[t] ** 0.5
        sqrt_one_minus_alpha_hat = (1 - self.alpha_hat[t]) ** 0.5

        noise = torch.randn_like(x).to(self.device)
        return sqrt_alpha_hat[:, None, None, None] * x + sqrt_one_minus_alpha_hat[:, None, None, None] * noise, noise

    def sample_timesteps(self, batch_size):
        return torch.randint(low=0, high=self.timesteps, size=(batch_size,), device=self.device)

    def denoise_step(self, model, x, t):
        beta_t = self.beta[t][:, None, None, None]
        sqrt_one_minus_alpha_hat_t = (1 - self.alpha_hat[t])[:, None, None, None]
        sqrt_recip_alpha_t = (1. / self.alpha[t])[:, None, None, None]

        pred_noise = model(x, t)
        mean = sqrt_recip_alpha_t * (x - beta_t / sqrt_one_minus_alpha_hat_t * pred_noise)

        noise = torch.randn_like(x).to(self.device)
        z = torch.where(t[:, None, None, None] == 0, torch.zeros_like(x), noise)  # No noise at t=0
        sigma = beta_t ** 0.5

        return mean + sigma * z

    @torch.no_grad()
    def sample(self, model, lr_image):
        x = lr_image.to(self.device)  # Start from LR image, not noise
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
            x = self.denoise_step(model, x, t_tensor)
        return torch.clamp(x, 0., 1.)
