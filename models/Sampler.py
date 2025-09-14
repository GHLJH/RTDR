import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import os
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda:0")

def extract(v, t, x_shape):
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, i):

        t = torch.linspace(self.T, 0, (20 + 1)).to(torch.long).to(device)
        # t = torch.randint(t[i], size=(x_0.shape[0],), device=x_0.device)
        t = torch.full((x_0.shape[0],), t[i-1]-1, device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        return x_t


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T):
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var = self.p_mean_variance(x_t=x_t, t=t)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        # torch.clip(x_0, -1, 1)
        return torch.clip(x_0, -1, 1)


class ImplicitDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, ddim_step=20, eta=1):
        super().__init__()

        self.model = model
        self.T = T
        self.eta = eta
        self.ddim_step = ddim_step
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(alphas, dim=0).to(device)
        self.alphas_bar_prev = F.pad(self.alphas_bar, [1, 0], value=1)[:T].to(device)

    def forward(self, x_T, t):
        x_t = x_T
        ts = torch.linspace(self.T, 0, (self.ddim_step + 1)).to(torch.long).to(device)
        # cur_t = (ts[i - 1] - 1).to(device)
        cur_t = ts[t - 1] - 1
        ct = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * cur_t
        # prev_t = (ts[i] - 1).to(device)
        prev_t = ts[t] - 1
        pt = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * prev_t
        ab_cur = extract(self.alphas_bar, ct, x_T.shape).to(device)
        ab_prev = extract(self.alphas_bar, pt, x_T.shape).to(device) if prev_t >= 0 else 1
        eps = self.model(x_t, ct)
        var = (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
        noise = torch.randn_like(x_T)
        a = (ab_prev / ab_cur) ** 0.5 * x_t
        b = ((1 - ab_prev - var) ** 0.5 - (ab_prev * (1 - ab_cur) / ab_cur) ** 0.5) * eps
        c = var ** 0.5 * noise
        x_t = a + b + c
        x_0 = x_t
        return torch.clip(x_0, -1, 1)



class ConditionalSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, ddim_step=20, eta=1):
        super().__init__()

        self.model = model
        self.T = T
        self.eta = eta
        self.ddim_step = ddim_step
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(alphas, dim=0).to(device)
        self.alphas_bar_prev = F.pad(self.alphas_bar, [1, 0], value=1)[:T].to(device)

    def forward(self, x_T, t, grad):
        ts = torch.linspace(self.T, 0, (self.ddim_step + 1)).to(torch.long).to(device)
        x_t = x_T
        cur_t = ts[t-1] - 1
        ct = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * cur_t
        prev_t = ts[t] - 1
        pt = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * prev_t
        ab_cur = extract(self.alphas_bar, ct, x_T.shape).to(device)
        ab_prev = extract(self.alphas_bar, pt, x_T.shape).to(device) if prev_t >= 0 else 1
        eps = self.model(x_t, ct)
        e = eps - ((1 - ab_cur) ** 0.5) * grad
        a = (ab_prev / ab_cur) ** 0.5 * x_t

        b = ((1 - ab_prev) ** 0.5) * e - ((((1 - ab_cur) ** 0.5) * (ab_prev ** 0.5)) / ab_cur ** 0.5) * e
        x_t = a + b

        x_0 = x_t
        return torch.clip(x_0, -1, 1)


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_T, t):
        x_t = x_T
        output1, output2 = self.model(x_t, t)
        eps = output1 - output2
        x_t = x_t - eps
        x_0 = x_t
        return output1, output2
