import os
from typing import Dict
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models.Sampler import ImplicitDiffusionSampler, GaussianDiffusionTrainer, ConditionalSampler, GaussianDiffusionSampler
from models.ddpm_unet import UNet
from models.divide_unet import Model
import torch.nn.functional as F
import random
from dataset.load_roof import load_dataset, load_image
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def train(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        dataset = load_image()
        dataloader = DataLoader(
            dataset, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=0, drop_last=True, pin_memory=True)

        ddpm_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                          attn=modelConfig["attn"],
                          num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
        ddpm_ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["ddpm_weight"]), map_location=device)
        ddpm_model.load_state_dict(ddpm_ckpt)
        ddpm_model.eval()

        occl_model = Model(T=20, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                               attn=modelConfig["attn"],
                               num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
        occl_ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["occl_weight"]), map_location=device)
        occl_model.load_state_dict(occl_ckpt)
        occl_model.eval()

        trainer = GaussianDiffusionTrainer(ddpm_model, modelConfig["beta_1"], modelConfig["beta_T"],
                                           modelConfig["T"]).to(device)
        sampler = ConditionalSampler(
            ddpm_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

        sampler2 = GaussianDiffusionSampler(occl_model).to(device)

        sampler3 = ImplicitDiffusionSampler(ddpm_model, modelConfig["beta_1"], modelConfig["beta_T"],
                                            modelConfig["T"]).to(device)

    save_dir = r""

    def cond_fn(x, t, z, r):
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            time = torch.randint(20, (1,)).to(device)
            for i in range(1):
                time[i] = t
            pred_mask, pred_pixel = sampler2(x_in, time)
            o1 = pred_mask * z * r[0][19-t]
            o2 = pred_pixel * pred_mask * r[0][19-t]
            eps = o1 - o2
            x_t = x - eps
            e = F.log_softmax(eps, dim=-1)
            e = e.requires_grad_(True)
            log = torch.autograd.grad(e.sum(), x_in)[0]
        return log, x_t, pred_mask, pred_pixel, eps

    with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
        for idx, (images, labels) in enumerate(tqdmDataLoader):
            x = images.to(device)
            random_lists, lists = generate_random_lists(20)
            x_m = x
            noise = torch.randn(size=[modelConfig["batch_size"], 3, 64, 64], device=device)
            x_n = noise
            for time_step in reversed(range(20)):
                print(time_step)
                log, x_m, o1, o2, eps = cond_fn(x_m, 19-time_step, x, lists)
                save_image(x_m, os.path.join(save_dir, f"x_{idx}.jpg"))
                torch.cuda.empty_cache()
                if time_step == 19:
                    mask = o1
                x_d = trainer(x, 20 - time_step)
                x_n = sampler(x_n, 20 - time_step, log).to(device)
                x_n = x_n * mask + x_d * (1-mask)
                x_n = sampler3(x_n, 20 - time_step).to(device)
                if time_step != 0:
                    y = x_n * mask + x * (1-mask)
                else:
                    y = x_n
            binary = torch.where(mask > 0.1, 1., 0.)
            y = y * binary + x * (1 - binary)
            save_image(y, os.path.join(save_dir, f"y_{idx}.jpg"))
            save_image(mask, os.path.join(save_dir, f"mask_{idx}.jpg"))
            torch.cuda.empty_cache()


def generate_random_lists(length):
    lists = []
    random_list = [random.random() for _ in range(length)]
    random_list.sort(reverse=True)
    total_sum = sum(random_list)
    normalized_list = [value / total_sum for value in random_list]
    lists.append(normalized_list)

    cumulative = []
    for lst in lists:
        cum_sum = [lst[0]]
        for j in range(1, len(lst)):
            cum_sum.append(cum_sum[-1] + lst[j])
        # cumulative.append(cum_sum)
        cumulative.append(cum_sum[::-1])
    return cumulative, lists