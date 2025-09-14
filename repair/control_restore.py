from torchvision.utils import save_image
import os
from typing import Dict
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models.ddpm import ImplicitDiffusionSampler, GaussianDiffusionTrainer, ConditionalSampler
from models.ddpm import GaussianDiffusionSampler
from models.ddpm_unet import UNet
from models.divide_unet import Model
from dataset.load_roof import load_dataset, load_image
import torch.nn.functional as F
import random

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def restore(modelConfig: Dict):
    # load model and evaluate
    save_dir = r""
    save_dir_p = r""
    save_dir_b = r""
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        dataset = load_image()
        dataloader = DataLoader(
            dataset, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=0, drop_last=True, pin_memory=True)

        ddpm_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                          attn=modelConfig["attn"],
                          num_res_blocks=modelConfig["num_res_blocks1"], dropout=modelConfig["dropout"]).to(device)
        ddpm_ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["ddpm_weight"]), map_location=device)
        ddpm_model.load_state_dict(ddpm_ckpt)
        ddpm_model.eval()

        occl_model = Model(T=5, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
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

        def cond_fn(x, t, z, r):
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                time = torch.tensor([t], device=device)
                pred_mask, pred_pixel = sampler2(x_in, time)
                o1 = pred_mask * z
                o2 = pred_mask * pred_pixel
                eps = o1 - o2
                x_t = z - eps * r[0][t]
                min_val = eps.amin(dim=(2, 3), keepdim=True)
                max_val = eps.amax(dim=(2, 3), keepdim=True)
                normalized_eps = (eps - min_val) / (max_val - min_val + 1e-7)
                eps = normalized_eps * 2 - 1
                e = F.log_softmax(eps, dim=0)
                log = torch.autograd.grad(e.mean(), x_in)[0]
            return log, x_t, pred_mask, pred_pixel

        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for idx, (images, labels) in enumerate(tqdmDataLoader):
                x = images.to(device)
                random_lists = generate_random_lists(5)
                x_m = x
                noise = torch.randn(size=[modelConfig["batch_size"], 3, 64, 64], device=device)
                x_n = noise
                o1 = torch.zeros_like(x)
                for time_step in reversed(range(20)):
                    print(time_step)
                    tc = 0
                    x_remain = trainer(x, 20 - time_step)
                    if (((time_step + 1) % 5) == 0) or (time_step == 0):
                        tc = int((time_step + 1) / 5)

                        log, x_m, o1, o2 = cond_fn(x_m, tc, x, random_lists)
                    if time_step == 19:
                        mask = o1
                    x_cond = sampler(x_n, 20 - time_step, log).to(device)

                    x_mix = x_cond * mask + x_remain * (1-mask)
                    x_n = sampler3(x_mix, 20 - time_step).to(device)
                    # if time_step != 0:
                    #     y = x_n * mask + x * (1-mask)
                    # else:u
                    #     y = x_n
                save_image(x_n, os.path.join(save_dir, f"y_{idx}.jpg"))


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
    return cumulative


def main(model_config=None):
    modelConfig = {
        "state": "restore",
        "batch_size": 1,
        "T": 1000,
        "channel": 64,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 1,
        "num_res_blocks1": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 64,
        "grad_clip": 1.,
        "device": "cuda:0",
        "save_weight_dir": r"",
        "ddpm_weight": "",
        "occl_weight": "",
        "sampled_dir": "/",
        "sampledNoisyImgName": "g",
        "sampledImgName": "",
        "nrow": 8
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "restore":
        restore(modelConfig)


if __name__ == '__main__':
    main()










