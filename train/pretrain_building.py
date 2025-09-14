import os
from typing import Dict
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from models.ddpm import GaussianDiffusionSampler, GaussianDiffusionTrainer, ImplicitDiffusionSampler
from models.ddpm_unet import UNet
from Scheduler import GradualWarmupScheduler
from dataset.load_roof import load_dataset, load_image


def train(modelConfig: Dict):

    device = torch.device(modelConfig["device"])
    dataset = load_dataset()
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
    # model setup
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 30, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                # save_image(x_0, '1.png')
                loss = trainer(x_0).sum()
                loss.requires_grad_(True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))


# def eval(modelConfig: Dict):
#     with torch.no_grad():
#         device = torch.device(modelConfig["device"])
#         model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
#                      num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
#         ckpt = torch.load(os.path.join(
#             modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
#         model.load_state_dict(ckpt)
#         print("model load weight done.")
#         model.eval()
#         sampler = GaussianDiffusionSampler(
#             model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
#         noisyImage = torch.randn(
#             size=[modelConfig["batch_size"], 3, 64, 64], device=device)
#         save_image(noisyImage, os.path.join(
#             modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
#         sampledImgs = sampler(noisyImage)
#         save_image(sampledImgs, os.path.join(
#             modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])

def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()

        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        # Sampled from standard normal distribution
        # noisyImage = torch.randn(
        #     size=[modelConfig["batch_size"], 3, 64, 64], device=device)
        # # saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        # save_image(noisyImage, os.path.join(
        #     modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        # sampledImgs = sampler(noisyImage)
        # # sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        # save_image(sampledImgs, os.path.join(
        #     modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])
        save_dir_x = r""
        for e in range(modelConfig["epoch"]):
            noisyImage = torch.randn(
                size=[modelConfig["batch_size"], 3, 64, 64], device=device)
            sampledImgs = sampler(noisyImage)
            B,C,H,W = noisyImage.shape
            for i in range(B):
                filename_x = os.path.join(save_dir_x, f"{e}_{i}.png")
                save_image(sampledImgs[i], filename_x)

def main(model_config=None):
    modelConfig = {
        "state": "eval",  # train or eval
        "epoch": 500,
        "batch_size": 24,
        "T": 1000,
        "channel": 64,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 256,
        "grad_clip": 1.,
        "device": "cuda:0",
        "training_load_weight": None,
        "save_weight_dir": "",
        "test_load_weight": "",
        "sampled_dir": "",
        "sampledNoisyImgName": "",
        "sampledImgName": "",
        "pre_noiseName": "",
        "nrow": 8
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    if modelConfig["state"] == "eval":
        eval(modelConfig)


if __name__ == '__main__':
    main()