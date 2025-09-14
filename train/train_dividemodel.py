from models.divide_unet import Model
import os
import torch
from typing import Dict
import torch.nn as nn
import torch.optim as optim
from dataset.devide_model_dataset import CustomDataset
from dataset.createdata import CreateDataset
from dataset.val_dataset import ValDataset
from Scheduler import GradualWarmupScheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from math import exp
from torchvision.utils import save_image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import random


def train(modelConfig: Dict):
    image_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.338722, 0.346459, 0.337364], std=[0.267317, 0.263739, 0.268044])
    ])

    p_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    dataset = CustomDataset(root_dir='', image_transform=image_transforms,
                            p_transform=p_transforms)
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=2, drop_last=True)

    val_dataset = ValDataset(root_dir='', image_transform=image_transforms,
                             p_transform=p_transforms)
    val_dataloader = DataLoader(
        val_dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=2, drop_last=True)

    net_model = Model(T=modelConfig["T"], ch=modelConfig["ch"], ch_mult=modelConfig["ch_mult"],
                      num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"],
                      attn=modelConfig["attn"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))

    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)

    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10,
        after_scheduler=cosineScheduler)
    SSIM = SSIMLoss()
    MSE = nn.MSELoss()
    L1 = nn.L1Loss()

    for e in range(modelConfig["epoch"]):
        total_loss = 0.0
        net_model.train()
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for batch_idx, (image, y, p) in enumerate(tqdmDataLoader):
                optimizer.zero_grad()
                B, C, H, W = image.shape
                random_lists = generate_random_lists(B, 5)
                input = image.to(device)
                label = y.to(device)
                value = torch.randint(0, 5, (1,)).item()
                t = torch.full((B,), value).to(device)
                # t = torch.randint(0, 5, (B,)).to(device)
                p = p.to(device)
                for i in range(B):
                    input[i] = input[i] - (input[i] - label[i]) * random_lists[i][t[i]] * p[i]
                pred_mask, pred_pixel = net_model(input, t)
                pred = input * (1 - pred_mask) + pred_pixel * pred_mask

                loss1 = L1(pred_mask, p)
                loss2 = MSE(pred_pixel, label)
                loss3 = 1 - SSIM(pred, label)
                loss = ((value + 4) / 4) * loss1 + loss2 + loss3
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                total_loss += loss.item()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "totol_loss: ": total_loss,
                    "loss: ": loss,
                    "input shape: ": input.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {e + 1}/{modelConfig["epoch"]}, Loss: {avg_loss}')

        net_model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for image, y, p in val_dataloader:
                random_lists = generate_random_lists(B, 5)
                input = image.to(device)
                label = y.to(device)
                value = torch.randint(0, 5, (1,)).item()
                t = torch.full((B,), value).to(device)
                p = p.to(device)
                for i in range(B):
                    input[i] = input[i] - (input[i] - label[i]) * random_lists[i][t[i]] * p[i]

                pred_mask, pred_pixel = net_model(input, t)
                pred = input * (1 - pred_mask) + pred_pixel * pred_mask

                loss1 = L1(pred_mask, p)
                loss2 = MSE(pred_pixel, y)
                loss3 = 1 - SSIM(pred, y)

                loss = ((value + 4) / 4) * loss1 + loss2 + loss3
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f'Validation Loss: {avg_val_loss}')

        warmUpScheduler.step()

        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_weight_dir"], 'ckpt_' + str(e) + '_.pt'))


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=7, window=None, size_average=True, full=False):
    if window is None:
        channel = img1.size(1)
        window = create_window(window_size, channel)

    window = window.to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.size(1))

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.size(1)) - mu1_mu2

    C1 = 0.1 ** 2
    C2 = 0.3 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(nn.Module):
    def __init__(self, window_size=7, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
        # return 1 - ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


# def generate_random_lists(batch_size, length):
#     lists = []
#
#     for _ in range(batch_size):
#         random_list = [random.random() for _ in range(length)]
#         random_list.sort(reverse=True)
#         total_sum = sum(random_list)

#         normalized_list = [value / total_sum for value in random_list]
#         lists.append(normalized_list)
#
#     cumulative = []
#     for lst in lists:
#         cum_sum = [lst[-1]]
#         for j in range(len(lst) - 2, -1, -1):
#             cum_sum.append(cum_sum[-1] + lst[j])
#         cumulative.append(cum_sum[::-1])
#
#     return cumulative

from itertools import accumulate

def generate_random_lists(batch_size, length):
    lists = []

    for _ in range(batch_size):
        random_list = sorted([random.random() for _ in range(length)], reverse=False)
        total_sum = sum(random_list)
        normalized_list = [value / total_sum for value in random_list]

        cumulative_list = [0] * length
        cumulative_list[-1] = 0
        cumulative_list[-2] = normalized_list[-1]

        for i in range(length - 3, -1, -1):
            cumulative_list[i] = cumulative_list[i + 1] + normalized_list[i + 1]

        lists.append(cumulative_list)

    return lists

def createdata(modelConfig: Dict):
    # Data transforms
    data_transforms = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    dataset = CreateDataset(root_dir=r'', transform=data_transforms)
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

    save_dir_x = r""
    with torch.no_grad():
        for e in range(modelConfig["epoch"]):
            with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
                for batch_idx, (image, tree, p) in enumerate(tqdmDataLoader):
                    batch, C, H, W = image.shape
                    input = image.to(device)
                    O = tree.to(device)
                    x = torch.zeros_like(image).to(device)
                    P = p.to(device)
                    for i in range(batch):
                        x[i] = input[i] + (O[i] - input[i]) * P[i]
                        filename_x = os.path.join(save_dir_x, f"{batch_idx}.png")
                        save_image(x[i], filename_x)


def main(model_config=None):
    modelConfig = {
        "state": "train",  # train or eval
        "epoch": 300,
        "T": 5,
        "image_size": 64,
        "batch_size": 4,
        "ch": 64,
        "ch_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.2,
        "lr": 1e-4,
        "multiplier": 2.,
        "grad_clip": 1.,
        "device": "cuda:0",
        "training_load_weight": None,
        "testing_load_weight": "",
        "save_weight_dir": r"",
        "sampled_dir": r"",
        "sampledNoisyImgName": "",
        "sampledImgName": "",
        "nrow": 8,
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    if modelConfig["state"] == "createdata":
        createdata(modelConfig)


if __name__ == '__main__':
    main()




