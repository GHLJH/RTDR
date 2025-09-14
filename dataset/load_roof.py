from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import torch
from torchvision import transforms
import numpy as np
import math


class CreateDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_index = self.images[index]
        img_path = os.path.join(self.root_dir, image_index)
        img = Image.open(img_path)
        label = int(image_index[
                        -5])

        if self.transform:
            sample = self.transform(img)
        return sample, label


def load_dataset():
    data_transforms = [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
    data_transforms = transforms.Compose(data_transforms)
    train = CreateDataset('', transform=data_transforms)
    return train


def load_image():
    data_transforms = [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.338722, 0.346459, 0.337364], std=[0.267317, 0.263739, 0.268044])
    ]
    data_transforms = transforms.Compose(data_transforms)
    image = CreateDataset('', transform=data_transforms)
    return image







