import os
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, root_dir, image_transform=None, p_transform=None):
        self.root_dir = root_dir
        # self.transform = transform
        self.image_transform = image_transform
        self.p_transform = p_transform
        self.image_paths = sorted(os.listdir(os.path.join(root_dir, 'input')))
        # self.occlusion_paths = sorted(os.listdir(os.path.join(root_dir, 'tree')))
        self.p_paths = sorted(os.listdir(os.path.join(root_dir, 'p')))
        self.x_paths = sorted(os.listdir(os.path.join(root_dir, 'label')))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, 'input', self.image_paths[idx])
        # tree_path = os.path.join(self.root_dir, 'tree', self.occlusion_paths[idx])
        p_path = os.path.join(self.root_dir, 'p', self.p_paths[idx])
        x_path = os.path.join(self.root_dir, 'label', self.x_paths[idx])

        image = Image.open(image_path).convert('RGB')
        # tree = Image.open(tree_path).convert('RGB')
        p = Image.open(p_path).convert('L')
        x = Image.open(x_path).convert('RGB')

        # if self.transform:
        #     image = self.transform(image)
        #     tree = self.transform(tree)
        #     x = self.transform(x)
        if self.image_transform:
            image = self.image_transform(image)
            x = self.image_transform(x)
            # tree = self.image_transform(tree)
        if self.p_transform:
            p = self.p_transform(p)

        return image, x, p





