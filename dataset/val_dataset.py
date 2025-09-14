import os
from PIL import Image
from torch.utils.data import Dataset


class ValDataset(Dataset):
    def __init__(self, root_dir, image_transform=None, p_transform=None):
        self.root_dir = root_dir
        # self.transform = transform
        self.image_transform = image_transform
        self.p_transform = p_transform

        self.image_paths = sorted(os.listdir(os.path.join(root_dir, 'input')))
        self.occlusion_paths = sorted(os.listdir(os.path.join(root_dir, 'p')))
        self.x_paths = sorted(os.listdir(os.path.join(root_dir, 'label')))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, 'input', self.image_paths[idx])
        tree_path = os.path.join(self.root_dir, 'p', self.occlusion_paths[idx])
        x_path = os.path.join(self.root_dir, 'label', self.x_paths[idx])

        image = Image.open(image_path).convert('RGB')
        tree = Image.open(tree_path).convert('L')
        x = Image.open(x_path).convert('RGB')

        if self.image_transform:
            image = self.image_transform(image)
            x = self.image_transform(x)
        if self.p_transform:
            tree = self.p_transform(tree)

        return image, tree, x




