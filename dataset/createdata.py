import os
from PIL import Image
from torch.utils.data import Dataset


class CreateDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = sorted(os.listdir(os.path.join(root_dir, 'label')))
        self.occlusion_paths = sorted(os.listdir(os.path.join(root_dir, 'tree')))
        self.x_paths = sorted(os.listdir(os.path.join(root_dir, 'p')))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, 'label', self.image_paths[idx])
        tree_path = os.path.join(self.root_dir, 'tree', self.occlusion_paths[idx])
        x_path = os.path.join(self.root_dir, 'p', self.x_paths[idx])

        image = Image.open(image_path).convert('RGB')
        tree = Image.open(tree_path).convert('RGB')
        x = Image.open(x_path).convert('L')

        if self.transform:
            image = self.transform(image)
            tree = self.transform(tree)
            x = self.transform(x)

        return image, tree, x


