import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_lightning import LightningDataModule
import os
from PIL import Image
import yaml

class VisionDataset(Dataset):
    def __init__(self, root_dir, transform=None, task='classification'):
        self.root_dir = root_dir
        self.transform = transform
        self.task = task
        self.image_paths, self.labels = self._load_dataset()

    def _load_dataset(self):
        image_paths = []
        labels = []
        class_dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        for class_idx, class_dir in enumerate(class_dirs):
            class_path = os.path.join(self.root_dir, class_dir)
            for img_name in os.listdir(class_path):
                image_paths.append(os.path.join(class_path, img_name))
                labels.append(class_idx)
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.task == 'classification':
            return image, label
        elif self.task == 'detection':
            # TODO: Implement bounding box loading
            return image, label
        elif self.task == 'segmentation':
            # TODO: Implement segmentation mask loading
            return image, label

class VisionDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = config['data_dir']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.task = config['task']

    def setup(self, stage=None):
        # Define transforms
        self.transform = self._get_transforms()

        # Create datasets
        self.train_dataset = VisionDataset(os.path.join(self.data_dir, 'train'), self.transform, self.task)
        self.val_dataset = VisionDataset(os.path.join(self.data_dir, 'val'), self.transform, self.task)
        self.test_dataset = VisionDataset(os.path.join(self.data_dir, 'test'), self.transform, self.task)

    def _get_transforms(self):
        aug_config = self.config['augmentations']
        transform_list = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

        if aug_config['horizontal_flip']:
            transform_list.insert(1, transforms.RandomHorizontalFlip())
        if aug_config['vertical_flip']:
            transform_list.insert(1, transforms.RandomVerticalFlip())
        if aug_config['random_crop']['enabled']:
            size = tuple(aug_config['random_crop']['size'])
            transform_list.insert(1, transforms.RandomCrop(size))
        if aug_config['color_jitter']['enabled']:
            jitter_config = aug_config['color_jitter']
            transform_list.insert(1, transforms.ColorJitter(
                brightness=jitter_config['brightness'],
                contrast=jitter_config['contrast'],
                saturation=jitter_config['saturation'],
                hue=jitter_config['hue']
            ))

        return transforms.Compose(transform_list)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)