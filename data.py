import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_lightning import LightningDataModule
import os
from PIL import Image
import yaml
import numpy as np
import cv2

class VisionDataset(Dataset):
    def __init__(self, root_dir, transform=None, task='classification', detection_dir=None, segmentation_dir=None, config_path=None):
        self.root_dir = root_dir
        self.transform = transform
        self.task = task
        self.detection_dir = detection_dir
        self.segmentation_dir = segmentation_dir
        self.config_path = config_path
        self.scale = self._get_scale_from_config()
        self.image_paths, self.labels, self.detection_paths, self.segmentation_paths = self._load_dataset()

    def _get_scale_from_config(self):
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('segmentation_scale', 1)
        return 1

    def _load_dataset(self):
        image_paths = []
        labels = []
        detection_paths = []
        segmentation_paths = []
        class_dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        for class_idx, class_dir in enumerate(class_dirs):
            class_path = os.path.join(self.root_dir, class_dir)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                image_paths.append(img_path)
                labels.append(class_idx)
                
                if self.detection_dir:
                    det_path = os.path.join(self.detection_dir, f"{os.path.splitext(img_name)[0]}.txt")
                    detection_paths.append(det_path if os.path.exists(det_path) else None)
                
                if self.segmentation_dir:
                    seg_path = os.path.join(self.segmentation_dir, f"{os.path.splitext(img_name)[0]}.png")
                    segmentation_paths.append(seg_path if os.path.exists(seg_path) else None)
        
        return image_paths, labels, detection_paths, segmentation_paths

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
            detections = self._load_detections(idx)
            return image, detections
        elif self.task == 'segmentation':
            mask = self._load_segmentation_mask(idx)
            return image, mask
        elif self.task == 'detection_segmentation':
            detections = self._load_detections(idx)
            mask = self._load_segmentation_mask(idx)
            return image, detections, mask

    def _load_detections(self, idx):
        if self.detection_paths[idx]:
            with open(self.detection_paths[idx], 'r') as f:
                lines = f.readlines()
            detections = [list(map(float, line.strip().split())) for line in lines]
            return torch.tensor(detections)
        return torch.tensor([])

    def _load_segmentation_mask(self, idx):
        if self.segmentation_paths[idx]:
            mask = cv2.imread(self.segmentation_paths[idx], cv2.IMREAD_GRAYSCALE)
            mask = (mask / self.scale).astype(np.int64)
            return torch.from_numpy(mask)
        return torch.tensor([])

class VisionDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = config['data_dir']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.task = config['task']
        self.detection_dir = config.get('detection_dir')
        self.segmentation_dir = config.get('segmentation_dir')
        self.config_path = config.get('config_path')

    def setup(self, stage=None):
        # Define transforms
        self.transform = self._get_transforms()

        # Create datasets
        self.train_dataset = VisionDataset(os.path.join(self.data_dir, 'train'), self.transform, self.task,
                                           self.detection_dir, self.segmentation_dir, self.config_path)
        self.val_dataset = VisionDataset(os.path.join(self.data_dir, 'val'), self.transform, self.task,
                                         self.detection_dir, self.segmentation_dir, self.config_path)
        self.test_dataset = VisionDataset(os.path.join(self.data_dir, 'test'), self.transform, self.task,
                                          self.detection_dir, self.segmentation_dir, self.config_path)

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