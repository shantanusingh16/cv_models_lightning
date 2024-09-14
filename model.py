import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import Accuracy, JaccardIndex
from torchmetrics.detection import MeanAveragePrecision

from models.backbones.mobilenetv2 import mobilenetv2
from models.backbones.mobilenetv3 import mobilenetv3
from models.detection.ssd import create_ssd
from models.detection.ssdlite import create_ssdlite
from models.segmentation.deeplabv3 import create_deeplabv3
from models.segmentation.deeplabv3plus import create_deeplabv3plus
from losses.cross_entropy import CrossEntropyLoss
from losses.focal_loss import FocalLoss

class VisionModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.task = config['task']
        self.num_classes = config['num_classes']
        self.learning_rate = config['learning_rate']
        
        # Create model based on task and backbone
        if self.task == 'classification':
            if config['backbone'] == 'mobilenetv2':
                self.model = mobilenetv2(num_classes=self.num_classes, pretrained=config['pretrained'])
            elif config['backbone'] == 'mobilenetv3':
                self.model = mobilenetv3(num_classes=self.num_classes, pretrained=config['pretrained'])
        elif self.task == 'detection':
            if config['detection_head'] == 'ssd':
                self.model = create_ssd(num_classes=self.num_classes, backbone=config['backbone'], pretrained=config['pretrained'])
            elif config['detection_head'] == 'ssdlite':
                self.model = create_ssdlite(num_classes=self.num_classes, backbone=config['backbone'], pretrained=config['pretrained'])
        elif self.task == 'segmentation':
            if config['segmentation_head'] == 'deeplabv3':
                self.model = create_deeplabv3(num_classes=self.num_classes, backbone=config['backbone'], pretrained=config['pretrained'])
            elif config['segmentation_head'] == 'deeplabv3plus':
                self.model = create_deeplabv3plus(num_classes=self.num_classes, backbone=config['backbone'], pretrained=config['pretrained'])
        
        # Set up loss function
        if config['loss'] == 'cross_entropy':
            self.loss_fn = CrossEntropyLoss()
        elif config['loss'] == 'focal':
            self.loss_fn = FocalLoss()
        
        # Set up metrics
        if self.task == 'classification':
            self.train_accuracy = Accuracy()
            self.val_accuracy = Accuracy()
            self.test_accuracy = Accuracy()
        elif self.task == 'detection':
            self.train_map = MeanAveragePrecision()
            self.val_map = MeanAveragePrecision()
            self.test_map = MeanAveragePrecision()
        elif self.task == 'segmentation':
            self.train_iou = JaccardIndex(num_classes=self.num_classes)
            self.val_iou = JaccardIndex(num_classes=self.num_classes)
            self.test_iou = JaccardIndex(num_classes=self.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        if self.task == 'classification':
            self.train_accuracy(y_hat, y)
            self.log('train_acc', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        elif self.task == 'detection':
            self.train_map(y_hat, y)
            self.log('train_map', self.train_map, on_step=True, on_epoch=True, prog_bar=True)
        elif self.task == 'segmentation':
            self.train_iou(y_hat, y)
            self.log('train_iou', self.train_iou, on_step=True, on_epoch=True, prog_bar=True)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        if self.task == 'classification':
            self.val_accuracy(y_hat, y)
            self.log('val_acc', self.val_accuracy, on_epoch=True, prog_bar=True)
        elif self.task == 'detection':
            self.val_map(y_hat, y)
            self.log('val_map', self.val_map, on_epoch=True, prog_bar=True)
        elif self.task == 'segmentation':
            self.val_iou(y_hat, y)
            self.log('val_iou', self.val_iou, on_epoch=True, prog_bar=True)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        if self.task == 'classification':
            self.test_accuracy(y_hat, y)
            self.log('test_acc', self.test_accuracy, on_epoch=True)
        elif self.task == 'detection':
            self.test_map(y_hat, y)
            self.log('test_map', self.test_map, on_epoch=True)
        elif self.task == 'segmentation':
            self.test_iou(y_hat, y)
            self.log('test_iou', self.test_iou, on_epoch=True)
        
        self.log('test_loss', loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }