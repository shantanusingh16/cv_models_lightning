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
from losses.smooth_l1_loss import SmoothL1Loss
from losses.dice_loss import DiceLoss

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
        
        # Set up loss functions
        self.loss_fns = nn.ModuleDict()
        for task, losses in config['losses'].items():
            self.loss_fns[task] = nn.ModuleList()
            for loss in losses:
                if isinstance(loss, dict):
                    loss_type = loss['type']
                    loss_weight = loss['weight']
                    loss_params = loss.get('params', {})
                else:
                    loss_type = loss
                    loss_weight = 1.0
                    loss_params = {}
                
                if loss_type == 'cross_entropy':
                    self.loss_fns[task].append((CrossEntropyLoss(**loss_params), loss_weight))
                elif loss_type == 'focal':
                    self.loss_fns[task].append((FocalLoss(**loss_params), loss_weight))
                elif loss_type == 'smooth_l1':
                    self.loss_fns[task].append((SmoothL1Loss(**loss_params), loss_weight))
                elif loss_type == 'dice':
                    self.loss_fns[task].append((DiceLoss(**loss_params), loss_weight))
        
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

    def compute_loss(self, y_hat, y, valid_lengths=None):
        total_loss = 0
        if self.task == 'classification':
            for loss_fn, weight in self.loss_fns['classification']:
                total_loss += weight * loss_fn(y_hat, y)
        elif self.task == 'detection':
            cls_preds, bbox_preds = y_hat
            cls_targets, bbox_targets = y
            for loss_fn, weight in self.loss_fns['detection']['classification']:
                total_loss += weight * loss_fn(cls_preds, cls_targets, valid_lengths)
            for loss_fn, weight in self.loss_fns['detection']['localization']:
                total_loss += weight * loss_fn(bbox_preds, bbox_targets, valid_lengths)
        elif self.task == 'segmentation':
            for loss_fn, weight in self.loss_fns['segmentation']:
                total_loss += weight * loss_fn(y_hat, y)
        return total_loss

    def prepare_detection_data(self, y_hat, y, valid_lengths):
        cls_preds, bbox_preds = y_hat
        cls_targets, bbox_targets = y
        
        preds = []
        targets = []
        
        for i in range(len(valid_lengths)):
            # Prepare predictions (use all predictions)
            pred_boxes = bbox_preds[i]
            pred_scores = cls_preds[i].softmax(dim=-1)
            pred_labels = pred_scores.argmax(dim=-1)
            preds.append({
                'boxes': pred_boxes,
                'scores': pred_scores.max(dim=-1)[0],
                'labels': pred_labels
            })
            
            # Prepare targets (use only valid detections)
            length = valid_lengths[i]
            target_boxes = bbox_targets[i, :length]
            target_labels = cls_targets[i, :length]
            targets.append({
                'boxes': target_boxes,
                'labels': target_labels
            })
        
        return preds, targets

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y_hat = self(x)
        
        if self.task == 'classification':
            y = batch['label']
            loss = self.compute_loss(y_hat, y)
            self.train_accuracy(y_hat, y)
            self.log('train_acc', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        elif self.task == 'detection':
            y = batch['detection']
            valid_lengths = batch['valid_lengths']
            loss = self.compute_loss(y_hat, y, valid_lengths)
            preds, targets = self.prepare_detection_data(y_hat, y, valid_lengths)
            self.train_map(preds, targets)
            self.log('train_map', self.train_map, on_step=True, on_epoch=True, prog_bar=True)
        elif self.task == 'segmentation':
            y = batch['mask']
            loss = self.compute_loss(y_hat, y)
            self.train_iou(y_hat, y)
            self.log('train_iou', self.train_iou, on_step=True, on_epoch=True, prog_bar=True)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y_hat = self(x)
        
        if self.task == 'classification':
            y = batch['label']
            loss = self.compute_loss(y_hat, y)
            self.val_accuracy(y_hat, y)
            self.log('val_acc', self.val_accuracy, on_epoch=True, prog_bar=True)
        elif self.task == 'detection':
            y = batch['detection']
            valid_lengths = batch['valid_lengths']
            loss = self.compute_loss(y_hat, y, valid_lengths)
            preds, targets = self.prepare_detection_data(y_hat, y, valid_lengths)
            self.val_map(preds, targets)
            self.log('val_map', self.val_map, on_epoch=True, prog_bar=True)
        elif self.task == 'segmentation':
            y = batch['mask']
            loss = self.compute_loss(y_hat, y)
            self.val_iou(y_hat, y)
            self.log('val_iou', self.val_iou, on_epoch=True, prog_bar=True)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x = batch['image']
        y_hat = self(x)
        
        if self.task == 'classification':
            y = batch['label']
            loss = self.compute_loss(y_hat, y)
            self.test_accuracy(y_hat, y)
            self.log('test_acc', self.test_accuracy, on_epoch=True)
        elif self.task == 'detection':
            y = batch['detection']
            valid_lengths = batch['valid_lengths']
            loss = self.compute_loss(y_hat, y, valid_lengths)
            preds, targets = self.prepare_detection_data(y_hat, y, valid_lengths)
            self.test_map(preds, targets)
            self.log('test_map', self.test_map, on_epoch=True)
        elif self.task == 'segmentation':
            y = batch['mask']
            loss = self.compute_loss(y_hat, y)
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