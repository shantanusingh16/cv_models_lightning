import os
import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from data import VisionDataModule
from model import VisionModel

def main(config_path):
    # Load configuration
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Create data module
    data_module = VisionDataModule(config)

    # Create model
    model = VisionModel(config)

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )

    # Set up logger
    logger = TensorBoardLogger('logs', name=config['experiment_name'])

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        gpus=config['gpus'],
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=50,
        deterministic=True
    )

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, datamodule=data_module)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a vision model')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    main(args.config)