import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import AttributeDict

from data import VisionDataModule
from model import VisionModel

def main(config_path, checkpoint_path):
    # Load configuration
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    config = AttributeDict(config)

    # Create data module
    data_module = VisionDataModule(config)

    # Load model from checkpoint
    model = VisionModel.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()

    # Create trainer
    trainer = pl.Trainer(gpus=config.gpus if torch.cuda.is_available() else None)

    # Evaluate the model
    results = trainer.test(model, datamodule=data_module)

    # Print results
    print("Evaluation Results:")
    for k, v in results[0].items():
        print(f"{k}: {v}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a vision model')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    args = parser.parse_args()

    main(args.config, args.checkpoint)