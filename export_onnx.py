import argparse
import yaml
import torch
import onnx
from onnxsim import simplify
from pytorch_lightning.utilities.parsing import AttributeDict

from model import VisionModel

def export_onnx(model, config, output_path):
    # Set the model to evaluation mode
    model.eval()

    # Create a dummy input tensor
    if config.task == 'classification':
        dummy_input = torch.randn(1, 3, 224, 224)
    elif config.task == 'detection':
        dummy_input = torch.randn(1, 3, 300, 300)  # Adjust size if needed for your SSD implementation
    elif config.task == 'segmentation':
        dummy_input = torch.randn(1, 3, 513, 513)  # Common size for DeepLabV3, adjust if needed

    # Export the model
    torch.onnx.export(model,
                      dummy_input,
                      output_path,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})

    # Simplify the ONNX model
    onnx_model = onnx.load(output_path)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, output_path)

    print(f"ONNX model exported and simplified successfully: {output_path}")

def main(config_path, checkpoint_path, output_path):
    # Load configuration
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    config = AttributeDict(config)

    # Load model from checkpoint
    model = VisionModel.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()

    # Export the model to ONNX
    export_onnx(model, config, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export a vision model to ONNX')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Path to save the ONNX model')
    args = parser.parse_args()

    main(args.config, args.checkpoint, args.output)