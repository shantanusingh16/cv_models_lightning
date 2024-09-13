# CB Vision Models

This project implements various computer vision models using PyTorch, timm (PyTorch Image Models), and PyTorch Lightning, including:

- Backbones: MobileNetV2, MobileNetV3
- Detection heads: SSD, SSDLite
- Segmentation heads: DeepLabV3, DeepLabV3+
- Losses: Cross-entropy loss, Focal loss

It also includes training, evaluation, and ONNX export scripts.

## Project Structure

```
cb_vision_models/
├── models/
│   ├── backbones/
│   │   ├── mobilenetv2.py
│   │   └── mobilenetv3.py
│   ├── detection/
│   │   ├── ssd.py
│   │   └── ssdlite.py
│   └── segmentation/
│       ├── deeplabv3.py
│       └── deeplabv3plus.py
├── losses/
│   ├── cross_entropy.py
│   └── focal_loss.py
├── configs/
│   └── default_config.yaml
├── data.py
├── model.py
├── train.py
├── evaluate.py
├── export_onnx.py
└── README.md
```

## Setup and Usage

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Update the `configs/default_config.yaml` file with your desired settings and data paths.

3. Prepare your dataset in the following structure:
   ```
   data_dir/
   ├── train/
   │   ├── class1/
   │   ├── class2/
   │   └── ...
   ├── val/
   │   ├── class1/
   │   ├── class2/
   │   └── ...
   └── test/
       ├── class1/
       ├── class2/
       └── ...
   ```

4. Run the training script:
   ```
   python train.py --config configs/default_config.yaml
   ```

   This will start the training process using the specified configuration. The script will create checkpoints and log files in the `checkpoints/` and `logs/` directories, respectively.

5. To evaluate the model, use the `evaluate.py` script:
   ```
   python evaluate.py --config configs/default_config.yaml --checkpoint path/to/checkpoint.ckpt
   ```

   This will load the model from the specified checkpoint and evaluate it on the test dataset.

6. To export the model to ONNX format, use the `export_onnx.py` script:
   ```
   python export_onnx.py --config configs/default_config.yaml --checkpoint path/to/checkpoint.ckpt --output path/to/model.onnx
   ```

   This will load the model from the specified checkpoint, convert it to ONNX format, and save it to the specified output path.

## Configuration

The `default_config.yaml` file contains all the necessary parameters for training, including:

- Data configuration (data directory, batch size, number of workers)
- Model configuration (task, backbone, detection/segmentation head, number of classes)
- Training configuration (number of epochs, learning rate, GPU usage)
- Loss function
- Augmentation settings

Modify this file to adjust the training process to your needs.

## Implemented Features

- Flexible data module supporting classification, detection, and segmentation tasks
- Model implementation supporting various backbones and heads
- Training script with logging, checkpointing, and early stopping
- Evaluation script for model testing
- ONNX export script for model deployment
- Configuration-based model and training setup

## Next Steps

1. Add support for custom datasets and data formats
2. Implement additional backbone models and detection/segmentation heads
3. Add unit tests for each component
4. Improve documentation and add usage examples for each task type
5. Implement a demo script to showcase model inference on sample images

## Contributing

Contributions to improve and expand the project are welcome. Please submit pull requests for review.

## License

Apache License Version 2.0