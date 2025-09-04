# MatSegNet: An Attention U-Net for Material Segmentation

MatSegNet is a deep learning model designed for precise semantic segmentation of materials, particularly in scientific imaging applications like microscopy. It uses a powerful Attention U-Net architecture with a pre-trained ResNet-34 backbone to achieve high accuracy.

A key feature of MatSegNet is its multi-task learning approach, where it simultaneously predicts both the segmentation mask and the material boundaries (edges). This encourages the model to learn more robust and accurate representations, leading to sharper and more precise segmentation results.



## Features

-   **Architecture**: Implements an Attention U-Net, which uses attention gates on skip connections to focus on relevant features and suppress noise.
-   **Pre-trained Backbone**: Utilizes a ResNet-34 encoder pre-trained on ImageNet for robust feature extraction (Transfer Learning).
-   **Multi-Task Learning**: Outputs both a segmentation mask and an edge prediction, improving overall boundary delineation.
-   **Configurable Training**: The entire training process is managed via a `config.yaml` file, allowing for easy experimentation with hyperparameters, loss functions, and training strategies.
-   **Two-Stage Training**: Supports an optional two-stage training strategy to first train the decoder head and then fine-tune the entire network, leading to more stable convergence.

## Project Structure

```
Segmentation_Pipeline/
├── configs/                 
│   └── config.py
│         └── MatSegNet.yaml
│         └── Segformer.yaml
│         └── Unet.yaml
├── data/                     
│   ├── SEM_images/           
│   ├── datasets/              
│  		└──bainite_set
│		└──martensite_set
│		└──training_set
│		└──validation_set
│		└──test_set
├── models/   
│   ├──MatSegNet.py
│   ├──Segformer.py
│   ├──Unet.py
├── output/                  
│   ├── checkpoints/
│         └──best_matsegnet.pth
│         └──best_segformer.pth
│         └──best_unet_mobilenetv2.pth
│         └──matsegnet.pth
│         └──segformer.pth
│         └──unet_mobilenetv2.pth
│   └── accuracy_output/     
├── src/            
│   ├── datasets/
│         └──preprocessing.py
│         └──load_data.py
│         └──checkpoints.py
│         └──training.py
│         └──visualization.py
├── scripts/
│         └──segment_images.py
│         └──train_test_split.py
│         └──train.py
│         └──visualize_results.py
│         └──carbide_morphology.py
│         └──size_aspect_ratio.py


## Getting Started

### Prerequisites

numpy==2.0.2 
albumentations==2.0.8
matplotlib==3.10.5
opencv-python-headless==4.10.0.84
Pillow==11.3.0
PyYAML==6.0.2
scikit-learn==1.0.2
torch==2.5.1+cu121
torchvision==0.20.1+cu121
tqdm==4.66.5
transformers==4.55.0

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Ericalyhan94/Microstructure-Pipeline.git
    cd MatSegNet
    ```

2.  (Recommended) Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    *(TODO: You should create a `requirements.txt` file by running `pip freeze > requirements.txt` in your environment)*

### Data Preparation

1.  Organize your dataset into the following structure inside the `data/datasets/` directory:
    ```
    datasets/
    ├── train/
    │   ├── images/
    │   ├── masks/
    │   └── edges/  <-- (Required for MatSegNet)
    ├── validation/
    │   ├── images/
    │   ├── masks/
    │   └── edges/
    └── test/
        ├── images/
        ├── masks/
        └── edges/
    ```
2.  Ensure that for each image, the corresponding mask and edge files share the same name.

## Training

The entire training process is controlled by the `configs/MatSegNet.yaml` file. You can adjust the learning rate, batch size, number of epochs, loss functions, and more within this file.

To start training, run the `train.py` script:

```bash
python src/train.py --model MatSegNet
```

-   `--config`: Path to the configuration file.
-   `--checkpoint`: Which checkpoint to load ('best' or 'newest'). The training will resume from there if a checkpoint is found.

The script will automatically handle:
-   Loading the configuration.
-   Preparing the datasets and dataloaders.
-   Building the MatSegNet model.
-   Executing the standard or two-stage training loop.
-   Saving model checkpoints to the `outputs/checkpoints/` directory.

## Results


**Performance Metrics:**

| Metric      | Value |
| :---------- | :---- |
| F1-Score    | 0.92  |
| Accuracy    | 0.98  |
| Recall      | 0.91  |
| Precision   | 0.93  |


## Acknowledgements

-   This model architecture is based on the **Attention U-Net** paper: [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999).
-   The encoder implementation uses the pre-trained models provided by **PyTorch's `torchvision`**.