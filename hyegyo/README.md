# Document Image Enhancement and Classification with Augraphy and SwinT

This repository contains the implementation of a document image enhancement and classification pipeline using `Augraphy` for data augmentation and `SwinT` (Swin Transformer) for image classification.

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [About Augraphy](#about-augraphy)
- [About SwinT](#about-swint)
- [Data Augmentation with Augraphy](#data-augmentation-with-augraphy)
- [Model Training with SwinT](#model-training-with-swint)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to enhance and classify document images using state-of-the-art techniques. We use `Augraphy` to generate augmented versions of document images, simulating real-world distortions such as noise, blur, and folds. The augmented images are then used to train a `SwinT` model, a powerful transformer-based model designed for image classification tasks.

## Prerequisites

- Python 3.8 or higher
- PyTorch 1.8.0 or higher
- CUDA (if using GPU)
- Git

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/your_username/your_repository.git
    cd your_repository
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Install `Augraphy`:
    ```bash
    pip install augraphy
    ```

4. Install `SwinT`:
    ```bash
    pip install timm  # timm package provides the implementation of SwinT
    ```

## About Augraphy
Augraphy is a Python library designed to simulate a wide range of document degradations and printing artifacts. It is particularly useful for creating training datasets for machine learning models that need to handle real-world document images. By applying various augmentations like noise, blur, folds, stains, and ink bleed, Augraphy helps in generating realistic training data that improves the robustness of models in handling noisy and degraded documents.

Key features of Augraphy:

- Simulates realistic document degradations.
- Provides a variety of augmentations to customize the output.
- Easy to integrate into data pipelines for training deep learning models.

  ![image](https://github.com/user-attachments/assets/bacc90da-2339-46b2-a908-4d821f281cf6)
  ![image](https://github.com/user-attachments/assets/508d99d7-da87-4a2d-b0fb-123d37f2257e)


## About SwinT
SwinT (Swin Transformer) is a transformer-based architecture introduced by Microsoft Research for image classification tasks. Unlike traditional convolutional neural networks (CNNs), SwinT uses a hierarchical design that splits the input image into non-overlapping windows, applying self-attention within each window. This approach allows SwinT to efficiently model long-range dependencies and achieve state-of-the-art performance on various benchmarks.

Key features of SwinT:

- Hierarchical transformer design with shifted windows.
- Efficient computation with reduced memory consumption.
- Scalable to different image sizes and resolutions.
- Achieves competitive performance on image classification and other vision tasks.

![image](https://github.com/user-attachments/assets/e8770f19-f34d-46ba-9297-89000c1fc0bf)


## Data Augmentation with Augraphy

`Augraphy` is used to create a variety of document image augmentations that simulate common distortions in scanned documents. These augmentations include:

- Noise
- Blur
- Folds
- Stains
- Ink Bleed
- And more...

Example of applying Augraphy:

```python
from augraphy import AugraphyPipeline

pipeline = AugraphyPipeline(augmentations=[
    # List of augmentations
])


## Data Augmentation with Augraphy
augmented_images = pipeline.augment(images)
```

## Model Training with SwinT
The SwinT model is a transformer-based architecture that has shown excellent performance on various image classification tasks. We leverage this model to classify document images after augmentation.


## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
