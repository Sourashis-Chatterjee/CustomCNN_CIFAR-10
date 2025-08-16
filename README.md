# CustomCNN_CIFAR-10
Custom built CNN for CIFAR-10 dataset classification
## CIFAR-10 CNN Classification Report

## Overview

This project implements a custom Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset. The goal was to understand and improve model accuracy using modern deep learning techniques and best practices.
---

## Model Architecture

- **Input:** 32x32 RGB images
- **Layers:**
  - **4 Convolutional Layers**: Increasing filter sizes, each followed by Batch Normalization and ReLU activation
  - **Max Pooling:** After the last conv block
  - **Dropout:** Applied after the fully connected layer (`dropout=0.3`)
  - **Fully Connected Layer:** Final classification into 10 classes

---
## Techniques Used

### 1. **Batch Normalization**
- Normalizes activations in each mini-batch to improve training speed and stability.
- Added after every convolutional layer.

### 2. **Dropout**
- Randomly disables 30% of neurons in the fully connected layer during training.
- Reduces overfitting and encourages robust feature learning.

### 3. **Data Augmentation**
- Applied to training images to increase diversity and prevent overfitting:
  - **RandomCrop(32, padding=4):** Randomly crops and pads images.
  - **RandomHorizontalFlip():** Randomly flips images left-right.

### 4. **Training Improvements**
- **Epochs:** Increased from 10 to 40 for better convergence.
- **Optimizer:** Used Adam optimizer.
- **Learning Rate:** 0.001.
- **Batch Size:** Set to 64).

---

## Results

- **Final Test Accuracy:** **77%**
- **Training Details:**
  - 4 convolutional layers
  - Dropout (0.3)
  - Batch normalization
  - Data augmentation (random crop and flip)
  - 40 epochs

- **Performance Analysis:**
  - Model shows good generalization with limited overfitting.
  - Some classes are more challenging (refer to the confusion matrix in code).
  - Custom architecture is competitive for baseline models.
  - State-of-the-art models (e.g., ResNet, VGG) can exceed 90% on CIFAR-10.

---

## Limitations & Next Steps

- Accuracy plateaued at 77%. For further improvement we can:
  - Use deeper architectures (VGG, ResNet, DenseNet).
  - Implement learning rate scheduling.
  - Apply more advanced data augmentation.
  - Try transfer learning or pretrained models.
  - Run more epochs or tune hyperparameters further.
---

**Sourashis Chatterjee**  
[GitHub](https://github.com/Sourashis-Chatterjee)
