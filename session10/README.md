# Submission for Week 10

- [Problem Statement](#Problem-Statement)
- [File Structure](#File-Structure)
- [Applied Augmentation](#Applied-Augmentation)
- [Model Parameters](#Model-Parameters)
- [Receptive Field and Output Shape Calculation of Best Model](#Receptive-Field-and-Output-Shape-Calculation-of-Best-Model)
- [Training Logs](#Training-Logs)
- [Results](#Results)
  * [Accuracy](#Accuracy)
  * [Accuracy Plot](#Accuracy-Plot)


# Problem Statement

### Training CNN for CIFAR Dataset

1. Create Custom resnet architecture
2. Use one cycle policy for LR
3. Get at least 90% accuracy

# File Structure

* session10
  * Contains all the code required during training in different modules
    * models --> model.py -> Contains the model architecture
    * augmentation --> augmentation.py --> contains albumentations transforms
    * dataloader --> dataloader.py --> contains dataloaders for CIFAR10
    * train.py -> contains functions for training, testing
    * main.ipynb --> main notebook containing training logs




# Model Parameters

The Model Uses [***Dilated Convolution and Depthwise Separable Convolution***] 

<p align="center">
    <img src="images/architecture.JPG" alt="centered image" />
</p>

 
# Training Logs

<p align="center">
    <img src="images/training_logs.JPG" alt="centered image" />
</p>

       

# Results

## Accuracy 

  Test Accuracy : 90.95%
  Train Accuracy : 91.81%

