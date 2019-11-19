# Deep-Neural-Network-Image-Classification

This project aims to build a smart system which could classify new images into a set of categories.
All the 3 types of systems are built on the concept of deep neural networks to train and optimize themselves.

## Model types
There are 3 types of modules built for classifying images:
1. Building Neural network from scratch
2. Using pre-trained neural network
3. Using Transfer learning to build a neural network

## Dataset
We are using cifar10 dataset for training the systems.
Total images: 60,000
Size: 32 x 32
Total Classes: 10


## Model descriptions
### 1. From Scratch
All the layers for this system are build in combination of different types.

Layers:
1. Dense
2. Convoluted

A total of 12 layers are being used in the model with Dropout and Flatten.
Dropout rate: 50%
Activation function - Softmax

### 2. Using Pre-trained model
Model used - VGG16

### 3. Transfer Learning
Extracted features from a trained model on images.
Trained it and used for predictions.
