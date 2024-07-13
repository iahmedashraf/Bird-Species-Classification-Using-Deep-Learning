# Bird-Species-Classification-Using-Deep-Learning

This project aims to classify bird species using a deep learning approach. The dataset consists of images from 20 different bird species. The model is built using TensorFlow and Keras, leveraging a pre-trained InceptionV3 model for feature extraction and fine-tuning.

## Project Overview

- **Dataset**: The dataset includes images from 20 bird species, divided into training, validation, and test sets.
- **Pre-trained Model**: InceptionV3 is used as the base model for feature extraction.
- **Image Augmentation**: Applied to the training dataset using the ImageDataGenerator class.
- **YOLO Integration**: Utilized for object detection to enhance model predictions.

## Directory Structure

- `train`: Directory containing training images.
- `test`: Directory containing test images.
- `valid`: Directory containing validation images.

## Model Architecture

1. **InceptionV3**: Pre-trained on ImageNet, used for feature extraction.
2. **Dense Layers**: Added on top of the base model for classification.
3. **Dropout**: Applied to prevent overfitting.
4. **Activation**: ReLU for hidden layers and softmax for the output layer.

## Training the Model

The model is trained using RMSprop optimizer and categorical cross-entropy loss function. Early stopping is used to monitor validation accuracy and prevent overfitting.

## Visualization

- **Class Distribution**: Bar chart showing the number of images per bird species.
- **Training History**: Plots for loss and accuracy for both training and validation sets.
## Object Detection with YOLO

Integrated YOLO for bird detection and localization, improving the overall prediction pipeline.

## How to Use

1. **Training**: Utilize ImageDataGenerator for training and validation datasets.
2. **Prediction**: Load and preprocess images before making predictions with the model.
3. **Object Detection**: Apply YOLO for bird detection in images.

## Results

- Model achieved high accuracy on the validation set.
- YOLO model effectively detects and localizes birds in images.

## Requirements

- TensorFlow
- Keras
- OpenCV
- Matplotlib
- Seaborn
- Pandas
- NumPy

## Conclusion

This project demonstrates the effectiveness of combining pre-trained models and object detection techniques for bird species classification. The approach can be extended to other classification tasks with similar datasets.

