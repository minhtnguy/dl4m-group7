# **Project Overview**

This project uses deep learning to classify outdoor weather conditions from real-world image data. Our goal is to build a model that can accurately recognize and label images as one of seven weather types:

dew, fog/smog, lightning, rain, rainbow, sandstorm, and snow

The system leverages Convolutional Neural Networks (CNNs), regularization techniques, and transfer learning to achieve robust, accurate performance. Potential real-world applications include meteorology, environmental monitoring, and smart city systems.

## **Dataset Description**
Source: https://www.kaggle.com/datasets/jehanbhathena/weather-dataset

The dataset contains thousands of labeled weather condition images.

We used a subset of 7 specific weather types from the full dataset to ensure balance and clarity.

## **Code Structure**
```
dl4m-group7/
│
├── main.ipynb             # Main notebook with all experiments (Parts 1–7)
├── project.py             # Data loading functions
├── utils.py               # Training, evaluation, plots, pretrained models
├── models.py              # Regularized model architecture
├── custom-data/           # Folder for demo/test images
├── data/                  # Weather image dataset
├── environment.yml        # Conda environment setup
├── README.md              # Project description and documentation
```
## **Key Components**
Part 1: Data loading and preparation

Part 2: Baseline CNN training

Part 3: Regularized model with data augmentation

Part 4: Transfer learning using pretrained ResNet50

Part 5: Data analysis

Part 6: Confusion matrix and per-class accuracy

Part 7: Custom image demo

## **Results Summary**

Model	Test Accuracy	Notes

Baseline CNN	~81%	Overfit; good training, weak generalization

Regularized CNN	~83.5%	Improved generalization via dropout + weight decay

Pretrained ResNet (TL)	~89.25%	Best performance; robust and generalizes well

## **Key Findings**
Data augmentation helped reduce overfitting.

Transfer learning provided the best accuracy and generalization.

Confusion matrix analysis confirmed improvements across all weather classes, especially rare ones like rainbow and lightning.

## **Team Contribution**

**Member Role & Contributions**

Rita Tesfay	Data preparation, baseline model, documentation

Minh Nguyen	Pretrained model integration, result analysis

Alison Yang	Augmentation setup, visualization, custom image demo
