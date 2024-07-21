# Weather Image Classification

This repository contains the code and resources for the paper "Real-Time Weather Image Classification with SVM: A Feature-Based Approach". The project uses Support Vector Machine (SVM) to classify weather conditions in images into four categories: rainy, low light, haze, and clear.

## Features
- Brightness
- Saturation
- Noise Level
- Blur Metric
- Edge Strength
- Motion Blur
- Local Binary Patterns (LBP) mean and variance for radii 1, 2, and 3
- Edges mean and variance
- Color histogram mean and variance for blue, green, and red channels

## Results
The SVM model achieved an accuracy of 92.8%, surpassing typical benchmarks for classical machine learning methods.

## Scripts
- `add_artificial_rain.py`: Script to add artificial rain to images.
- `simulate_low_light.py`: Script to simulate low light conditions in images.
- `add_haze.py`: Script to add haze to images.
- `weather_image_classification_svm.py`: Script to extract features from images and classify them using SVM.

## Dataset

The dataset used for this project can be downloaded from the following link:

[Download Dataset](https://drive.google.com/file/d/1HK_mUBxvNd-DolmXbY9kGzLoZvr_8xfI/view?usp=share_link)

## Usage

### Adding Artificial Rain
python add_artificial_rain.py /path/to/source_folder /path/to/destination_folder --rain_intensity 1500

### Simulating Low Light
To simulate low light conditions on images, use the following command:
python simulate_low_light.py /path/to/input_folder /path/to/output_folder

### Adding Haze
To add haze to images, use the following command:
python add_haze.py /path/to/image.jpeg /path/to/output_folder --betas 0.05 0.06 0.07 --A 0.5

### Classifying Weather Images
To classify weather images using the SVM model, use the following command:
python weather_image_classification_svm.py /path/to/dataset


