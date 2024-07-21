import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from scipy.ndimage import laplace, sobel
import joblib
from datetime import datetime
import argparse

# Define the folders and corresponding labels
folders = {
    'real_haze': 0,
    'low_light': 1,
    'rain_augmented': 2,
    'clear': 3
}

# Function to calculate brightness
def calculate_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])
    return brightness

# Function to calculate saturation
def calculate_saturation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:, :, 1])
    return saturation

# Function to calculate noise level
def calculate_noise_level(image_gray):
    noise = laplace(image_gray)
    noise_level = np.var(noise)
    return noise_level

# Function to calculate blur metric
def calculate_blur_metric(image_gray):
    blur_metric = cv2.Laplacian(image_gray, cv2.CV_64F).var()
    return blur_metric

# Function to calculate edge strength X
def calculate_edge_strength_x(image_gray):
    sobel_x = sobel(image_gray, axis=0)
    edge_strength_x = np.mean(np.abs(sobel_x))
    return edge_strength_x

# Function to calculate motion blur X
def calculate_motion_blur_x(image_gray):
    motion_blur_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=5).var()
    return motion_blur_x

# Function to extract concise features from an image
def extract_features(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate additional features
    brightness = calculate_brightness(image)
    saturation = calculate_saturation(image)
    noise_level = calculate_noise_level(image_gray)
    blur_metric = calculate_blur_metric(image_gray)
    edge_strength_x = calculate_edge_strength_x(image_gray)
    motion_blur_x = calculate_motion_blur_x(image_gray)

    # LBP features with different radii
    lbp_features = []
    for radius in [1, 2, 3]:
        n_points = 8 * radius
        lbp = local_binary_pattern(image_gray, n_points, radius, method='uniform')
        lbp_mean = np.mean(lbp)
        lbp_var = np.var(lbp)
        lbp_features.extend([lbp_mean, lbp_var])

    # Edge feature: Canny Edge Detection
    edges = cv2.Canny(image_gray, 100, 200)
    edges_mean = np.mean(edges)
    edges_var = np.var(edges)

    # Color Histogram (using mean and variance)
    hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])

    color_features = []
    for hist in [hist_b, hist_g, hist_r]:
        color_mean = np.mean(hist)
        color_var = np.var(hist)
        color_features.extend([color_mean, color_var])

    # Concatenate all features into a single feature vector
    features = np.hstack([
        brightness, saturation, noise_level, blur_metric, edge_strength_x, motion_blur_x,
        *lbp_features, edges_mean, edges_var, *color_features[:10]
    ])

    return features

def main():
    parser = argparse.ArgumentParser(description="Weather image classification using SVM.")
    parser.add_argument('base_path', type=str, help='Base path to the dataset.')
    args = parser.parse_args()

    base_path = args.base_path

    # Prepare the dataset
    features = []
    labels = []

    min_images = float('inf')

    # Determine the minimum number of images in any folder
    for folder in folders.keys():
        folder_path = os.path.join(base_path, folder)
        num_images = len([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
        if num_images < min_images:
            min_images = num_images

    print(f"Minimum number of images per folder: {min_images}")

    for folder, label in folders.items():
        folder_path = os.path.join(base_path, folder)
        images_processed = 0
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.png', '.jpeg')) and images_processed < min_images:
                image_path = os.path.join(folder_path, filename)
                try:
                    feature_vector = extract_features(image_path)
                    features.append(feature_vector)
                    labels.append(label)
                    images_processed += 1
                except FileNotFoundError:
                    print(f"File not found: {image_path}. Skipping.")
                    continue

    # Convert lists to numpy arrays
    X = np.array(features)
    y = np.array(labels)

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Generate timestamp
    timestamp = datetime.now().strftime("%d%m%H%M")

    # Save the scaler
    scaler_filename = f'scaler_{timestamp}.joblib'
    joblib.dump(scaler, scaler_filename)
    print(f"Scaler has been saved to '{scaler_filename}'")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize and train the SVM classifier
    clf = SVC(kernel='linear', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=folders.keys())

    # Print the stats of the SVM
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

    # Save the trained SVM model
    svm_model_filename = f'svm_model_{timestamp}.joblib'
    joblib.dump(clf, svm_model_filename)
    print(f"SVM model has been saved to '{svm_model_filename}'")

    # Calculate per-class accuracy
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    class_accuracy_df = pd.DataFrame({
        'Class': folders.keys(),
        'Accuracy': class_accuracy
    })
    print("Per-Class Accuracy:")
    print(class_accuracy_df)

    # Feature importance using permutation importance
    perm_importance = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42)
    sorted_idx = perm_importance.importances_mean.argsort()

    feature_names = [
        'Brightness', 'Saturation', 'Noise Level', 'Blur Metric', 'Edge Strength X', 'Motion Blur X',
        'LBP Mean R1', 'LBP Var R1', 'LBP Mean R2', 'LBP Var R2', 'LBP Mean R3', 'LBP Var R3',
        'Edges Mean', 'Edges Var', 'Color Mean B', 'Color Var B', 'Color Mean G', 'Color Var G', 'Color Mean R',
        'Color Var R'
    ]

    # Prepare data for Excel export
    svm_stats = {
        'Metric': ['Accuracy'],
        'Value': [accuracy]
    }
    for line in report.split('\n'):
        if 'precision' in line:
            continue
        parts = line.split()
        if len(parts) > 0 and parts[0] in folders.keys():
            svm_stats['Metric'].append(f'Precision {parts[0]}')
            svm_stats['Value'].append(parts[1])
            svm_stats['Metric'].append(f'Recall {parts[0]}')
            svm_stats['Value'].append(parts[2])
            svm_stats['Metric'].append(f'F1-Score {parts[0]}')
            svm_stats['Value'].append(parts[3])
    svm_stats_df = pd.DataFrame(svm_stats)

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean[sorted_idx]
    }).sort_values(by='Importance', ascending=False)

    # Save to Excel
    excel_filename = f'svm_analysis_{timestamp}.xlsx'
    with pd.ExcelWriter(excel_filename) as writer:
        svm_stats_df.to_excel(writer, sheet_name='SVM Stats', index=False)
        feature_importance_df.to_excel(writer, sheet_name='Feature Importance', index=False)
        class_accuracy_df.to_excel(writer, sheet_name='Class Accuracy', index=False)

    print(f"SVM analysis has been exported to '{excel_filename}'")


if __name__ == "__main__":
    main()
