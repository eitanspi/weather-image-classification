import cv2
import numpy as np
import random
import os
import argparse


def add_rain_to_images(source_folder, destination_folder, rain_intensity=1500):
    """
    Adds artificial rain to each image in the source folder and saves them in the destination folder.

    Parameters:
        source_folder (str): Path to the folder containing the input images.
        destination_folder (str): Path to the folder to save the output images.
        rain_intensity (int): Number of raindrops to add. Default is 1500.
    """
    # Ensure destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get list of images in source folder
    images = [f for f in os.listdir(source_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))]

    for image_name in images:
        # Full path to the image
        image_path = os.path.join(source_folder, image_name)

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to open image at {image_path}")
            continue

        # Create a copy of the image to add rain
        rainy_image = image.copy()
        height, width, _ = image.shape

        # Generate random raindrops
        for _ in range(rain_intensity):
            x1 = random.randint(0, width - 1)
            y1 = random.randint(0, height - 1)
            length = random.randint(10, 20)
            angle = random.uniform(-np.pi / 18, np.pi / 18)  # Small random tilt

            x2 = int(x1 + length * np.sin(angle))
            y2 = int(y1 + length * np.cos(angle))

            # Draw the raindrop as a line
            cv2.line(rainy_image, (x1, y1), (x2, y2), (200, 200, 200), 1)

        # Apply a motion blur to simulate the rain effect
        kernel_size = 3
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
        kernel = kernel / kernel_size

        rainy_image = cv2.filter2D(rainy_image, -1, kernel)

        # Save the output image
        output_image_path = os.path.join(destination_folder, image_name)
        cv2.imwrite(output_image_path, rainy_image)
        print(f"Rainy image saved at {output_image_path}")


def main():
    parser = argparse.ArgumentParser(description="Add artificial rain to images.")
    parser.add_argument('source_folder', type=str, help='Path to the folder containing the input images.')
    parser.add_argument('destination_folder', type=str, help='Path to the folder to save the output images.')
    parser.add_argument('--rain_intensity', type=int, default=1500, help='Number of raindrops to add. Default is 1500.')

    args = parser.parse_args()

    add_rain_to_images(args.source_folder, args.destination_folder, args.rain_intensity)


if __name__ == "__main__":
    main()
