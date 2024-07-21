import cv2
import numpy as np
import random
import os
import argparse


def simulate_low_light(image_path, output_path):
    """
    Simulates low light conditions on the given image and saves the result.

    Parameters:
        image_path (str): Path to the input image.
        output_path (str): Path to save the output image.
    """
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return

    # Normalize the image to range [0, 1]
    img_f = image / 255.0

    # Generate a random low light parameter
    lowlight_param = np.around(random.uniform(1.5, 5), 2)
    print(f"Lowlight parameter for {image_path}: {lowlight_param}")

    # Apply the low light transformation
    img_f = np.power(img_f, lowlight_param)

    # Convert back to range [0, 255] and uint8 type
    img_lowlight = (img_f * 255).astype(np.uint8)

    # Write the output image
    cv2.imwrite(output_path, img_lowlight)
    print(f"Low light image saved at {output_path}")


def process_folder(input_folder, output_folder):
    """
    Processes all images in the input folder to simulate low light conditions and saves them in the output folder.

    Parameters:
        input_folder (str): Path to the folder containing the input images.
        output_folder (str): Path to the folder to save the output images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            simulate_low_light(input_path, output_path)
        else:
            print(f"Skipped non-image file: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Simulate low light conditions on images.")
    parser.add_argument('input_folder', type=str, help='Path to the folder containing the input images.')
    parser.add_argument('output_folder', type=str, help='Path to the folder to save the output images.')

    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
