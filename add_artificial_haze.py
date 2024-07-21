import cv2
import numpy as np
import math
import os
import argparse


def add_haze(image_path, output_path, beta, A):
    """
    Adds haze to an image and saves the output.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path where the output image will be saved.
        beta (float): Haze density coefficient.
        A (float): Atmospheric light intensity.

    Returns:
        None
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"No image found at {image_path}")

    img_f = image / 255.0  # Normalize image to 0-1 range

    # Process the image with the haze effect
    img_f = apply_haze(img_f, beta, A)

    # Convert back to 0-255 and uint8 type
    img_f = np.clip(img_f * 255, 0, 255).astype(np.uint8)

    # Save the foggy image
    cv2.imwrite(output_path, img_f)


def apply_haze(img_f, beta, A):
    row, col, chs = img_f.shape
    size = math.sqrt(max(row, col))
    center = (row // 2, col // 2)

    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img_f[j, l, :] = img_f[j, l, :] * td + A * (1 - td)
    return img_f


def main():
    parser = argparse.ArgumentParser(description="Add haze to an image.")
    parser.add_argument('image_path', type=str, help='Path to the input image.')
    parser.add_argument('output_folder', type=str, help='Path to the folder to save the output images.')
    parser.add_argument('--betas', type=float, nargs='+', default=[0.01 * i + 0.05 for i in range(10)],
                        help='List of beta values for haze density coefficient.')
    parser.add_argument('--A', type=float, default=0.5, help='Atmospheric light intensity. Default is 0.5.')

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    for beta in args.betas:
        output_path = os.path.join(args.output_folder, f'hazy_beta_{beta:.2f}.jpeg')
        add_haze(args.image_path, output_path, beta, args.A)


if __name__ == "__main__":
    main()
