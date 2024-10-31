import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def visualize_yolo_annotation(image_path, annotation_path):
    # Load the image
    image = Image.open(image_path)
    w, h = image.size
    draw = ImageDraw.Draw(image)

    # Read the annotation file
    with open(annotation_path, "r") as file:
        for line in file:
            parts = line.strip().split()

            # class label assigned is always equal to 0
            cls = int(parts[0])
            x_center = float(parts[1]) * w
            y_center = float(parts[2]) * h
            width = float(parts[3]) * w
            height = float(parts[4]) * h
            mask_points = list(map(float, parts[5:]))

            # Draw bounding box
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

            # Draw segmentation mask
            mask_points = [
                (mask_points[i] * w, mask_points[i + 1] * h)
                for i in range(0, len(mask_points), 2)
            ]
            draw.polygon(mask_points, outline="blue")

    # Convert to numpy array for visualization
    image_np = np.array(image)

    # Display the image with annotations
    plt.figure(figsize=(8, 8))
    plt.title(f"Class label: {cls}")
    plt.imshow(image_np)
    plt.axis("off")
    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Visualize annotated files with segmentation."
    )
    parser.add_argument(
        "--dataset_folder", type=str, default="yolo", help="Name of the model to use"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    dataset_folder = args.dataset_folder

    path = Path(dataset_folder)
    image_files = sorted([*path.glob("**/*.jpg")])
    annotation_files = sorted([*path.glob("**/*.txt")])
    image_path = next(iter(image_files), None)
    annotation_path = next(iter(annotation_files), None)

    visualize_yolo_annotation(image_path, annotation_path)


if __name__ == "__main__":
    main()
