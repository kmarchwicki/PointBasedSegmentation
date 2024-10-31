import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from loguru import logger
from skimage import measure
from tqdm import tqdm
from ultralytics import SAM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class WindowHandler:
    def __init__(self, window_name):
        self.window_name = window_name
        self.point = None
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.click_event)

    def click_event(self, event, x, y, _, __):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point = (x, y)
            print(f"Point selected: {self.point}")


def supervision_mask_to_yolo_polygon(mask):
    # Find contours in the mask
    contours = measure.find_contours(mask, 0.5)

    yolo_annotations = []
    for contour in contours:
        # Normalize the coordinates to [0, 1]
        contour = contour[:, [1, 0]]  # Swap columns to get (x, y) format
        contour[:, 0] /= mask.shape[1]
        contour[:, 1] /= mask.shape[0]

        # Flatten the contour points and convert to string
        contour_str = " ".join(map(str, contour.flatten()))
        yolo_annotations.append(contour_str)

    return yolo_annotations


def save_yolo_format(image, detections, frame_number, output_dir):
    image_path = os.path.join(output_dir, "images", f"frame_{frame_number}.jpg")
    label_path = os.path.join(output_dir, "labels", f"frame_{frame_number}.txt")

    # Save the image
    cv2.imwrite(image_path, image)

    # Save labels
    with open(label_path, "w") as f:
        for i in range(len(detections.xyxy)):
            x1, y1, x2, y2 = detections.xyxy[i]
            mask = detections.mask[i]
            cls = detections.class_id[i]

            # Convert to YOLO format
            x_center = (x1 + x2) / 2 / image.shape[1]
            y_center = (y1 + y2) / 2 / image.shape[0]
            width = (x2 - x1) / image.shape[1]
            height = (y2 - y1) / image.shape[0]

            mask_strings = supervision_mask_to_yolo_polygon(mask)
            final_mask_string_custom = " ".join(mask_strings)

            final_annotation = f"{cls} {x_center} {y_center} {width} {height} {final_mask_string_custom}\n"
            f.write(final_annotation)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process video files for segmentation."
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        default="video",
        help="Folder containing video files",
    )
    parser.add_argument(
        "--dataset_folder", type=str, default="yolo", help="Name of the model to use"
    )
    parser.add_argument(
        "--model_name", type=str, default="sam2_t.pt", help="Name of the model to use"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    dataset_folder = args.dataset_folder
    video_folder = args.video_folder
    model_name = args.model_name

    # Load the model
    model = SAM(model_name)
    if DEVICE == "cuda":
        model = model.to(DEVICE)

    # Find the video file
    path = Path(video_folder)
    video_files = [*path.glob("**/*.mp4")]
    video_path = next(iter(video_files), None)

    if video_path is None:
        logger.info("No video files found")
        return

    target_path = video_path.with_name(f"{video_path.stem}_annotated.mp4")
    frame_generator = sv.get_video_frames_generator(source_path=video_path)
    video_info = sv.VideoInfo.from_video_path(video_path=video_path)

    mask_annotator = sv.PolygonAnnotator()
    box_annotator = sv.BoxAnnotator()

    window_handler = WindowHandler("Output")

    # Create YOLOv8 ddataset directory
    os.makedirs(os.path.join(dataset_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, "labels"), exist_ok=True)

    # Main loop
    frame_number = 0
    with sv.VideoSink(target_path=target_path, video_info=video_info) as video_sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            if window_handler.point is not None:
                # Point input
                results = model(frame, points=[window_handler.point], labels=[1])[0]
                masks = results.masks.data.cpu().numpy()
                xyxy = sv.mask_to_xyxy(masks)
                cls = np.array([0] * len(xyxy))
                detections = sv.Detections(xyxy, masks, class_id=cls)

                # Annotations (boxes, masks)
                frame = box_annotator.annotate(scene=frame, detections=detections)
                frame = mask_annotator.annotate(scene=frame, detections=detections)

                # Save in YOLOv8 format
                save_yolo_format(frame, detections, frame_number, dataset_folder)

            # Write the frame to the video stream
            video_sink.write_frame(frame)

            # Display frame and wait for a click event
            cv2.imshow(window_handler.window_name, frame)

            key = cv2.waitKey(1)
            # Break the loop if the 'q' key is pressed
            if key & 0xFF == ord("q"):
                break

            frame_number += 1

    # Ensure the window is closed
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
