#!/usr/bin/env python
# --------------------------------------------------------------
# script, which downloads the videos needed for running the code
# --------------------------------------------------------------

import argparse
import os
import sys
from collections import defaultdict
from typing import List, Tuple

import cv2
import dill
import numpy as np
from cap_from_youtube import cap_from_youtube
from pathlib import Path
from shapely.geometry import Polygon

# imports
from ultralytics import YOLO

def get_coords_from_xywh(box) -> List[Tuple[float, float]]:
    """TODO TR"""
    coords: List[Tuple[int, int]] = [
        (box[0], box[1]),
        (box[0] + box[2], box[1]),
        (box[0] + box[2], box[1] + box[3]),
        (box[0], box[1] + box[3]),
    ]
    return coords


# first we handle to arguments passed to the script:
parser = argparse.ArgumentParser(
    prog="python detect_and_track.py",
    description="This code detects objects in a specified video, identifies them and tracks their trajectories",
    epilog='To manually stop the processing, press "q"',
)
parser.add_argument(
    "video_filename",
    type=str,
    nargs=1,
    action="store",
    help="Video file to be processed. It can be either a local path or a YouTube URL.",
)
parser.add_argument(
    "--max-frames",
    dest="max_frames",
    default=0,
    type=int,
    help="maximal number of frames to be processed",
)
parser.add_argument(
    "--every-nth",
    dest="every_nth",
    default=15,
    type=int,
    help="saves every nth frame to a .jpg",
)

args = parser.parse_args()
print(args)
model_name: str 
# model_name = "yolov8n"
model_name = "yolov8n_kitti_camera"
# model_name = "yolov8n_kitti_lidar"

# Load the model
print("loading the YOLO model ...")
model: YOLO = YOLO(f"models/{model_name}.pt")

# Initialize empty track history
track_history = defaultdict(lambda: [])

if args.video_filename[0].endswith(".jpg") or args.video_filename[0].endswith(".png"):
    # TODO TR:  This is terribly redundant and should be refactored at some point.
    print(f"Analyzing image: {args.video_filename[0]}")

    frame = cv2.imread(args.video_filename[0])

    results = model.track(frame)

    # Get the boxes and track IDs
    boxes = results[0].boxes.xywh.cpu()
    polygons: List[Polygon] = []

    for box in boxes:
        polygons.append(Polygon(get_coords_from_xywh(box)))

    track_ids = results[0].boxes.id.int().cpu().tolist()

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Plot the tracks
    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))  # x, y center point
        if len(track) > 30:  # retain 90 tracks for 90 frames
            track.pop(0)

        # Draw the tracking lines
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            annotated_frame, [points], isClosed=False, color=(255, 0, 0), thickness=10
        )

    # Display the annotated frame (requires X-forwarding ...)
    # cv2.imshow("YOLOv8 Tracking", annotated_frame)

    print(f"saving annoted frame and boxes from image...")
    f_name: str = args.video_filename[0].split("/")[-1].replace(".jpg", "")
    
    output_root: str = f"output/{model_name}/"
    Path(output_root).mkdir(parents=True, exist_ok=True)

    cv2.imwrite(f"{output_root}/YOLO_{f_name}.jpg", annotated_frame)  # save frame as JPEG file

    with open(f"{output_root}/YOLO_{f_name}_boxes.dil", "wb") as dill_file:
        dill.dump(polygons, dill_file)

    sys.exit(0)

# Open the video file
if "youtu" in args.video_filename[0]:
    print("Opening a video from YouTube ... ")
    fname = "from_yt"
    cap = cap_from_youtube(args.video_filename[0], "720p")
else:
    print("Opening a local video file ... ")
    path = "".join(args.video_filename)
    fname = os.path.split(path)[-1]
    cap = cv2.VideoCapture(path)


# Define the output file
output_video = cv2.VideoWriter(
    "output/processed_" + fname,
    cv2.VideoWriter_fourcc(*"mp4v"),  # writer object
    int(cap.get(cv2.CAP_PROP_FPS)),  # FPS
    (852, 480),
)  # frame size

# Loop through the video frames
i = 0
while cap.isOpened():
    i += 1
    # Read a frame from the video
    success, frame = cap.read()

    if not success:
        print("failed to read the frame :<")
        # Break the loop if the end of the video is reached
        break

    # print(frame.shape)
    frame = cv2.resize(frame, dsize=(852, 480))  # reshape to 480, 852
    # print(frame.shape)

    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(
        frame, persist=True, tracker="bytetrack.yaml"
    )  # , show=True, stream=True

    # Get the boxes and track IDs
    boxes = results[0].boxes.xywh.cpu()
    polygons: List[Polygon] = []

    for box in boxes:
        polygons.append(Polygon(get_coords_from_xywh(box)))

    track_ids = results[0].boxes.id.int().cpu().tolist()

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Plot the tracks
    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))  # x, y center point
        if len(track) > 30:  # retain 90 tracks for 90 frames
            track.pop(0)

        # Draw the tracking lines
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            annotated_frame, [points], isClosed=False, color=(255, 0, 0), thickness=10
        )

    # Display the annotated frame (requires X-forwarding ...)
    # cv2.imshow("YOLOv8 Tracking", annotated_frame)

    # write the frame to the output file
    output_video.write(annotated_frame)

    # save every n-th frame as jpg
    if i % args.every_nth == 0:
        print(f"saving {args.every_nth}-th frame ...")
        cv2.imwrite(
            "output/frame_%d.jpg" % i, annotated_frame
        )  # save frame as JPEG file

        with open(f"output/boxes_{i}.dil", "wb") as dill_file:
            dill.dump(polygons, dill_file)

    if (i == args.max_frames) & (args.max_frames > 0):
        print("ok, that's enough ...")
        break


# Release the video capture object and close the display window
cap.release()
output_video.release()
cv2.destroyAllWindows()

print("Finished processing the file!")
