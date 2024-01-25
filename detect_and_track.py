#!/usr/bin/env python
# --------------------------------------------------------------
# script, which downloads the videos needed for running the code
# --------------------------------------------------------------

# imports
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict

# Load an official or custom model
model = YOLO('../models/yolov8n.pt')  # Load an official Detect model
# model = YOLO('yolov8n-seg.pt')  # Load an official Segment model
# model = YOLO('yolov8n-pose.pt')  # Load an official Pose model
# model = YOLO('path/to/best.pt')  # Load a custom trained model


# Open the video file
# fname="highway_drone_footage"
# fname="apollo15_10_1_1st"
fname="cars"
# cap = cv2.VideoCapture("../videos/cars_n_palms.mp4")
cap = cv2.VideoCapture("../videos/"+fname+".mp4")
# cap = cv2.VideoCapture("https://youtu.be/40xZVEFVBuE?si=sYF-8V0nWvB4ztto")

# Store the track history
track_history = defaultdict(lambda: [])


# Below VideoWriter object will create a frame of above defined The output  
# is stored in 'output.mp4' file. 
output_video = cv2.VideoWriter("../output/"+fname+"_output.mp4",  
                        #  cv2.VideoWriter_fourcc(*'XVID'), # writer object
                         cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                         int(cap.get(cv2.CAP_PROP_FPS)), # FPS
                         (852,480)) # frame size

# Loop through the video frames
i=0
while cap.isOpened():
    i+=1
    # Read a frame from the video
    success, frame = cap.read()

    if success:

        # print(frame.shape)
        frame = cv2.resize(frame, dsize=(852,480))# reshape to 480, 852
        # print(frame.shape)

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True,tracker="bytetrack.yaml") # , show=True, stream=True

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
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
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(255,0,0), thickness=10)

        # Display the annotated frame (requires X-forwarding ...)
        # cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # write the frame to the output file
        output_video.write(annotated_frame)

        # save every n-th frame as jpg
        if i % 100 == 0:
            print('saving 15-th frame ...')
            cv2.imwrite("../output/frame%d.jpg" % i, annotated_frame)     # save frame as JPEG file 

        # if i==100:
        #     print("ok, that's enough ...")
        #     break

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print("failed to read the frame :<")
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
output_video.release()
cv2.destroyAllWindows()