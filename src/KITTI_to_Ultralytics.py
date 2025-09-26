#!/usr/bin/env python
"""
This is a set of utility functions, which help to convert the KITTI dataset to the Ultralytics format
and prepare the data to be used for training and validation of a object detection model (default: YOLO)
"""

# fixed size of KITTI images:
IMG_WIDTH = 1242
IMG_HEIGHT = 375


def lidar_to_images(
    kitti_dir="../datasets/KITTI/",
    subset="training",
):
    """
    converts the KITTI velodyne lidar point clouds to RBG images and saves them
    """
    import cv2
    import numpy as np
    import sys

    sys.path.append("../utils/")
    import kitti_util as utils
    import lidar
    from glob import glob

    # get the list of scenes:
    scene_ids = [
        fname[-4:] for fname in glob(kitti_dir + subset + "/image_02/*", recursive=True)
    ]
    scene_ids.sort()

    for scene_id in scene_ids:
        print("Processing scene ", scene_id)
        # get the list of frames:
        frame_ids = [
            fname[-10:-4]
            for fname in glob(
                kitti_dir + "training/image_02/" + scene_id + "/*png", recursive=True
            )
        ]
        frame_ids.sort()

        for frame_id in frame_ids:
            # get the lidar points:
            lidar_point_cloud = lidar.get_lidar(
                dir="../datasets/KITTI/training/velodyne/" + scene_id,
                filename=frame_id + ".bin",
                point_cloud_only=True,
            )
            lidar_distance = lidar.get_lidar(
                dir="../datasets/KITTI/training/velodyne/" + scene_id,
                filename=frame_id + ".bin",
                distance_only=True,
            )
            if lidar_point_cloud.size > 0:
                # get the calibration:
                calibration = utils.Calibration(
                    "../datasets/KITTI/training/calib/" + scene_id + ".txt"
                )

                # project the lidar points onto an image:
                img_lidar = lidar.show_lidar_on_image(
                    pc_velo=lidar_point_cloud,
                    img=np.zeros(
                        (IMG_HEIGHT, IMG_WIDTH)
                    ),  # we just use a blank background
                    calib=calibration,
                    img_width=IMG_WIDTH,
                    img_height=IMG_HEIGHT,
                )

                # save the image:
                cv2.imwrite(
                    "../datasets/KITTI_for_YOLO/"
                    + subset
                    + "/lidar/images/scene_"
                    + scene_id
                    + "_frame_"
                    + frame_id
                    + ".jpg",
                    img_lidar,
                )


def kitti_to_ultra_labels(
    kitti_dir="../datasets/KITTI/",
    subset="training",
):
    """
    converts a single KITTI-formatted label file to multiple corresponding files in ultralytics format
    """
    import pandas as pd
    from glob import glob

    if subset == "training":
        print("Converting the KITTI labels ...")
    elif subset == "testing":
        print(
            "The KITTI dataset does NOT have labels provided for the testing/ directory ..."
        )
        raise NotImplementedError
    else:
        print("unknown option")
        raise SyntaxError

    colnames = [
        "frame",
        "track_id",
        "type",
        "truncated",
        "occluded",
        "alpha",
        "bbox_left",
        "bbox_top",
        "bbox_right",
        "bbox_bottom",
        "obj_height",
        "obj_width",
        "obj_length",
        "obj_x",
        "obj_y",
        "obj_z",
        "rotation_y",
        "score",
    ]

    ultra_colnames = ["class", "x_center", "y_center", "width", "height"]

    # get the list of scenes:
    scene_ids = [
        fname[-4:] for fname in glob(kitti_dir + subset + "/image_02/*", recursive=True)
    ]
    scene_ids.sort()

    for scene_id in scene_ids:
        print("Processing scene ", scene_id)
        # read the data:
        labels = pd.read_csv(
            kitti_dir + subset + "/label_02/" + scene_id + ".txt",
            sep=" ",
            header=None,
            names=colnames,
        )[["frame", "type", "bbox_left", "bbox_bottom", "bbox_right", "bbox_top"]]

        for frame in labels["frame"].unique():
            slc = labels["frame"] == frame

            # convert to ultralytics format:
            ultra_df = pd.DataFrame(columns=ultra_colnames)

            ultra_df["class"] = labels[slc]["type"]

            # We need to norm to (0,1) by the image size:
            ultra_df["x_center"] = (
                0.5 * (labels[slc]["bbox_left"] + labels[slc]["bbox_right"]) / IMG_WIDTH
            )
            ultra_df["y_center"] = (
                0.5
                * (labels[slc]["bbox_top"] + labels[slc]["bbox_bottom"])
                / IMG_HEIGHT
            )
            ultra_df["width"] = (
                labels[slc]["bbox_right"] - labels[slc]["bbox_left"]
            ) / IMG_WIDTH
            ultra_df["height"] = (
                labels[slc]["bbox_bottom"] - labels[slc]["bbox_top"]
            ) / IMG_HEIGHT

            # convert class names to integer IDs (matching KITTI.yaml):
            class_name_to_id = {
                "Car": 0,
                "Pedestrian": 1,
                "Van": 2,
                "Cyclist": 3,
                "Truck": 4,
                "Misc": 5,
                "Tram": 6,
                "Person_sitting": 7,
                "Person": 7,
                "DontCare": 8,
            }
            # print(scene_id,frame,ultra_df['class'].unique())
            ultra_df["class"] = [class_name_to_id[c] for c in ultra_df["class"]]

            # prepare the filename to write to:
            output_filename = (
                "scene_" + scene_id + "_frame_" + str(frame).zfill(6) + ".txt"
            )  # list of output files
            # print(output_filename)
            ultra_df.to_csv(
                "../datasets/KITTI_for_YOLO/"
                + subset
                + "/camera_1/labels/"
                + output_filename,
                sep=" ",
                index=None,
                float_format="%.6f",
                header=None,
            )
            ultra_df.to_csv(
                "../datasets/KITTI_for_YOLO/"
                + subset
                + "/camera_2/labels/"
                + output_filename,
                sep=" ",
                index=None,
                float_format="%.6f",
                header=None,
            )
            ultra_df.to_csv(
                "../datasets/KITTI_for_YOLO/"
                + subset
                + "/lidar/labels/"
                + output_filename,
                sep=" ",
                index=None,
                float_format="%.6f",
                header=None,
            )


def split_train_test(
    kitti_dir="../datasets/KITTI/",
    mode="simple",
    start_scene_id=0,
    stop_scene_id=999999,
    only_labels=False,
    only_lidar=False,
):
    """
    Divides the images and/or labels in training/ in the KITTI
    directory into a proper train and test dataset
    """
    import os
    from glob import glob
    from PIL import Image  # needed for .png -> .jpg conversion
    import numpy as np

    if mode == "simple":
        # get the list of scenes:
        scene_ids = [
            fname[-4:]
            for fname in glob(kitti_dir + "training/image_02/*", recursive=True)
        ]
        scene_ids.sort()
        scene_ids = np.asarray(scene_ids)
        scene_ids = scene_ids[
            (scene_ids.astype(int) >= start_scene_id)
            & (scene_ids.astype(int) <= stop_scene_id)
        ]

        for scene_id in scene_ids:
            """
            The split is done:
                train: scenes 0  - 16
                test:  scenes 17 - 20
            """
            print("processing scene ", scene_id)
            if int(scene_id) <= 16:
                subset = "training"
            else:
                subset = "testing"
                print("copying the test labels")
                # move the labels:
                if not only_lidar:
                    os.system(
                        "cp /home/piokal/automotive-tracking/datasets/KITTI_for_YOLO/training/camera_1/labels/scene_"
                        + scene_id
                        + "_frame_*.txt /home/piokal/automotive-tracking/datasets/KITTI_for_YOLO/"
                        + subset
                        + "/camera_1/labels/"
                    )
                    os.system(
                        "cp /home/piokal/automotive-tracking/datasets/KITTI_for_YOLO/training/camera_2/labels/scene_"
                        + scene_id
                        + "_frame_*.txt /home/piokal/automotive-tracking/datasets/KITTI_for_YOLO/"
                        + subset
                        + "/camera_2/labels/"
                    )
                    os.system(
                        "cp /home/piokal/automotive-tracking/datasets/KITTI_for_YOLO/training/lidar/labels/scene_"
                        + scene_id
                        + "_frame_*.txt /home/piokal/automotive-tracking/datasets/KITTI_for_YOLO/"
                        + subset
                        + "/lidar/labels/"
                    )
                if not only_labels:
                    print("copying the test lidar projection images")
                    os.system(
                        "cp /home/piokal/automotive-tracking/datasets/KITTI_for_YOLO/training/lidar/images/scene_"
                        + scene_id
                        + "_frame_*.jpg /home/piokal/automotive-tracking/datasets/KITTI_for_YOLO/"
                        + subset
                        + "/lidar/images/"
                    )

            if not only_labels and not only_lidar:
                # get the list of frames:
                frame_ids = [
                    fname[-10:-4]
                    for fname in glob(
                        kitti_dir + "training/image_02/" + scene_id + "/*png",
                        recursive=True,
                    )
                ]
                frame_ids.sort()

                for frame_id in frame_ids:
                    # camera images are opened and saved as .jpg:
                    img_png = Image.open(
                        kitti_dir
                        + "training/image_02/"
                        + scene_id
                        + "/"
                        + frame_id
                        + ".png"
                    )
                    img_png.save(
                        "../datasets/KITTI_for_YOLO/"
                        + subset
                        + "/images/camera_1/scene_"
                        + scene_id
                        + "_frame_"
                        + frame_id
                        + ".jpg"
                    )

                    img_png = Image.open(
                        kitti_dir
                        + "training/image_03/"
                        + scene_id
                        + "/"
                        + frame_id
                        + ".png"
                    )
                    img_png.save(
                        "../datasets/KITTI_for_YOLO/"
                        + subset
                        + "/images/camera_2/scene_"
                        + scene_id
                        + "_frame_"
                        + frame_id
                        + ".jpg"
                    )
    else:
        print("Other splitting modes are currently not implemented")
        raise NotImplementedError
