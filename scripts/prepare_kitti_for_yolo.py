"""
The idea of this script is to prepare raw KITTI dataset data, i.e.:
- images,
- labels,
for Ultralytics YOLO training. 

We will understand raw data as the one downloaded via either
`get_files_KITTI.sh` or `get_files_KITTI_curl.sh` script
found in `./datasets/KITTI/`. We will then unpack all the
zips inside this KITTI folder (or any other provided path).
Such data files will be the input for YOLO-ready datasets.

Our target folder structure is as proposed below.
https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format

There will be two steps:
1) Original data preparation:
    - [X] Folder structure preparation
    - [X] Move images
    - [X] Labels preparation
    - [X] Yaml preparation
2) Lidar data preparation:
    - [X] Folder structure preparation
    - [ ] Images preparation 
    - [ ] Labels copying
    - [ ] Yaml preparation

How to use:
    Run the script from the parent folder of the project.
"""
import os
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from enum import StrEnum
from glob import glob
from typing import List, Dict


class DataPurpose(StrEnum):
    """ 
    Purpose of the data in YOLO.

    :note:
        Used e.g. during KITTI images moving.
    """
    TRAIN: str = "train"
    VAL: str = "val"
    TEST: str = "test"


class YOLODatasetPreparator(ABC):
    """ An boilerplate class for YOLO-ready datasets preparation. """

    def __init__(self, root: str, include_test: bool = False) -> None:
        """
        YOLODatasetPreparator constructor. Gathers the usually
        needed 
        
        :args:
            root (str):
                Parent folder of the dataset.
            include_test (bool):
                Flag indicating whether to create test folders.
        """
        self._root: str = root
        self._include_test: bool = include_test
        

    @staticmethod
    def prepare_yolo_dataset_folder_structure(root: str, include_test: bool = False) -> None:
        """
        Prepares YOLO dataset structure according to:
        https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format.

        :args:
            root (str):
                Parent folder of the dataset.
            include_test (bool):
                Flag indicating whether to create test folders.
            
            :note:
                Not used for KITTI, but makes the method reusable
                in the broader context.
        """
        print("Prepare folder structure.")
        Path(root).mkdir(parents=True, exist_ok=True)
        Path(root+"/images/train/").mkdir(parents=True, exist_ok=True)
        Path(root+"/images/val/").mkdir(parents=True, exist_ok=True)
        
        Path(root+"/labels/train/").mkdir(parents=True, exist_ok=True)
        Path(root+"/labels/val/").mkdir(parents=True, exist_ok=True)
        
        if include_test:
            Path(root+"/images/test/").mkdir(parents=True, exist_ok=True)
            Path(root+"/labels/test/").mkdir(parents=True, exist_ok=True)
        print(f"Dataset folder structure created in: {root}")

    @abstractmethod
    def prepare_yolo_dataset(self) -> None:
        """ Abstract method for dataset preparation. """
        raise NotImplementedError


class YOLOKITTIDatasetPreparator(YOLODatasetPreparator):
    """ A class for KITTI dataset preparation for YOLO. """

    def __init__(self, root: str, include_test: bool = False) -> None:
        """ TODO """
        super().__init__(root, include_test)

        # For labels creation.
        self._KITTI_labels_col_names: List[str] = [
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

        self._ultra_colnames: List[str] = [
            "class", "x_center", "y_center", "width", "height"
        ]

        # Convert class names to integer IDs (matching KITTI.yaml):
        self._class_name_to_id: Dict[str, int] = {
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

        # Fixed size of KITTI images. For labels generation.
        self.IMG_WIDTH: int = 1242
        self.IMG_HEIGHT: int = 375

    def _move_images(self, purpose: DataPurpose, data_subfolder: str) -> None:
        """
        Move images from the raw KITTI folders, to preprepared folders
        of the YOLO-ready version of the dataset.

        :note:
            We assume KITTI zips were unpacked to ./datasets/KITTI.

        :args:
            purpose (DataPurpose):
                ...
            data_subfolder (str):
                ...
        """
        images_root: str = f"./datasets/KITTI/{data_subfolder}/"
        for subdir, dirs, files in os.walk(images_root):
            for file in files:
                source_path: str = subdir + os.sep + file
                
                scene: str = subdir.split("/")[-1]
                target_path: str = f"{self._root}/images/{purpose}/{scene}_{file}"
                os.replace(source_path, target_path) 

    def _process_label_file(self, file_path: str) -> None:
        """
        Creates YOLO-ready labels from the raw KITTI label file.

        :note:
            The main point of this function is to reduce nesting in the
            `_prepare_labels` 

        :args:
            file_path (str):
                Path to the label file.
        """
        print(f"\tProcessing: {file_path}")
        scene_id: str = file_path.split("/")[-1].split(".")[0]

        labels: pd.DataFrame = pd.read_csv(
            file_path,
            sep=" ",
            header=None,
            names=self._KITTI_labels_col_names,
        )[["frame", "type", "bbox_left", "bbox_bottom", "bbox_right", "bbox_top"]]

        for frame in labels["frame"].unique():
            slc = labels["frame"] == frame

            # convert to ultralytics format:
            ultra_df = pd.DataFrame(columns=self._ultra_colnames)
            ultra_df["class"] = labels[slc]["type"]

            # We need to norm to (0,1) by the image size:
            ultra_df["x_center"] = (
                0.5 * (labels[slc]["bbox_left"] + labels[slc]["bbox_right"]) / self.IMG_WIDTH
            )
            ultra_df["y_center"] = (
                0.5
                * (labels[slc]["bbox_top"] + labels[slc]["bbox_bottom"])
                / self.IMG_HEIGHT
            )
            
            ultra_df["width"] = (
                labels[slc]["bbox_right"] - labels[slc]["bbox_left"]
            ) / self.IMG_WIDTH
            
            ultra_df["height"] = (
                labels[slc]["bbox_bottom"] - labels[slc]["bbox_top"]
            ) / self.IMG_HEIGHT

            ultra_df["class"] = [self._class_name_to_id[c] for c in ultra_df["class"]]
            
            output_filename: str = f"{scene_id}_{str(frame).zfill(6)}.txt"                
            out_file_path: str = f"{self._root}labels/train/{output_filename}"

            ultra_df.to_csv(
                out_file_path,
                sep=" ",
                index=None,
                float_format="%.6f",
                header=None,
            )

        print(f"\t{file_path} processed")

    def _prepare_labels(self) -> None:
        """
        Converts a single KITTI-formatted label file to multiple corresponding
        files in Ultralytics format.

        :note:
            KITTI only provides the labels for training data.
        """
        print("Preparing YOLO labels.")
        images_root: str = f"./datasets/KITTI/training/label_02/"
        for subdir, dirs, files in os.walk(images_root):
            for file in files:
                source_path: str = subdir + os.sep + file
                self._process_label_file(source_path)
        print("YOLO labels ready.")

    def _create_yaml(self) -> None:
        """
        Creates YOLO-compatible YAML for YOLO-ready KITTI data.
        """
        print("Creating YOLO training YAML for KITTI dataset.")

        yaml_content: str = "# This file was autogenerated and assumes folder structure similar to:\n"
        yaml_content += "# https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format\n\n"
        
        yaml_content += f"path: {self._root}\n"
        yaml_content += f"train: images/train\n"
        yaml_content += f"val: images/val\n"
        yaml_content += f"test: # KITTI has no test images\n\n"
        
        yaml_content += f"names:\n"

        for k in self._class_name_to_id.keys():
            yaml_content += f'  {self._class_name_to_id[k]}: {k}\n'

        with open(self._root + "yolo_kitti.yaml", "w") as f:
            f.write(yaml_content)

        print("YAML created.")

    def prepare_yolo_dataset(self) -> None:
        """ Creates YOLO-ready version of KITTI dataset. """
        self.prepare_yolo_dataset_folder_structure(self._root, self._include_test)
        
        # Move data
        self._move_images(
            DataPurpose.TRAIN,
            "data_tracking_image_2/training/image_02"
        )

        self._move_images(
            DataPurpose.VAL,
            "data_tracking_image_2/testing/image_02"
        )

        self._prepare_labels()
        self._create_yaml()


def prepare_KITTI_YOLO() -> None:
    """
    Prepare RAW KITTI dataset for YOLO model training.
    """
    print("Preparing KITTI YOLO.")
    preparator: YOLODatasetPreparator = YOLOKITTIDatasetPreparator(
        "./datasets/YOLO_KITTI/"
    )
    preparator.prepare_yolo_dataset()





if __name__ == "__main__":
    print("Start")
    prepare_KITTI_YOLO()
    print("End")
