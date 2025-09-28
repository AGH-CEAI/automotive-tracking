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
    - [ ] Labels preparation
    - [ ] Yaml preparation
2) Lidar data preparation:
    - [X] Folder structure preparation
    - [ ] Images preparation 
    - [ ] Labels copying
    - [ ] Yaml preparation

How to use:
    Run the script from the parent folder of the project.
"""
import os
from abc import ABC, abstractmethod
from pathlib import Path
from enum import StrEnum



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
