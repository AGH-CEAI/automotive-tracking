"""
Scripts for running YOLO training on preprepared KITTI data.
"""

from ultralytics import YOLO


def train_yolo_on_kitti(yaml_path: str, model_name: str, n_epochs:int = 10) -> None:
    """
    YOLO training on preprepared KITTI data.

    :note:
        See ./scripts/prepare_kitti_for_yolo.py for info about data preparation.
    """
    n_imgsz: int = 1248 # TODO TR: Seems to be the image width.

    print(f"Start {model_name} training using {yaml_path}.")

    model: YOLO = YOLO("./models/yolov8n.pt")
    results = model.train(
        data = yaml_path,
        epochs=n_epochs,
        imgsz=n_imgsz,
        project="../output/",
        name=model_name,
    )

    print("Training done.")



if __name__ == "__main__":
    print("Start")

    # Train YOLO on KITTI camera 2.
    train_yolo_on_kitti(
        yaml_path="./datasets/YOLO_KITTI/yolo_kitti.yaml",
        model_name="yolov8n_trained_on_camera_2_KITTI"
    )

    # Train YOLO on KITTI lidar.
    train_yolo_on_kitti(
        yaml_path="./datasets/YOLO_KITTI/yolo_kitti_lidar.yaml",
        model_name="yolov8n_trained_on_lidar_KITTI"
    )

    print("Done")