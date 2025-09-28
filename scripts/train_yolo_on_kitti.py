"""

"""

from ultralytics import YOLO


def train_on_camera() -> None:
    """

    """
    model: YOLO = YOLO("./models/yolov8n.pt")

    print("Start YOLO training on KITTI camera data.")

    results = model.train(
        # data="/home/piokal/automotive-tracking/datasets/KITTI_for_YOLO/KITTI.yaml",
        data = "./datasets/YOLO_KITTI/yolo_kitti.yaml",
        epochs=10,
        imgsz=1248,
        project="../output/",
        name="yolov8n_trained_on_camera_1_KITTI",
    )

    print("Training done.")


if __name__ == "__main__":
    print("Start")
    train_on_camera()
    print("Done")