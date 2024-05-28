# automotive-tracking

This code performs object detection and tracking on automotive video data.

## Prerequisites

To be able to perform object detection and tracking, one must first prepare the environment and the data.

### Setting up the environment

Recommended environment can be installed on Unix systems via:
  ```bash
  # install Miniconda
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b -u
  rm -rf Miniconda3-latest-Linux-x86_64.sh
  conda list # Verify the installaton

  # create separate environment and install necessary packages:
  conda create -n automotive python=3.11.7 numpy
  conda activate automotive
  pip install ultralytics opencv-python cap_from_youtube moviepy jupyter
  ```

### Obtaining the data

If just playing with YOLO model, it is enough to find a YouTube url or a local video/image file of interest and use it with `/src/detect_and_track_YOLO.py` or `examples/detect_and_track_YOLO.ipynb`.

To get the large automotive datasets with object annotations etc., one can use the download scripts, provided in their respective subdirectories in `datasets/`.


## How to run

### Simple YOLO

The object detection and tracking with YOLO can be run from the terminal:
```bash
python src/detect_and_track_YOLO.py videos/video_filename.mp4 --arg1
```

or by using the notebook: `examples/detect_and_track_YOLO.ipynb`

### QUBO

This has not been implemented yet.

