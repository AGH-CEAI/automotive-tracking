# automotive-tracking

This code performs object detection and tracking on automotive video data.

## Prerequisites

Recommended environment can be installed on Unix systems via:
  ```bash
  # install Miniconda
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b -u
  rm -rf Miniconda3-latest-Linux-x86_64.sh
  conda list # Verify the installaton

  # create separate environment and install necessary packages:
  conda create -n automotive python=3.11.7 numpy moviepy
  conda activate automotive
  pip install ultralytics opencv-python cap_from_youtube
  ```
<!-- - Julia:
  ```bash
  curl -fsSL https://install.julialang.org | sh
  juliaup add 1.6.7 # version 1.6.7 is recommended, since there are compatibility issues with OpenCV.jl for newer Julia versions
  juliaup default 1.6.7
  julia
  ```
  Then, in the opened Julia REPL:
  ```julia
  ] add OpenCV, CUDA, ObjectDetector, FileIO, ImageIO, VideoIO,
  ``` -->

The video files must be first downloaded before running the code. This can be done using:
- the bash script:
  ```bash
  bash get_files.sh
  ```
- or the python script:
  ```bash
  python get_files.py
  ```


## How to run

The object detection and tracking can be run from the terminal:
```bash
python detect_and_track.py videos/video_filename.mp4 --arg1
```

or by using the jupyter notebooks provided in `examples/`
