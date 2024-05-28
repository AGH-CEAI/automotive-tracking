#!/usr/bin/env python
# --------------------------------------------------------------
# script, which downloads the videos needed for running the code
# --------------------------------------------------------------

#imports
from urllib.request import urlretrieve

# download:
full_movie="videos/apollo15_full_movie.mp4"
urlretrieve(
    url="https://ia902809.us.archive.org/28/items/Apollo15And1616mmOnboardFilm/apollo15_10_1.mp4", 
    filename=full_movie,
)

# split:
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# Replace the filename below.
required_video_file = "filename.mp4"

with open("times.txt") as f:
  times = f.readlines()

times = [x.strip() for x in times] 

for time in times:
  starttime = int(time.split("-")[0])
  endtime = int(time.split("-")[1])
  ffmpeg_extract_subclip(full_movie, starttime="00:00:00", endtime="00:00:05", targetname="videos/apollo15_1st.mp4")