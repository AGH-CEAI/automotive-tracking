#!/bin/bash
# --------------------------------------------------------------
# script, which downloads the videos needed for running the code
# --------------------------------------------------------------

# Apollo15:
wget https://ia902809.us.archive.org/28/items/Apollo15And1616mmOnboardFilm/apollo15_10_1.mp4 videos/apollo15_full_movie.mp4
# extracting smaller chunks (if missing, ffmpeg can be installed with `apt install ffmpeg`)
ffmpeg -i apollo15_10_1.mp4 -acodec copy -vcodec copy -ss 00:00:00 -t 00:00:05 OUTFILE.mp4
ffmpeg -i apollo15_10_1.mp4 -acodec copy -vcodec copy -ss 00:00:00 -t 00:00:05 OUTFILE.mp4

wget -O videos/highway_drone_footage.mp4 https://www.shutterstock.com/shutterstock/videos/1095410759/preview/stock-footage-establishing-drone-shot-of-downtown-houston-from-i-and-i-intersection-revealing-aerial-shot.mp4