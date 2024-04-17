#!/bin/bash
mkdir data/videos_processed
for file in data/videos_raw/*.avi;
do
  base=$(basename "$file" .avi)
  mkdir -p "data/videos_processed/$base"
  ffmpeg -i "$file" -c copy -map 0 -segment_time 2 -f segment -reset_timestamps 1 "data/videos_processed/$base/%03d.mp4"
done