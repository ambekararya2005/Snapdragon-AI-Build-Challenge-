#!/usr/bin/env bash
# Poster -> brag.jpg, bake it as frame 0, mux the normalized soundtrack -> ../brag.mp4
set -euo pipefail
POSTER_T=${1:-15.3}
ffmpeg -y -hide_banner -loglevel error -ss "$POSTER_T" -i video.mp4 -frames:v 1 -q:v 2 ../brag.jpg
ffmpeg -y -hide_banner -loglevel error -i video.mp4 -loop 1 -i ../brag.jpg -i music.wav \
  -filter_complex "[1:v]format=yuv420p[p];[0:v][p]overlay=enable='eq(n\,0)':shortest=1[v]" \
  -map "[v]" -map 2:a -t 38 -c:v libx264 -preset slow -crf 17 -pix_fmt yuv420p -profile:v high -r 30 \
  -c:a aac -b:a 192k -ar 48000 -movflags +faststart ../brag.mp4
ffprobe -v error -show_entries format=duration,size:stream=codec_name,width,height,r_frame_rate,nb_frames -of compact ../brag.mp4
