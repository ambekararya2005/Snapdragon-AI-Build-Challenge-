#!/usr/bin/env bash
# Rebuild ../brag.mp4 and ../brag.jpg from source. Run from anywhere: bash brag-output/work/build.sh
# Needs: node + playwright (Chromium), python3 + numpy, ffmpeg (libx264, aac), and the system fonts
# Inter, Inter Display and JetBrains Mono. The Devanagari font is downloaded once (Noto Sans Devanagari, OFL).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p assets
cp ../../demo/screenshots/overlay.png ../../demo/screenshots/overlay_help.png ../../demo/screenshots/caution.png \
   ../../demo/test_pages/mock_bank_transfer.html assets/
[ -f assets/NotoSansDevanagari.woff2 ] || curl -sSfL -o assets/NotoSansDevanagari.woff2 \
  "https://fonts.gstatic.com/s/notosansdevanagari/v33/TuG7UUFzXI5FBtUq5a8bjKYTZjtRU6Sgv3NaV_SNmI0b8QQCQmHN5TV_9qo.woff2"
python3 music.py music_raw.wav
python3 normalize.py
node render.cjs video.mp4
./finalize.sh 15.3
