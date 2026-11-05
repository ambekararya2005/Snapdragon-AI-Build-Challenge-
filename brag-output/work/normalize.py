"""Two-pass loudness normalization: music_raw.wav -> music.wav at -14 LUFS, -1.5 dBTP (linear, no pumping)."""
import json
import re
import subprocess

TARGET = "I=-14:TP=-1.5:LRA=11"
p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", "music_raw.wav", "-af", f"loudnorm={TARGET}:print_format=json",
                    "-f", "null", "-"], capture_output=True, text=True, check=True)
j = json.loads(re.search(r"\{[^{}]*\}", p.stderr).group(0))
af = (f"loudnorm={TARGET}:measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}:"
      f"measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true")
subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", "music_raw.wav", "-af", af, "-ar", "48000", "music.wav"], check=True)
print("music.wav normalized from", j["input_i"], "LUFS")
