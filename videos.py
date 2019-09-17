import os
from os.path import isfile, join
import subprocess
from functools import reduce

def parse_attempt(name):
  return f.split(".")[-2].replace("video", "")

videos_dir = "./videos/"
video_files = [f for f in os.listdir(videos_dir) if isfile(join(videos_dir, f)) and f.endswith(".mp4")]
os.chdir(videos_dir)
video_files.sort()
for i, f in enumerate(video_files):
  attempt = parse_attempt(f)
  cmd = "ffmpeg -i %s -vf drawtext=\"text='Attempt %s':fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5: \
      boxborderw=5:x=w-tw-10:y=h-th-10\" -codec:a copy %d.mp4" % (f, attempt, i)
  print(cmd)
  subprocess.call(cmd, shell=True)

names = ["file %d.mp4" % (i) for i in range(len(video_files))]
fp = open("myfiles.txt", "w")
fp.write("\n".join(names))
fp.close()
concat_cmd = "ffmpeg -f concat -i myfiles.txt -c copy output.mp4"
print(concat_cmd)
subprocess.call(concat_cmd, shell=True)

