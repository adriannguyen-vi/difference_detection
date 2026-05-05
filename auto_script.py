import os
import subprocess
from generate_heatmap_drone.utils import compress_and_upload

data_dir = "/home/ubuntu/adrian/drone_AI/progress_monitoring/data/C1/20260121"
videos_name = [x for x in os.listdir(data_dir) if x.lower().endswith(".mp4")]

# for vid_name in videos_name:
#     video_path = os.path.join(data_dir, vid_name)
#     command = ["python3", 
#                "/home/ubuntu/adrian/drone_AI/progress_monitoring/generate_heatmap_drone/extract_frames.py",
#               "--video_path", video_path,
#               "--output_dir", data_dir]
#     subprocess.run(command)

video_dirs = [x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))]
video_dirs.sort()
new_week_video_dir = "/home/ubuntu/adrian/drone_AI/progress_monitoring/data/C1/20260420"
for video_dir in video_dirs:
    metadata_path = os.path.join(data_dir, video_dir, "metadata.json")
    new_video_path = os.path.join(new_week_video_dir, f"{video_dir}.MP4")
    new_srt_path = os.path.join(new_week_video_dir, f"{video_dir}.SRT")
    output_dir = "/home/ubuntu/adrian/drone_AI/progress_monitoring/comparision_result_C1"
    out_video_name = f"{video_dir}.mp4"
    os.makedirs(output_dir, exist_ok=True)
    command = [
        "python3", 
        "/home/ubuntu/adrian/drone_AI/progress_monitoring/src/main.py",
        "--metadata", metadata_path,
        "--new_video", new_video_path,
        "--new_srt", new_srt_path,
        "--output_dir", output_dir,
        "--out_video_name", out_video_name
    ]
    subprocess.run(command)
    output_video_path = os.path.join(output_dir, out_video_name)

    compressed_video_name = f"{video_dir}_compressed.mp4"
    compressed_vid_path = os.path.join(output_dir, compressed_video_name)
    compress_and_upload(input_file=output_video_path, output_file=compressed_vid_path, up_load_s3=False)

    os.remove(output_video_path)