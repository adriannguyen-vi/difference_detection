import cv2
import numpy as np
import json
import os
import re
import math
import subprocess
import os


def compress_and_upload(input_file, output_file, up_load_s3=False):
  

    # 1. Run FFmpeg compression
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",               # Automatically overwrite existing output file
        "-i", input_file,
        "-vcodec", "libx264",
        "-crf", "23",
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-acodec", "aac",
        "-movflags", "+faststart",
        output_file
    ]

    print(f"Starting compression: {input_file} ...")
    try:
        # check=True ensures an exception is thrown if the command fails
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Compression complete: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error: FFmpeg compression failed.\n{e}")
        return  # Stop execution if compression fails

    if up_load_s3:
        # 2. Run AWS S3 upload
        S3_DESTINATION = "s3://viact-adrian-storage-655384763347-ap-east-1-an/"
        aws_cmd = [
            "aws", "s3", "cp", 
            output_file, 
            S3_DESTINATION
        ]

        print(f"Starting upload to {S3_DESTINATION} ...")
        try:
            subprocess.run(aws_cmd, check=True)
            print("Upload complete!")
        except subprocess.CalledProcessError as e:
            print(f"Error: AWS S3 upload failed.\n{e}")


def parse_drone_srt(srt_path):
    data = []
    time_pattern = re.compile(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})')
    attr_pattern = re.compile(r'(\w+)\s*:\s*([^\s\]]+)')
    datetime_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})')
    
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = content.strip().split('\n\n')

    for block in blocks:
        lines = block.split('\n')
        if len(lines) < 3:
            continue
        
        time_match = time_pattern.search(lines[1])
        if not time_match:
            continue
            
        start_time_str, end_time_str = time_match.groups()
        entry = {
            "start_timestamp": start_time_str,
            "start_timestamp_ms": time_str_to_ms(start_time_str),
            "end_timestamp": end_time_str,
            "end_timestamp_ms": time_str_to_ms(end_time_str),
        }
        
        telemetry_text = " ".join(lines[2:])
        matches = attr_pattern.findall(telemetry_text)
        dt_match = datetime_pattern.search(telemetry_text)
        if dt_match:
            entry['datetime'] = dt_match.group(1)

        for key, value in matches:
            try:
                if key.isnumeric():
                    continue
                if '.' in value:
                    entry[key] = float(value)
                else:
                    entry[key] = int(value)
            except ValueError:
                entry[key] = value

        data.append(entry)

    return data

def time_str_to_ms(time_str):
    h, m, s_ms = time_str.split(':')
    s, ms = s_ms.split(',')
    total_ms = (int(h) * 3600000) + (int(m) * 60000) + (int(s) * 1000) + int(ms)
    return total_ms

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the Earth surface."""
    R = 6371000  # Radius of Earth in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

