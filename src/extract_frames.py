import cv2
import os
import json
import re
from utils import parse_drone_srt

def extract_frames_by_features(video_path, output_folder, match_threshold=0.1):
    video_basename = os.path.basename(video_path)
    output_folder = os.path.join(output_folder, os.path.splitext(video_basename)[0])
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    prev_descriptors = None
    frame_idx = 0
    metadata = {"video_path": video_path, "frames": []}
    srt_metadata = parse_drone_srt(video_path.replace('.mp4', '.srt').replace('.MP4', '.SRT'))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        keypoints, descriptors = orb.detectAndCompute(gray, None)
        if descriptors is None:
            continue

        if prev_descriptors is None:
            metadata["frames"].append({
                "frame_index": frame_idx,
                "frame_path": os.path.join(output_folder, f"frame_{frame_idx}.jpg"),
                **(srt_metadata[frame_idx - 1] if frame_idx - 1 < len(srt_metadata) else {})
            })
            cv2.imwrite(f"{output_folder}/frame_{frame_idx}.jpg", frame)
            prev_descriptors = descriptors
            prev_keypoints = keypoints
            print(f"Saved initial frame {frame_idx}")
            continue

        matches = bf.knnMatch(prev_descriptors, descriptors, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.85 * n.distance:
                good_matches.append(m)

        match_ratio = len(good_matches) / len(prev_descriptors)

        if match_ratio < match_threshold:
            out_frame_path = os.path.join(output_folder, f"frame_{frame_idx}.jpg")
            print(f"Saving frame {frame_idx} (Match Ratio: {match_ratio:.2f})")
            metadata["frames"].append({
                "frame_index": frame_idx,
                "frame_path": out_frame_path,
                **(srt_metadata[frame_idx - 1] if frame_idx - 1 < len(srt_metadata) else {})
            })
            cv2.imwrite(out_frame_path, frame)
            prev_descriptors = descriptors
            prev_keypoints = keypoints
            
            # # Show keypoints matched between current and previous frame (for debugging)
            # matched_img = cv2.drawMatches(
            #     frame, keypoints,
            #     cv2.imread(metadata["frames"][-2]["frame_path"]), prev_keypoints,
            #     good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # matched_img = cv2.resize(matched_img, (1280, 720))
            # cv2.imshow("Matched Keypoints", matched_img)
            # cv2.waitKey(1)

    with open(os.path.join(output_folder, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)
    cap.release()
    print(f"Done. Extracted {len(metadata['frames'])} frames to {output_folder}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract frames from video based on feature matching.")
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./data')
    parser.add_argument('--threshold', type=float, default=0.2)
    args = parser.parse_args()
    extract_frames_by_features(
        video_path=args.video_path,
        output_folder=args.output_dir,
        match_threshold=args.threshold
    )