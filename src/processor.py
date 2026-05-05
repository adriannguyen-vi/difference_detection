import cv2
import json
import os
import numpy as np
import logging 

from utils import parse_drone_srt
from core.telemetry import find_closest_telemetry
from core.alignment import ImageAligner
from core.detection import extract_valid_contours
from core.visualization import apply_heatmap_overlay
from core.geospatial import GeospatialMapper

class SiteChangeProcessor:
    def __init__(self, metadata_path, new_video_path, new_srt_path, output_dir="comparison_results", ortho_image_path=None, out_video_name=None):
        self.new_video_path = new_video_path
        self.output_dir = output_dir
        self.fps = 15 
        
        os.makedirs(output_dir, exist_ok=True)
        with open(metadata_path, "r") as f:
            self.data = json.load(f)

        
        logging.info(f"Parsing telemetry from {new_srt_path}...")
        self.new_telemetry = parse_drone_srt(new_srt_path)
        self.aligner = ImageAligner()
        
        self.geo_mapper = None
        if ortho_image_path and os.path.exists(ortho_image_path):
            self.geo_mapper = GeospatialMapper(ortho_image_path, output_dir)

        if out_video_name is not None:
            self.out_video_name = out_video_name
        else:
            self.out_video_name = "difference_summary.mp4"
    def find_best_match_spatial(self, cap, ref_item, ref_img, search_window_ms=1000):
        ref_lat, ref_lon, ref_alt = ref_item["latitude"], ref_item["longitude"], ref_item["rel_alt"]
        
        best_entry = find_closest_telemetry(self.new_telemetry, ref_lat, ref_lon, ref_alt)
        if not best_entry: return None, -1, None, None

        anchor_ms = best_entry["start_timestamp_ms"]
        new_alt = best_entry.get("rel_alt", ref_alt)
        
        start_ms, end_ms = max(0, anchor_ms - search_window_ms), anchor_ms + search_window_ms
        cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)

        best_score, best_frame, best_index = -1, None, -1
        ref_h, ref_w = ref_img.shape[:2]
        alt_scale = new_alt / ref_alt if ref_alt > 0 else 1.0

        while cap.get(cv2.CAP_PROP_POS_MSEC) <= end_ms:
            current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if not ret: break

            frame_for_ssim = cv2.resize(frame, (int(frame.shape[1] * alt_scale), int(frame.shape[0] * alt_scale))) if alt_scale != 1.0 else frame
            if frame_for_ssim.shape[:2] != (ref_h, ref_w):
                frame_for_ssim = cv2.resize(frame_for_ssim, (ref_w, ref_h))

            score = self.aligner.calculate_ssim_optimized(frame_for_ssim, ref_img)
            if score > best_score:
                best_score, best_frame, best_index = score, frame, current_frame_idx

        return best_frame, best_score, best_index, new_alt

    def process(self):
        cap = cv2.VideoCapture(self.new_video_path)
        sorted_frames = sorted(self.data["frames"], key=lambda x: x["frame_index"])
        
        if not cap.isOpened():
            ("Error: Could not open new video.")
            return

        if not sorted_frames:
            logging.error("Error: No frames found in metadata.")
            return
        h, w = cv2.imread(sorted_frames[0]["frame_path"]).shape[:2]
        out_w, out_h = int(w * 0.5), int(h * 0.5)
        
        out_vid_path = os.path.join(self.output_dir, self.out_video_name)

        out_video = cv2.VideoWriter(out_vid_path, 
                                    cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (out_w * 3, out_h))

        logging.info(f"Generating 3-panel video... Output will be saved to {out_vid_path}")

        for item in sorted_frames:
            idx = item["frame_index"]
            if not os.path.exists(item["frame_path"]): 
                continue
            img_ref = cv2.imread(item["frame_path"])
            
            best_frame, _, best_index, new_alt = self.find_best_match_spatial(cap, item, img_ref, search_window_ms=1500)
            
            if best_frame is None: 
                logging.info(f"Alignment failed for best match of frame {idx}. Skipping snippet.")
                continue

            best_aligned, best_M_mat = self.aligner.align_images(img_ref, best_frame, item.get("rel_alt"), new_alt)
            
            if best_aligned is None:
                logging.info(f"Alignment failed for best match of frame {idx}. Skipping snippet.")
                continue
            
            fixed_contours = extract_valid_contours(best_aligned, img_ref)
            
            if not fixed_contours:
                logging.info(f"No significant changes detected for Ref {idx}. Skipping snippet.")
                continue
            
            # Save the best match frame to debug dir 
            # debug_dir_path = "./debug_dir"
            # if not os.path.exists(debug_dir_path):
            #     os.makedirs(debug_dir_path)

            # overlay = apply_heatmap_overlay(best_frame, best_aligned, img_ref, fixed_contours, best_M_mat)
            # vis_ref = cv2.resize(img_ref, (out_w, out_h))
            # vis_curr = cv2.resize(best_frame, (out_w, out_h)) 
            # vis_overlay = cv2.resize(overlay, (out_w, out_h))
            # debug_img = np.hstack((vis_ref, vis_curr, vis_overlay))
            # # debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
            # image_path = os.path.join(debug_dir_path, f"debug_image_ref-{idx}_frame-{best_index}.png")
            # cv2.imwrite(image_path, debug_img)
            logging.info(f"Changes detected! Frame {best_index} for Ref {idx}...")
            logging.info(f"Going to create a short duration video of difference detection ...")
            # Write Summary Video frames

            start_vid, end_vid = max(0, best_index - 45), best_index + 45
            for f_idx in np.linspace(start_vid, end_vid, num=self.fps * 4, dtype=int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret: break
                
                aligned_frame, M_mat = self.aligner.align_images(img_ref, frame)
                if aligned_frame is None or M_mat is None: continue 

                overlay = apply_heatmap_overlay(frame, aligned_frame, img_ref, fixed_contours, M_mat)
                
                vis_ref = cv2.resize(img_ref, (out_w, out_h))
                vis_curr = cv2.resize(frame, (out_w, out_h)) 
                vis_overlay = cv2.resize(overlay, (out_w, out_h))
                
                cv2.putText(vis_ref, "Previous", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)
                cv2.putText(vis_curr, "Current", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)
                cv2.putText(vis_overlay, "Change", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 4)

                out_video.write(np.hstack((vis_ref, vis_curr, vis_overlay)))

            if self.geo_mapper:
                gps_info = {
                    "latitude": item.get("latitude"),
                    "longitude": item.get("longitude")
                }
                self.geo_mapper.map_contours_to_ortho(img_ref, fixed_contours, f"ortho_map_ref_{item['frame_index']}.jpg", gps_info=gps_info)

        cap.release()
        out_video.release()
        logging.info("Processing complete!")