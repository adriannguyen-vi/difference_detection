import cv2
import numpy as np
import json
import os
import rasterio
import logging
from rasterio.warp import transform

class GeospatialMapper:
    def __init__(self, ortho_image_path, output_dir):
        self.ortho_image_path = ortho_image_path
        self.output_dir = output_dir

    def map_contours_to_ortho(self, img_ref, valid_contours, output_filename="ortho_overlay.jpg"):
        """
        Maps detected tree-cutting contours from img_ref space onto a WebODM Orthomosaic.
        """

        if not valid_contours:
            logging.info("No valid contours to map to Ortho.")
            return
        
        logging.info(f"Loading Ortho image from {self.ortho_image_path}...")

        ortho_img = cv2.imread(self.ortho_image_path, cv2.IMREAD_COLOR)
        if ortho_img is None: return

        logging.info("Detecting keypoints for Ortho mapping...")
        sift = cv2.SIFT_create(nfeatures=200000)
        gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
        gray_ortho = cv2.cvtColor(ortho_img, cv2.COLOR_BGR2GRAY)

        kp_ref, des_ref = sift.detectAndCompute(gray_ref, None)
        kp_ortho, des_ortho = sift.detectAndCompute(gray_ortho, None)

        if des_ref is None or des_ortho is None:
            logging.info("Failed to find keypoints.")
            return

        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        logging.info("Matching features...")
        matches = flann.knnMatch(des_ref, des_ortho, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) < 10:
            logging.info(f"Not enough good matches found ({len(good_matches)}/10).")
            return
            
        logging.info(f"Found {len(good_matches)} good matches. Calculating Homography...")

        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ortho[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M_ortho, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M_ortho is None:
            logging.info("Failed to compute Homography matrix for Ortho mapping.")
            return

        warped_contours = [np.int32(cv2.perspectiveTransform(np.float32(cnt), M_ortho)) for cnt in valid_contours]
        self.export_geojson(warped_contours, output_filename.replace(".jpg", ".geojson"))

        ortho_overlay = ortho_img.copy()
        mask_layer = np.zeros_like(ortho_overlay)
        cv2.drawContours(mask_layer, warped_contours, -1, (0, 0, 255), -1)

        mask_indices = mask_layer > 0
        ortho_overlay[mask_indices] = cv2.addWeighted(ortho_img, 0.5, mask_layer, 0.5, 0)[mask_indices]
        cv2.drawContours(ortho_overlay, warped_contours, -1, (0, 255, 255), 3)

        output_path = os.path.join(self.output_dir, output_filename)
        cv2.imwrite(output_path, ortho_overlay)

        logging.info(f"Success! Ortho overlay saved to {output_path}")

    def export_geojson(self, warped_contours, output_filename):
        """
        Converts pixel contours to geographic coordinates and exports a GeoJSON.
        """

        try:
            with rasterio.open(self.ortho_image_path) as src:
                transform_matrix, crs = src.transform, src.crs
        except Exception as e:
            logging.info(f"Could not open Ortho for geospatial export: {e}")
            return

        features = []
        for cnt in warped_contours:
            poly_coords = [transform_matrix * (pt[0][0], pt[0][1]) for pt in cnt]
            if poly_coords[0] != poly_coords[-1]: poly_coords.append(poly_coords[0])

            if crs.to_string() != "EPSG:4326":
                xs, ys = zip(*poly_coords)
                poly_coords = list(zip(*transform(crs, 'EPSG:4326', xs, ys)))

            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [poly_coords]},
                "properties": {"change_type": "Tree Cutting / Clearing", "source": "Drone AI Detection"}
            })

        output_path = os.path.join(self.output_dir, output_filename)
        with open(output_path, "w") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f, indent=4)
        
        logging.info(f"GeoJSON exported successfully to {output_path}")