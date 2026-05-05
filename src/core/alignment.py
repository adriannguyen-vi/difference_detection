import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

class ImageAligner:
    def __init__(self, nfeatures=10000):
        self.sift = cv2.SIFT_create(nfeatures=nfeatures)
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    @staticmethod
    def calculate_ssim_optimized(image_1, image_2, downscale_factor=0.25):
        h, w = image_2.shape[:2]
        new_dims = (int(w * downscale_factor), int(h * downscale_factor))
        
        img1_small = cv2.resize(image_1, new_dims, interpolation=cv2.INTER_NEAREST)
        img2_small = cv2.resize(image_2, new_dims, interpolation=cv2.INTER_NEAREST)

        gray_1 = cv2.cvtColor(img1_small, cv2.COLOR_BGR2GRAY)
        gray_2 = cv2.cvtColor(img2_small, cv2.COLOR_BGR2GRAY)

        return ssim(gray_1, gray_2, data_range=255)

    def align_images(self, img_ref, img_target, ref_alt=None, new_alt=None):
        h, w = img_ref.shape[:2]
        
        # 1. Apply Theoretical Altitude Scale
        alt_scale = 1.0
        if ref_alt and new_alt and ref_alt > 0:
            alt_scale = new_alt / ref_alt
            
        if alt_scale != 1.0:
            new_w, new_h = int(img_target.shape[1] * alt_scale), int(img_target.shape[0] * alt_scale)
            if new_w > 10 and new_h > 10: 
                img_target_scaled = cv2.resize(img_target, (new_w, new_h))
            else:
                img_target_scaled = img_target.copy()
                alt_scale = 1.0
        else:
            img_target_scaled = img_target.copy()

        # 2. Apply performance scaling for feature matching
        perf_scale = 0.5
        img_ref_small = cv2.resize(img_ref, None, fx=perf_scale, fy=perf_scale)
        img_target_small = cv2.resize(img_target_scaled, None, fx=perf_scale, fy=perf_scale)

        gray_ref = cv2.cvtColor(img_ref_small, cv2.COLOR_BGR2GRAY)
        gray_target = cv2.cvtColor(img_target_small, cv2.COLOR_BGR2GRAY)

        # 3. Detect with SIFT
        kp1, des1 = self.sift.detectAndCompute(gray_ref, None)
        kp2, des2 = self.sift.detectAndCompute(gray_target, None)

        if des1 is None or des2 is None:
            return None, None

        # 4. Match with FLANN
        matches = self.flann.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) < 10:
            return None, None

        # 5. Extract points and adjust for BOTH scales
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) * (1 / perf_scale)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2) * (1 / (perf_scale * alt_scale))
        
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        if M is None:
            return None, None

        aligned_target = cv2.warpPerspective(img_target, M, (w, h))
        return aligned_target, M