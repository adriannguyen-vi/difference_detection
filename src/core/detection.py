import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from functools import partial

def extract_valid_contours_heuristic(img_aligned, img_ref, sensitivity_threshold=100, min_area=0.01, max_area=0.40):
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY)
    
    # --- NEW: Create a mask to ignore the black borders from image warping ---
    # Find pixels where all color channels are exactly 0 (the warped background)
    black_border_mask = np.all(img_aligned == 0, axis=-1).astype(np.uint8) * 255
    valid_warp_mask = cv2.bitwise_not(black_border_mask)
    # Erode slightly to remove blending/interpolation artifacts right at the edge
    valid_warp_mask = cv2.erode(valid_warp_mask, np.ones((7, 7), np.uint8), iterations=1)
    # -------------------------------------------------------------------------
    
    diff_cv2 = cv2.absdiff(gray_ref, gray_aligned)
    diff_cv2 = diff_cv2.astype("float32") / 255.0
    diff_cv2 = cv2.GaussianBlur(diff_cv2, (15, 15), 0)

    _, diff_ssim = ssim(gray_ref, gray_aligned, full=True, data_range=255)
    diff_ssim_inv = cv2.GaussianBlur(1.0 - diff_ssim, (15, 15), 0)
    
    diff = np.clip((255 * (diff_cv2*0.7 + diff_ssim_inv*0.3)), 0, 255).astype("uint8")

    hsv_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2HSV)
    hsv_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2HSV)
    
    lower_green, upper_green = np.array([35, 40, 40]), np.array([85, 255, 255])
    
    tree_mask_ref = cv2.dilate(cv2.inRange(hsv_ref, lower_green, upper_green), np.ones((15, 15), np.uint8))
    tree_mask_aligned = cv2.dilate(cv2.inRange(hsv_aligned, lower_green, upper_green), np.ones((15, 15), np.uint8))
    
    valid_change_mask = cv2.bitwise_not(cv2.bitwise_and(tree_mask_ref, tree_mask_aligned))
    
    # --- UPDATED: Apply BOTH the change mask and the warp boundary mask ---
    combined_mask = cv2.bitwise_and(valid_change_mask, valid_warp_mask)
    diff = cv2.bitwise_and(diff, diff, mask=combined_mask)

    _, thresh = cv2.threshold(diff, sensitivity_threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh_clean = cv2.dilate(cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel), kernel, iterations=3) 

    contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_area = gray_ref.shape[0] * gray_ref.shape[1]
    valid_contours = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (min_area <= cv2.contourArea(cnt) / img_area <= max_area) and ((w * h) / img_area <= max_area):
            valid_contours.append(cnt)

    tree_cutting_contours = []
    hsv_ref_pure = cv2.cvtColor(img_ref, cv2.COLOR_BGR2HSV)
    hsv_aligned_pure = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2HSV)
    mask_green_ref_pure = cv2.inRange(hsv_ref_pure, lower_green, upper_green)
    mask_green_aligned_pure = cv2.inRange(hsv_aligned_pure, lower_green, upper_green)
    
    for cnt in valid_contours:
        cnt_mask = np.zeros(mask_green_ref_pure.shape, dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
        
        green_pixels_ref = cv2.countNonZero(cv2.bitwise_and(mask_green_ref_pure, mask_green_ref_pure, mask=cnt_mask))
        green_pixels_aligned = cv2.countNonZero(cv2.bitwise_and(mask_green_aligned_pure, mask_green_aligned_pure, mask=cnt_mask))
        
        contour_area = cv2.contourArea(cnt)
        if contour_area > 0 and green_pixels_ref > (contour_area * 0.05) and green_pixels_aligned < (green_pixels_ref * 0.40):
            tree_cutting_contours.append(cnt)

    return tree_cutting_contours


def extract_valid_contours_sam3(img_aligned, img_ref, sam_client=None, sensitivity_threshold=100, min_area=0.01, max_area=0.40):
    if sam_client is None:
        from core.sam_client import SAM3Client # Adjusted to standard import block if none provided
        sam_client = SAM3Client()

    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY)
    
    # --- NEW: Create a mask to ignore the black borders from image warping ---
    # Find pixels where all color channels are exactly 0 (the warped background)
    black_border_mask = np.all(img_aligned == 0, axis=-1).astype(np.uint8) * 255
    valid_warp_mask = cv2.bitwise_not(black_border_mask)
    # Erode slightly to remove blending/interpolation artifacts right at the edge
    valid_warp_mask = cv2.erode(valid_warp_mask, np.ones((7, 7), np.uint8), iterations=1)
    # -------------------------------------------------------------------------

    diff_cv2 = cv2.absdiff(gray_ref, gray_aligned)
    diff_cv2 = diff_cv2.astype("float32") / 255.0
    diff_cv2 = cv2.GaussianBlur(diff_cv2, (15, 15), 0)

    _, diff_ssim = ssim(gray_ref, gray_aligned, full=True, data_range=255)
    diff_ssim_inv = cv2.GaussianBlur(1.0 - diff_ssim, (15, 15), 0)
    
    diff = np.clip((255 * (diff_cv2 * 0.7 + diff_ssim_inv * 0.3)), 0, 255).astype("uint8")

    tree_mask_ref = sam_client.get_vegetation_mask(img_ref)
    tree_mask_aligned = sam_client.get_vegetation_mask(img_aligned)
    
    kernel_tree = np.ones((15, 15), np.uint8)
    tree_mask_ref_dilated = cv2.dilate(tree_mask_ref, kernel_tree, iterations=1)
    tree_mask_aligned_dilated = cv2.dilate(tree_mask_aligned, kernel_tree, iterations=1)
    
    persistent_trees = cv2.bitwise_and(tree_mask_ref_dilated, tree_mask_aligned_dilated)
    
    valid_change_mask = cv2.bitwise_not(persistent_trees)
    
    # --- UPDATED: Apply BOTH the change mask and the warp boundary mask ---
    combined_mask = cv2.bitwise_and(valid_change_mask, valid_warp_mask)
    diff = cv2.bitwise_and(diff, diff, mask=combined_mask)

    _, thresh = cv2.threshold(diff, sensitivity_threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh_clean = cv2.dilate(cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel), kernel, iterations=3) 

    contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_area = gray_ref.shape[0] * gray_ref.shape[1]
    valid_contours = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (min_area <= cv2.contourArea(cnt) / img_area <= max_area) and ((w * h) / img_area <= max_area):
            valid_contours.append(cnt)

    tree_cutting_contours = []
    
    for cnt in valid_contours:
        cnt_mask = np.zeros(tree_mask_ref.shape, dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
        
        tree_pixels_ref = cv2.countNonZero(cv2.bitwise_and(tree_mask_ref, tree_mask_ref, mask=cnt_mask))
        tree_pixels_aligned = cv2.countNonZero(cv2.bitwise_and(tree_mask_aligned, tree_mask_aligned, mask=cnt_mask))
        
        contour_area = cv2.contourArea(cnt)
        
        if contour_area > 0 and tree_pixels_ref > (contour_area * 0.05) and tree_pixels_aligned < (tree_pixels_ref * 0.40):
            tree_cutting_contours.append(cnt)

    return tree_cutting_contours


def extract_valid_contours_combined(img_aligned, img_ref, sam_client=None, sensitivity_threshold=100, min_area=0.01, max_area=0.40):
    if sam_client is None:
        from core.sam_client import SAM3Client
        sam_client = SAM3Client()

    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY)
    
    # --- 1. Create a mask to ignore the black borders from image warping ---
    black_border_mask = np.all(img_aligned == 0, axis=-1).astype(np.uint8) * 255
    valid_warp_mask = cv2.bitwise_not(black_border_mask)
    valid_warp_mask = cv2.erode(valid_warp_mask, np.ones((7, 7), np.uint8), iterations=1)

    # --- 2. Calculate structural and pixel differences ---
    diff_cv2 = cv2.absdiff(gray_ref, gray_aligned)
    diff_cv2 = diff_cv2.astype("float32") / 255.0
    diff_cv2 = cv2.GaussianBlur(diff_cv2, (15, 15), 0)

    _, diff_ssim = ssim(gray_ref, gray_aligned, full=True, data_range=255)
    diff_ssim_inv = cv2.GaussianBlur(1.0 - diff_ssim, (15, 15), 0)
    
    diff = np.clip((255 * (diff_cv2 * 0.7 + diff_ssim_inv * 0.3)), 0, 255).astype("uint8")

    # --- 3. Generate Heuristic Masks (Color-based) ---
    hsv_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2HSV)
    hsv_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2HSV)
    lower_green, upper_green = np.array([35, 40, 40]), np.array([85, 255, 255])
    
    heuristic_mask_ref = cv2.inRange(hsv_ref, lower_green, upper_green)
    heuristic_mask_aligned = cv2.inRange(hsv_aligned, lower_green, upper_green)

    # --- 4. Generate SAM3 Masks (Semantic-based) ---
    sam3_mask_ref = sam_client.get_vegetation_mask(img_ref)
    sam3_mask_aligned = sam_client.get_vegetation_mask(img_aligned)

    # --- 5. COMBINE MASKS: Intersection of Heuristic and SAM3 ---
    # A pixel is only a tree if it is BOTH strictly green AND semantically a tree.
    refined_tree_mask_ref = cv2.bitwise_and(heuristic_mask_ref, sam3_mask_ref)
    refined_tree_mask_aligned = cv2.bitwise_and(heuristic_mask_aligned, sam3_mask_aligned)

    # Dilate the refined masks to compute persistent trees robustly
    kernel_tree = np.ones((15, 15), np.uint8)
    tree_mask_ref_dilated = cv2.dilate(refined_tree_mask_ref, kernel_tree, iterations=1)
    tree_mask_aligned_dilated = cv2.dilate(refined_tree_mask_aligned, kernel_tree, iterations=1)
    
    # Persistent trees are areas that remained trees in both images
    persistent_trees = cv2.bitwise_and(tree_mask_ref_dilated, tree_mask_aligned_dilated)
    
    # We only care about changes where trees DID NOT persist
    valid_change_mask = cv2.bitwise_not(persistent_trees)
    
    # --- 6. Apply masks to the difference map ---
    combined_mask = cv2.bitwise_and(valid_change_mask, valid_warp_mask)
    diff = cv2.bitwise_and(diff, diff, mask=combined_mask)

    # --- 7. Threshold and clean up the difference map ---
    _, thresh = cv2.threshold(diff, sensitivity_threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh_clean = cv2.dilate(cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel), kernel, iterations=3) 

    contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_area = gray_ref.shape[0] * gray_ref.shape[1]
    valid_contours = []
    
    # Filter contours by size
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (min_area <= cv2.contourArea(cnt) / img_area <= max_area) and ((w * h) / img_area <= max_area):
            valid_contours.append(cnt)

    # --- 8. Final Contour Validation using Refined Masks ---
    tree_cutting_contours = []
    
    for cnt in valid_contours:
        cnt_mask = np.zeros(refined_tree_mask_ref.shape, dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
        
        # Use the REFINED pure masks (no dilation) to verify the actual transition of trees
        tree_pixels_ref = cv2.countNonZero(cv2.bitwise_and(refined_tree_mask_ref, refined_tree_mask_ref, mask=cnt_mask))
        tree_pixels_aligned = cv2.countNonZero(cv2.bitwise_and(refined_tree_mask_aligned, refined_tree_mask_aligned, mask=cnt_mask))
        
        contour_area = cv2.contourArea(cnt)
        
        # Check if it WAS a tree area and is NO LONGER a tree area
        if contour_area > 0 and tree_pixels_ref > (contour_area * 0.05) and tree_pixels_aligned < (tree_pixels_ref * 0.40):
            tree_cutting_contours.append(cnt)

    return tree_cutting_contours



def extract_valid_contours_variance(img_aligned, img_ref, sensitivity_threshold=100, min_area=0.01, max_area=0.40):
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY)
    
    black_border_mask = np.all(img_aligned == 0, axis=-1).astype(np.uint8) * 255
    valid_warp_mask = cv2.bitwise_not(black_border_mask)
    valid_warp_mask = cv2.erode(valid_warp_mask, np.ones((7, 7), np.uint8), iterations=1)
    
    diff_cv2 = cv2.absdiff(gray_ref, gray_aligned)
    diff_cv2 = diff_cv2.astype("float32") / 255.0
    diff_cv2 = cv2.GaussianBlur(diff_cv2, (15, 15), 0)

    _, diff_ssim = ssim(gray_ref, gray_aligned, full=True, data_range=255)
    diff_ssim_inv = cv2.GaussianBlur(1.0 - diff_ssim, (15, 15), 0)
    
    diff = np.clip((255 * (diff_cv2*0.7 + diff_ssim_inv*0.3)), 0, 255).astype("uint8")

    hsv_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2HSV)
    hsv_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2HSV)
    lower_green, upper_green = np.array([35, 40, 40]), np.array([85, 255, 255])
    
    color_mask_ref = cv2.inRange(hsv_ref, lower_green, upper_green)
    color_mask_aligned = cv2.inRange(hsv_aligned, lower_green, upper_green)
    
    kernel_size = 15
    min_variance = 50.0  
    
    def get_variance_mask(gray_img):
        gray_fp = gray_img.astype(np.float32)
        mu = cv2.blur(gray_fp, (kernel_size, kernel_size))
        mu_2 = cv2.blur(gray_fp**2, (kernel_size, kernel_size))
        variance = cv2.absdiff(mu_2, mu**2)
        return (variance > min_variance).astype(np.uint8) * 255
        
    texture_mask_ref = get_variance_mask(gray_ref)
    texture_mask_aligned = get_variance_mask(gray_aligned)
    
    refined_tree_mask_ref = cv2.bitwise_and(color_mask_ref, texture_mask_ref)
    refined_tree_mask_aligned = cv2.bitwise_and(color_mask_aligned, texture_mask_aligned)
    
    tree_mask_ref_dilated = cv2.dilate(refined_tree_mask_ref, np.ones((15, 15), np.uint8))
    tree_mask_aligned_dilated = cv2.dilate(refined_tree_mask_aligned, np.ones((15, 15), np.uint8))
    
    valid_change_mask = cv2.bitwise_not(cv2.bitwise_and(tree_mask_ref_dilated, tree_mask_aligned_dilated))
    combined_mask = cv2.bitwise_and(valid_change_mask, valid_warp_mask)
    diff = cv2.bitwise_and(diff, diff, mask=combined_mask)

    _, thresh = cv2.threshold(diff, sensitivity_threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh_clean = cv2.dilate(cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel), kernel, iterations=3) 

    contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_area = gray_ref.shape[0] * gray_ref.shape[1]
    valid_contours = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (min_area <= cv2.contourArea(cnt) / img_area <= max_area) and ((w * h) / img_area <= max_area):
            valid_contours.append(cnt)

    tree_cutting_contours = []
    
    for cnt in valid_contours:
        cnt_mask = np.zeros(refined_tree_mask_ref.shape, dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
        
        green_pixels_ref = cv2.countNonZero(cv2.bitwise_and(refined_tree_mask_ref, refined_tree_mask_ref, mask=cnt_mask))
        green_pixels_aligned = cv2.countNonZero(cv2.bitwise_and(refined_tree_mask_aligned, refined_tree_mask_aligned, mask=cnt_mask))
        
        contour_area = cv2.contourArea(cnt)
        if contour_area > 0 and green_pixels_ref > (contour_area * 0.05) and green_pixels_aligned < (green_pixels_ref * 0.40):
            tree_cutting_contours.append(cnt)

    return tree_cutting_contours


def extract_valid_contours_lbp(img_aligned, img_ref, sensitivity_threshold=100, min_area=0.01, max_area=0.40):
    try:
        from skimage.feature import local_binary_pattern
    except ImportError:
        print("skimage not fully available, falling back to variance filter for LBP variant.")
        return extract_valid_contours_variance(img_aligned, img_ref, sensitivity_threshold, min_area, max_area)

    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY)
    
    black_border_mask = np.all(img_aligned == 0, axis=-1).astype(np.uint8) * 255
    valid_warp_mask = cv2.bitwise_not(black_border_mask)
    valid_warp_mask = cv2.erode(valid_warp_mask, np.ones((7, 7), np.uint8), iterations=1)
    
    diff_cv2 = cv2.absdiff(gray_ref, gray_aligned)
    diff_cv2 = diff_cv2.astype("float32") / 255.0
    diff_cv2 = cv2.GaussianBlur(diff_cv2, (15, 15), 0)

    _, diff_ssim = ssim(gray_ref, gray_aligned, full=True, data_range=255)
    diff_ssim_inv = cv2.GaussianBlur(1.0 - diff_ssim, (15, 15), 0)
    
    diff = np.clip((255 * (diff_cv2*0.7 + diff_ssim_inv*0.3)), 0, 255).astype("uint8")

    hsv_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2HSV)
    hsv_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2HSV)
    lower_green, upper_green = np.array([35, 40, 40]), np.array([85, 255, 255])
    
    color_mask_ref = cv2.inRange(hsv_ref, lower_green, upper_green)
    color_mask_aligned = cv2.inRange(hsv_aligned, lower_green, upper_green)
    
    radius = 3
    n_points = 8 * radius
    
    def get_lbp_mask(gray_img):
        lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
        kernel_size = 15
        min_lbp_variance = 5.0
        
        lbp_fp = lbp.astype(np.float32)
        mu = cv2.blur(lbp_fp, (kernel_size, kernel_size))
        mu_2 = cv2.blur(lbp_fp**2, (kernel_size, kernel_size))
        variance = cv2.absdiff(mu_2, mu**2)
        return (variance > min_lbp_variance).astype(np.uint8) * 255
        
    texture_mask_ref = get_lbp_mask(gray_ref)
    texture_mask_aligned = get_lbp_mask(gray_aligned)
    
    refined_tree_mask_ref = cv2.bitwise_and(color_mask_ref, texture_mask_ref)
    refined_tree_mask_aligned = cv2.bitwise_and(color_mask_aligned, texture_mask_aligned)
    
    tree_mask_ref_dilated = cv2.dilate(refined_tree_mask_ref, np.ones((15, 15), np.uint8))
    tree_mask_aligned_dilated = cv2.dilate(refined_tree_mask_aligned, np.ones((15, 15), np.uint8))
    
    valid_change_mask = cv2.bitwise_not(cv2.bitwise_and(tree_mask_ref_dilated, tree_mask_aligned_dilated))
    combined_mask = cv2.bitwise_and(valid_change_mask, valid_warp_mask)
    diff = cv2.bitwise_and(diff, diff, mask=combined_mask)

    _, thresh = cv2.threshold(diff, sensitivity_threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh_clean = cv2.dilate(cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel), kernel, iterations=3) 

    contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_area = gray_ref.shape[0] * gray_ref.shape[1]
    valid_contours = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (min_area <= cv2.contourArea(cnt) / img_area <= max_area) and ((w * h) / img_area <= max_area):
            valid_contours.append(cnt)

    tree_cutting_contours = []
    
    for cnt in valid_contours:
        cnt_mask = np.zeros(refined_tree_mask_ref.shape, dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
        
        green_pixels_ref = cv2.countNonZero(cv2.bitwise_and(refined_tree_mask_ref, refined_tree_mask_ref, mask=cnt_mask))
        green_pixels_aligned = cv2.countNonZero(cv2.bitwise_and(refined_tree_mask_aligned, refined_tree_mask_aligned, mask=cnt_mask))
        
        contour_area = cv2.contourArea(cnt)
        if contour_area > 0 and green_pixels_ref > (contour_area * 0.05) and green_pixels_aligned < (green_pixels_ref * 0.40):
            tree_cutting_contours.append(cnt)

    return tree_cutting_contours


def extract_valid_contours_entropy(img_aligned, img_ref, sensitivity_threshold=100, min_area=0.01, max_area=0.40):
    try:
        from skimage.filters.rank import entropy
        from skimage.morphology import disk
    except ImportError:
        print("skimage filters not fully available, falling back to variance.")
        return extract_valid_contours_variance(img_aligned, img_ref, sensitivity_threshold, min_area, max_area)

    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY)
    
    black_border_mask = np.all(img_aligned == 0, axis=-1).astype(np.uint8) * 255
    valid_warp_mask = cv2.bitwise_not(black_border_mask)
    valid_warp_mask = cv2.erode(valid_warp_mask, np.ones((7, 7), np.uint8), iterations=1)
    
    diff_cv2 = cv2.absdiff(gray_ref, gray_aligned)
    diff_cv2 = diff_cv2.astype("float32") / 255.0
    diff_cv2 = cv2.GaussianBlur(diff_cv2, (15, 15), 0)

    _, diff_ssim = ssim(gray_ref, gray_aligned, full=True, data_range=255)
    diff_ssim_inv = cv2.GaussianBlur(1.0 - diff_ssim, (15, 15), 0)
    
    diff = np.clip((255 * (diff_cv2*0.7 + diff_ssim_inv*0.3)), 0, 255).astype("uint8")

    hsv_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2HSV)
    hsv_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2HSV)
    lower_green, upper_green = np.array([35, 40, 40]), np.array([85, 255, 255])
    
    color_mask_ref = cv2.inRange(hsv_ref, lower_green, upper_green)
    color_mask_aligned = cv2.inRange(hsv_aligned, lower_green, upper_green)
    
    def get_entropy_mask(gray_img):
        # Calculate local entropy with a disk footprint
        ent_img = entropy(gray_img, disk(7))
        min_entropy = 4.0 
        return (ent_img > min_entropy).astype(np.uint8) * 255
        
    texture_mask_ref = get_entropy_mask(gray_ref)
    texture_mask_aligned = get_entropy_mask(gray_aligned)
    
    refined_tree_mask_ref = cv2.bitwise_and(color_mask_ref, texture_mask_ref)
    refined_tree_mask_aligned = cv2.bitwise_and(color_mask_aligned, texture_mask_aligned)
    
    tree_mask_ref_dilated = cv2.dilate(refined_tree_mask_ref, np.ones((15, 15), np.uint8))
    tree_mask_aligned_dilated = cv2.dilate(refined_tree_mask_aligned, np.ones((15, 15), np.uint8))
    
    valid_change_mask = cv2.bitwise_not(cv2.bitwise_and(tree_mask_ref_dilated, tree_mask_aligned_dilated))
    combined_mask = cv2.bitwise_and(valid_change_mask, valid_warp_mask)
    diff = cv2.bitwise_and(diff, diff, mask=combined_mask)

    _, thresh = cv2.threshold(diff, sensitivity_threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh_clean = cv2.dilate(cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel), kernel, iterations=3) 

    contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_area = gray_ref.shape[0] * gray_ref.shape[1]
    valid_contours = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (min_area <= cv2.contourArea(cnt) / img_area <= max_area) and ((w * h) / img_area <= max_area):
            valid_contours.append(cnt)

    tree_cutting_contours = []
    
    for cnt in valid_contours:
        cnt_mask = np.zeros(refined_tree_mask_ref.shape, dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
        
        green_pixels_ref = cv2.countNonZero(cv2.bitwise_and(refined_tree_mask_ref, refined_tree_mask_ref, mask=cnt_mask))
        green_pixels_aligned = cv2.countNonZero(cv2.bitwise_and(refined_tree_mask_aligned, refined_tree_mask_aligned, mask=cnt_mask))
        
        contour_area = cv2.contourArea(cnt)
        if contour_area > 0 and green_pixels_ref > (contour_area * 0.05) and green_pixels_aligned < (green_pixels_ref * 0.40):
            tree_cutting_contours.append(cnt)

    return tree_cutting_contours


try: 
    from core.sam3_client import SAM3Client
    sam3_client = SAM3Client()
    extract_valid_contours = partial(extract_valid_contours_combined, sam_client=sam3_client)
except ImportError:
    extract_valid_contours = extract_valid_contours_heuristic
