import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def apply_heatmap_overlay(unaligned_frame, aligned_frame, img_ref, valid_contours, M):
    overlay_result = unaligned_frame.copy()
    
    if not valid_contours or M is None:
        return overlay_result

    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_aligned = cv2.cvtColor(aligned_frame, cv2.COLOR_BGR2GRAY)
    
    diff_cv2 = cv2.absdiff(gray_ref, gray_aligned)
    diff_cv2 = cv2.GaussianBlur(diff_cv2.astype("float32") / 255.0, (15, 15), 0)

    _, diff_ssim = ssim(gray_ref, gray_aligned, full=True, data_range=255)
    diff_ssim_inv = cv2.GaussianBlur(1.0 - diff_ssim, (15, 15), 0)
    
    diff = np.clip((255 * (diff_cv2*0.7 + diff_ssim_inv*0.3)), 0, 255).astype("uint8")

    contour_mask = np.zeros(gray_ref.shape, dtype=np.uint8)
    heatmap_color = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    
    for cnt in valid_contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        cv2.rectangle(heatmap_color, (x, y), (x+w_box, y+h_box), (0, 255, 255), 2)
        cv2.rectangle(contour_mask, (x, y), (x+w_box, y+h_box), 255, 2)
        cv2.drawContours(contour_mask, [cnt], -1, 255, -1)

    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return overlay_result 
    
    h, w = unaligned_frame.shape[:2]
    heatmap_warped = cv2.warpPerspective(heatmap_color, M_inv, (w, h))
    mask_warped = cv2.warpPerspective(contour_mask, M_inv, (w, h), flags=cv2.INTER_NEAREST)

    mask_indices = mask_warped > 0
    overlay_result[mask_indices] = cv2.addWeighted(unaligned_frame, 0.5, heatmap_warped, 0.7, 0)[mask_indices]
            
    return overlay_result