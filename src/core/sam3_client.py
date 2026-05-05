import requests
import cv2
import numpy as np
import pycocotools.mask as mask_util
import logging
import os
import time

class SAM3Client:
    def __init__(self, host="192.168.89.101", port=8011):
        self.url = f"http://{host}:{port}/segment"

    def get_vegetation_mask(self, image, queries="forest", conf=0.4):
        """
        Sends an image to the SAM3 API and returns a binary mask of detected queries.
        """
        # Encode the image to memory to send over HTTP
        success, img_encoded = cv2.imencode('.jpg', image)
        if not success:
            logging.error("Failed to encode image for SAM3 API.")
            return np.zeros(image.shape[:2], dtype=np.uint8)

        try:
            r = requests.post(
                self.url,
                params={"text": queries, "conf": conf}, # Using the text and conf params
                files={"file": ("image.jpg", img_encoded.tobytes(), "image/jpeg")}, # Sending as multipart/form-data
                timeout=300 # Wait up to 5 minutes
            )
            r.raise_for_status()
            data = r.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"SAM3 API request failed: {e}", exc_info=True)
            return np.zeros(image.shape[:2], dtype=np.uint8)

        # Initialize an empty mask using the original image dimensions
        original_h, original_w = image.shape[:2]
        final_mask = np.zeros((original_h, original_w), dtype=np.uint8)

        # Retrieve sizing data to scale masks correctly
        orig_size = data.get("original_size", {"width": original_w, "height": original_h})
        proc_size = data.get("processed_size", {"width": original_w, "height": original_h})
        
        sx = orig_size["width"] / proc_size["width"] # Calculate width scaling factor
        sy = orig_size["height"] / proc_size["height"] # Calculate height scaling factor

        # Process each instance prediction
        logging.info(f"Number of SAM3 instance: {len(data.get('predictions', []))}")
        for pred in data.get("predictions", []):
            rle = pred["mask"]
            
            # Decode the COCO-style RLE string into a binary numpy array
            binary_mask = mask_util.decode(rle) 
            
            # Masks are returned in processed_size space and must be scaled back
            if sx != 1.0 or sy != 1.0:
                binary_mask = cv2.resize(
                    binary_mask, 
                    (orig_size["width"], orig_size["height"]), 
                    interpolation=cv2.INTER_NEAREST
                )
            
            # Combine this instance's mask with the final mask
            final_mask = cv2.bitwise_or(final_mask, binary_mask)
        
        # Convert the 0/1 binary mask to a 0/255 mask for standard OpenCV compatibility
        out_mask = final_mask * 255

        # # ==========================================
        # # DEBUG CODE: Save Original | Mask | Overlay
        # # ==========================================
        # try:
        #     debug_dir = "debug_sam3_dir"
        #     os.makedirs(debug_dir, exist_ok=True)

        #     # 1. Convert 1-channel mask to 3-channel so it can be concatenated with BGR images
        #     mask_3c = cv2.cvtColor(out_mask, cv2.COLOR_GRAY2BGR)

        #     # 2. Create an overlay (Green tint over detected areas)
        #     color_layer = np.zeros_like(image)
        #     color_layer[out_mask == 255] = [0, 255, 0]  # BGR format (Green)
        #     overlay_img = cv2.addWeighted(image, 0.7, color_layer, 0.3, 0)

        #     # 3. Concatenate horizontally: Original | Mask | Overlay
        #     debug_panel = np.hstack((image, mask_3c, overlay_img))

        #     # 4. Save to disk using a timestamp to prevent overwriting
        #     debug_filename = f"sam3_debug_{int(time.time() * 1000)}.jpg"
        #     debug_path = os.path.join(debug_dir, debug_filename)
        #     cv2.imwrite(debug_path, debug_panel)
        #     logging.debug(f"Saved SAM3 debug visualization to {debug_path}")

        # except Exception as e:
        #     logging.error(f"Failed to generate/save debug image: {e}")
        # # ==========================================

        return out_mask