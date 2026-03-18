import numpy as np
import cv2
from ultralytics import FastSAM
import random
from rembg import remove
from PIL import Image
import torch 

class Generator: 

    def __init__(self): 
        self.segment_model = FastSAM('FastSAM-s.pt') 
            
    def get_mask(self, image): 
        if isinstance(image, str):
            loaded_image = Image.open(image)
        else:
            loaded_image = image

        mask_pil = remove(loaded_image, only_mask=True)
        final_mask = np.array(mask_pil)
         
        return final_mask
    
    def background_noise(self, image, mask):
        binary_mask = mask > 127
        
        if not np.any(binary_mask):
            return image
            
        output = image.copy()
        target_shape = output[binary_mask].shape
        
        noise = np.random.normal(128, 80, target_shape)
        output[binary_mask] = np.clip(noise, 0, 255).astype(np.uint8)
        
        return output

    def destructive_scatter(self, image, nest_mask, mask, clones=5): 
        if np.count_nonzero(mask) == 0: 
            return image
            
        output = image.copy()
        h_img, w_img = image.shape[:2]

        # Obtain legal area, 10 pixels buffer area
        # nest_mask = self.get_mask(image)
        binary_nest_mask = (nest_mask > 127).astype(np.uint8) * 255

        valid_zone = cv2.erode(binary_nest_mask, np.ones((10, 10), np.uint8), iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < 10: 
                continue
            
            # Extract impurity patch, create binary mask 
            x, y, w, h = cv2.boundingRect(contour)
            source_roi = image[y:y+h, x:x+w]
            source_mask = mask[y:y+h, x:x+w]
            
            patch_mask = np.zeros((h, w), dtype=np.uint8)
            patch_mask[source_mask > 0] = 255
            
            # Scatter clones
            placed = 0
            attempts = 0
            max_attempts = 200 + (clones * 10)
            
            while placed < clones and attempts < max_attempts:
                attempts += 1
                
                # Pick random position nearby
                spread = 80 + (clones * 2)
                offset_x = np.random.randint(-spread, spread)
                offset_y = np.random.randint(-spread, spread)
                
                center_x = x + w//2 + offset_x
                center_y = y + h//2 + offset_y
                
                # Calculate bounds
                tl_x, tl_y = center_x - w//2, center_y - h//2
                br_x, br_y = tl_x + w, tl_y + h
                
                # Boundary check
                if tl_x < 5 or tl_y < 5 or br_x > w_img - 5 or br_y > h_img - 5:
                    continue
                
                try:
                    # Ensure clones are within the mask
                    zone_slice = valid_zone[tl_y:br_y, tl_x:br_x]
                    
                    if zone_slice.shape != patch_mask.shape:
                        continue
                    if np.sum((patch_mask > 0) & (zone_slice == 0)) > 0: 
                        continue

                    # Paste logic
                    output = cv2.seamlessClone(
                        source_roi, output, patch_mask, (center_x, center_y), cv2.MIXED_CLONE
                    )
                    placed += 1

                except Exception:
                    pass
                    
        return output
    
    def apply_clahe(self, image, mask, clip_limit=3.0): 
        binary_mask = mask > 127
        
        if not np.any(binary_mask):
            return image
              
        output = image.copy()
        
        # 1. Convert to LAB color space 
        lab = cv2.cvtColor(output, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # 2. Apply CLAHE strictly to the Lightness (L) channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        
        # 3. Merge channels back and convert to RGB
        merged_lab = cv2.merge((cl, a_channel, b_channel))
        enhanced_img = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
        
        # A. Create a "lighting map" by heavily blurring the L channel
        local_bg = cv2.GaussianBlur(l_channel, (31, 31), 0)
        
        # B. Identify impurities (Lowered to 5 to catch more faint edges)
        impurity_mask = l_channel < (local_bg - 5)
        
        # C. Intersect the user's brush mask WITH the impurity mask
        # We convert it to uint8 (0 or 255) so OpenCV can process it
        targeted_mask = (binary_mask & impurity_mask).astype(np.uint8) * 255
        
        # D. NEW: Thicken the impurities! (Morphological Dilation)
        # A 3x3 kernel expands the dark spots slightly. 
        # Change iterations=2 if you want them even thicker!
        kernel = np.ones((3, 3), np.uint8)
        thickened_mask = cv2.dilate(targeted_mask, kernel, iterations=1)
        
        # 4. Expand the 2D mask to 3D so it maps to RGB channels
        mask_3d = (thickened_mask > 0)[:, :, np.newaxis]
        
        # 5. Blend: Apply enhanced pixels ONLY to the thickened dark spots
        output = np.where(mask_3d, enhanced_img, output)
        
        return output

    def remove_impurity(self, image, nest_mask, mask): 
        if np.count_nonzero(mask) == 0: 
            return image
            
        h_img, w_img = image.shape[:2]

        # 1. Binarize masks
        binary_mask = (mask > 127).astype(np.uint8) * 255
        binary_nest_mask = (nest_mask > 127).astype(np.uint8) * 255

        # 2. Define valid zone (Erode to stay strictly inside the nest)
        valid_zone = cv2.erode(binary_nest_mask, np.ones((7, 7), np.uint8), iterations=1)
        valid_zone[binary_mask > 0] = 0 

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_output = image.copy()

        successfully_removed = 0 

        for contour in contours:
            if cv2.contourArea(contour) < 10: 
                continue
            
            # --- Extract Impurity Info ---
            x, y, w, h = cv2.boundingRect(contour)
            source_mask = binary_mask[y:y+h, x:x+w]
            
            patch_mask = np.zeros((h, w), dtype=np.uint8)
            patch_mask[source_mask > 0] = 255
            
            # ==========================================
            # ERASE (Texture Harvesting)
            # ==========================================
            healed = False
            heal_attempts = 0
            
            while not healed and heal_attempts < 100:
                heal_attempts += 1
                
                # Pick a random clean spot to harvest fibers from
                hx = np.random.randint(0, w_img - w)
                hy = np.random.randint(0, h_img - h)
                
                zone_slice = valid_zone[hy:hy+h, hx:hx+w]
                if zone_slice.shape != patch_mask.shape:
                    continue
                if np.sum((patch_mask > 0) & (zone_slice == 0)) > 0: 
                    continue

                try:
                    # Harvest the clean texture
                    clean_roi = image[hy:hy+h, hx:hx+w]
                    
                    # NORMAL_CLONE perfectly stitches the clean fibers over the impurity
                    final_output = cv2.seamlessClone(
                        clean_roi, final_output, patch_mask, 
                        (x + w//2, y + h//2), cv2.NORMAL_CLONE
                    )
                    healed = True
                    successfully_removed += 1
                except Exception:
                    pass

            # Fallback to inpainting ONLY if harvesting fails on tiny nests
            if not healed:
                inpaint_mask = np.zeros_like(binary_mask)
                inpaint_mask[y:y+h, x:x+w] = patch_mask
                inpaint_mask = cv2.dilate(inpaint_mask, np.ones((5, 5), np.uint8), iterations=1)
                final_output = cv2.inpaint(final_output, inpaint_mask, 3, cv2.INPAINT_TELEA)
                successfully_removed += 1

        print(f"\nSummary: Successfully removed {successfully_removed}/{len(contours)} impurities")
        return final_output
    
if __name__ == "__main__": 
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    
    # 1. Setup
    # TEST_IMAGE = "dataset_hitl/unlabeled/(0)_LightFeather-2-_bmp_jpg.rf.8ac680312c062df4f2563f22d241e296.jpg"
    TEST_IMAGE = "dataset_hitl/unlabeled/(1)_Beige-11-_bmp_jpg.rf.8ee90c21ea247658ed3f151fd0ac5db8.jpg"
    
    # 2. Load
    image_bgr = cv2.imread(TEST_IMAGE)
    
    if image_bgr is not None:
        gen = Generator()

        # --- A. Test Background Noise ---
        # Create Dummy Mask (200x200 square)
        h, w = image_bgr.shape[:2]
        dummy_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(dummy_mask, (100, 100), (300, 300), 255, thickness=cv2.FILLED)
        
        result_bgr = gen.background_noise(image_bgr, dummy_mask)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        # --- B. Test Get Mask ---
        # Generate the nest mask from the original image
        generated_mask = gen.get_mask(image_bgr)

        # --- C. Display Side-by-Side ---
        plt.figure(figsize=(12, 6))

        # Plot 1: Noise Result
        plt.subplot(1, 2, 1)
        plt.imshow(result_rgb)
        plt.title("Result: Background Noise Applied")
        plt.axis('off')

        # Plot 2: Generated Mask
        plt.subplot(1, 2, 2)
        plt.imshow(generated_mask, cmap='gray')
        plt.title("Result: Generated Nest Mask")
        plt.axis('off')

        plt.show()
        
        print("✅ Test Complete!")
    else:
        print(f"Error loading image: {TEST_IMAGE}")
