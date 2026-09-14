import numpy as np
import cv2
from rembg import remove
from PIL import Image
from config import * 

class Generator: 

    def __init__(self): 
        pass 

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
        
        # Convert color space 
        lab = cv2.cvtColor(output, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        
        # Merge channels back and convert to RGB
        merged_lab = cv2.merge((cl, a_channel, b_channel))
        enhanced_img = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
        
        # Create a lighting map by blurring L channel
        local_bg = cv2.GaussianBlur(l_channel, (31, 31), 0)
        
        # Identify impurities
        impurity_mask = l_channel < (local_bg - 5)
        targeted_mask = (binary_mask & impurity_mask).astype(np.uint8) * 255
    
        kernel = np.ones((3, 3), np.uint8)
        thickened_mask = cv2.dilate(targeted_mask, kernel, iterations=1)
        
        # Expand 2D mask to 3D
        mask_3d = (thickened_mask > 0)[:, :, np.newaxis]

        # Apply enhanced pixels ONLY to the thickened dark spots
        output = np.where(mask_3d, enhanced_img, output)
         
        return output
    
    def remove_impurity(self, image, nest_mask, mask):
        if np.count_nonzero(mask) == 0:
            return image 

        h_img, w_img = image.shape[:2]

        binary_mask       = (mask > 127).astype(np.uint8) * 255
        binary_nest_mask  = (nest_mask > 127).astype(np.uint8) * 255

        valid_zone = cv2.erode(binary_nest_mask, np.ones((7, 7), np.uint8), iterations=1)
        valid_zone[binary_mask > 0] = 0

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_output = image.copy()
        successfully_removed = 0

        SHARPEN_KERNEL = np.array([[ 0,  -0.5,  0],
                                    [-0.5,  3, -0.5],
                                    [ 0,  -0.5,  0]], dtype=np.float32)

        NUM_CANDIDATES = 25   
        MAX_ATTEMPTS = 400  

        for contour in contours:
            if cv2.contourArea(contour) < 10:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            source_mask = binary_mask[y:y+h, x:x+w]
            patch_mask  = np.zeros((h, w), dtype=np.uint8)
            patch_mask[source_mask > 0] = 255
            patch_mask = cv2.dilate(patch_mask, np.ones((3, 3), np.uint8), iterations=2)

            ref_region   = final_output[y:y+h, x:x+w]
            border_mask  = patch_mask == 0
            has_border   = border_mask.any()

            best_score  = float('inf')
            best_roi    = None
            candidates  = 0
            attempts    = 0

            while candidates < NUM_CANDIDATES and attempts < MAX_ATTEMPTS:
                attempts += 1

                hx = np.random.randint(0, w_img - w)
                hy = np.random.randint(0, h_img - h)

                zone_slice = valid_zone[hy:hy+h, hx:hx+w]
                if zone_slice.shape != patch_mask.shape:
                    continue
                if np.any((patch_mask > 0) & (zone_slice == 0)):
                    continue

                candidates += 1
                candidate = final_output[hy:hy+h, hx:hx+w]

                if has_border:
                    diff  = cv2.absdiff(ref_region, candidate).astype(np.float32)
                    score = float(diff[border_mask].mean())
                else:
                    score = 0.0   

                if score < best_score:
                    best_score = score
                    best_roi   = candidate.copy()

            healed = False

            if best_roi is not None:
                try:
                    cx = int(np.clip(x + w // 2, w // 2 + 1, w_img - w // 2 - 1))
                    cy = int(np.clip(y + h // 2, h // 2 + 1, h_img - h // 2 - 1))

                    final_output = cv2.seamlessClone(
                        best_roi, final_output, patch_mask,
                        (cx, cy), cv2.NORMAL_CLONE
                    )

                    healed_region = final_output[y:y+h, x:x+w].copy()
                    sharpened     = cv2.filter2D(healed_region, -1, SHARPEN_KERNEL)
                    sharpened     = np.clip(sharpened, 0, 255).astype(np.uint8)

                    alpha = (patch_mask / 255.0)[..., np.newaxis]   # (h, w, 1)
                    blended = (sharpened * alpha + healed_region * (1.0 - alpha)).astype(np.uint8)
                    final_output[y:y+h, x:x+w] = blended

                    healed = True
                    successfully_removed += 1

                except Exception:
                    pass

            # Fallback
            if not healed:
                inpaint_mask = np.zeros_like(binary_mask)
                inpaint_mask[y:y+h, x:x+w] = patch_mask
                inpaint_mask = cv2.dilate(inpaint_mask, np.ones((5, 5), np.uint8), iterations=2)
                final_output = cv2.inpaint(final_output, inpaint_mask, 5, cv2.INPAINT_TELEA)
                successfully_removed += 1

        print(f"\nSummary: Successfully removed {successfully_removed}/{len(contours)} impurities")
        return final_output

if __name__ == "__main__": 
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    
    TEST_IMAGE = "dataset_hitl/unlabeled/(1)_Beige-11-_bmp_jpg.rf.8ee90c21ea247658ed3f151fd0ac5db8.jpg"
    
    image_bgr = cv2.imread(TEST_IMAGE)
    
    if image_bgr is not None:
        gen = Generator()
        h, w = image_bgr.shape[:2]
        dummy_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(dummy_mask, (100, 100), (300, 300), 255, thickness=cv2.FILLED)
        
        result_bgr = gen.background_noise(image_bgr, dummy_mask)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        generated_mask = gen.get_mask(image_bgr)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(result_rgb)
        plt.title("Result: Background Noise Applied")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(generated_mask, cmap='gray')
        plt.title("Result: Generated Nest Mask")
        plt.axis('off')

        plt.show()
        
        print("✅ Test Complete!")
    else:
        print(f"Error loading image: {TEST_IMAGE}")
