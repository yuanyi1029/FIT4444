import numpy as np
import cv2
from ultralytics import FastSAM
import random

class Generator: 

    def __init__(self): 
        self.segment_model = FastSAM('FastSAM-s.pt') 
            
    def get_mask(self, image): 
        # Inference
        results = self.segment_model(image, imgsz=640, conf=0.6, iou=0.9)
        
        if results[0].masks is not None:
            # Extract masks, select largest mask 
            masks_data = results[0].masks.data.cpu().numpy()
            
            best_mask = max(masks_data, key=lambda m: m.sum())            
            final_mask = (best_mask * 255).astype(np.uint8)
            
            return final_mask
            
        h, w = results[0].orig_shape
        return np.zeros((h, w), dtype=np.uint8)
    
    def background_noise(self, image, mask):
        if np.count_nonzero(mask) == 0:
            return image 
        
        # Identfiy mask, generate noise 
        binary_mask = mask > 127
        noise = np.random.randint(0, 256, image.shape, dtype=np.uint8)

        # Apply noise 
        output = image.copy()
        output[binary_mask] = noise[binary_mask]
        return output
    
    def destructive_scatter(self, image, mask, clones=5): 
        if np.count_nonzero(mask) == 0: 
            return image
            
        output = image.copy()
        h_img, w_img = image.shape[:2]

        # Obtain legal area, 10 pixels buffer area
        nest_mask = self.get_mask(image)
        valid_zone = cv2.erode(nest_mask, np.ones((10, 10), np.uint8), iterations=2)
        
        # Find countours
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

# class Generator: 

#     def __init__(self): 
#         pass 

#     def get_mask(self, image): 
#         pass 
    
#     def background_noise(self, image, mask):
#         pass 

#     def destructive_scatter(self, image): 
#         pass 

#     def destructive_transform(self, image): 
#         pass  