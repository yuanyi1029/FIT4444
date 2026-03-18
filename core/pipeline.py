from core.inference import * 
from core.saliency import * 
from core.generator import *
from core.dataset_manager import *  
from config import * 
import uuid

class HITLPipeline: 
    
    def __init__(self, yolo_model, torch_model):
        self.yolo_model = yolo_model
        self.torch_model = torch_model
        self.generator = Generator()
        self.dataset_manager = DatasetManager()
    
    # def init_active_session(self, input_files):
    #     queue = []
    #     print(f"Analyzing {len(input_files)} images for active learning...")
    #     for file in input_files:
    #         file_path = file.name
            
    #         filename = Path(file_path).name
    #         prefix = filename.split('_')[0]
    #         if prefix != "0":
    #             continue
            
    #         result = predict_image(self.yolo_model, file_path)
            
    #         if result:
    #             queue.append({
    #                 "path": file_path,
    #                 "margin": result['margin'],
    #                 "class": result['class']
    #             })
                
    #     # Sort the queue so the lowest margin (most confusing) is at index 0
    #     sorted_queue = sorted(queue, key=lambda x: x['margin'])
    #     return sorted_queue
    
    def init_active_session(self, input_files):
        queue = []
        print(f"Analyzing {len(input_files)} images for active learning...")
        
        for file in input_files:
            file_path = file.name
            
            result = predict_image(self.yolo_model, file_path)
            
            if result:
                queue.append({
                    "path": file_path,
                    "margin": result['margin'],
                    "class": result['class']
                })
                
        # Sort the queue so the lowest margin (most confusing) is at index 0
        sorted_queue = sorted(queue, key=lambda x: x['margin'])
        return sorted_queue

    def process_prediction(self, input_image):
        result = predict_image(self.yolo_model, input_image)
        saliency_map, _ = generate_saliency(self.torch_model, input_image)
        segmentation_mask = self.generator.get_mask(input_image)
        
        return {
            'result': result,
            'saliency': saliency_map,
            'segmentation': segmentation_mask
        }
    
    # def generate_counterexamples(self, original_image, original_path, current_grade, editor_data):
    #     # clean_name = Path(filename).stem
    #     outputs = [{ 
    #         "image": original_image, 
    #         "label": current_grade,
    #         "type": f"o-{original_path}",
    #         "folder": "corrected"
    #     }]
             
    #     subject_mask = self.generator.get_mask(original_image)
    #     background_mask = 255 - subject_mask
    #     bg_scatter_img = self.generator.background_noise(original_image, background_mask)
    #     outputs.append({
    #         "image": bg_scatter_img,
    #         "label": current_grade,
    #         "type": f"sn-{original_path}",
    #         "folder": "segnoise"
    #     })
    
    #     if editor_data is None or not editor_data["layers"]:
    #         return outputs
        
    #     # Extract mask  
    #     drawing_layer = editor_data["layers"][0]
    #     alpha = drawing_layer[:, :, 3] 
    #     drawn_mask = alpha > 10

    #     # Get hue channel  
    #     drawing_rgb = drawing_layer[:, :, :3]
    #     hsv_drawing = cv2.cvtColor(drawing_rgb, cv2.COLOR_RGB2HSV)
    #     hue = hsv_drawing[:, :, 0] 

    #     # Segment colours
    #     is_red = ((hue < 20) | (hue > 160)) & drawn_mask
    #     is_green = (hue > 35) & (hue < 85) & drawn_mask
    #     is_blue = (hue > 85) & (hue < 140) & drawn_mask

    #     red_mask = is_red.astype(np.uint8) * 255
    #     green_mask = is_green.astype(np.uint8) * 255
    #     blue_mask = is_blue.astype(np.uint8) * 255
        
    #     # Red 
    #     if np.any(is_red):
    #         noisy_base_img = self.generator.background_noise(original_image, red_mask)
    #     else:
    #         noisy_base_img = original_image.copy()

    #     if np.any(is_red):
    #         noisy_base_img = self.generator.background_noise(original_image, red_mask)
    #         outputs.append({ 
    #             "image": noisy_base_img, 
    #             "label": current_grade,
    #             "type": f"rn-{original_path}",
    #             "folder": "regnoise"
    #         })
    #     else:
    #         noisy_base_img = original_image.copy()

    #     if np.any(is_green):
    #         impurity_levels = [5, 15, 50]
    #         labels = ["s", "m", "l"]     
    #         folder_names = ["smallscatter", "mediumscatter", "largescatter"]
    #         for level, label, folder in zip(impurity_levels, labels, folder_names): 
    #             scatter_img = self.generator.destructive_scatter(noisy_base_img, subject_mask, green_mask, clones=level)
    #             outputs.append({ 
    #                 "image": scatter_img, 
    #                 "label": current_grade,
    #                 "type": f"{label}-{original_path}",
    #                 "folder": folder 
    #             })

    #         clahe_img = self.generator.apply_clahe(noisy_base_img, green_mask)
    #         outputs.append({
    #             "image": clahe_img,
    #             "label": current_grade,
    #             "type": f"ch-{original_path}",
    #             "folder": "clahe"
    #         })
            
    #         remove_img = self.generator.remove_impurity(noisy_base_img, subject_mask, green_mask)
    #         outputs.append({
    #             "image": remove_img,
    #             "label": current_grade,
    #             "type": f"rm-{original_path}",
    #             "folder": "remove"
    #         })

    #     # Blue
    #     if np.any(is_blue):
    #         print("Blue brush detected (Logic not connected yet)")
        
    #     return outputs

    def generate_counterexamples(self, original_image, original_path, current_grade, editor_data):
        outputs = []
        
        # ==========================================
        # 1. The Baseline (Item 1)
        # ==========================================
        outputs.append({ 
            "image": original_image, 
            "label": current_grade,
            "type": f"o-{original_path}",
            "folder": "baseline-clean"
        })
             
        # If no annotations exist, stop here (Prevents Grade 0 Duplicates)
        if editor_data is None or not editor_data["layers"]:
            return outputs
        
        # --- Extract Brush Masks ---
        drawing_layer = editor_data["layers"][0]
        alpha = drawing_layer[:, :, 3] 
        drawn_mask = alpha > 10

        drawing_rgb = drawing_layer[:, :, :3]
        hsv_drawing = cv2.cvtColor(drawing_rgb, cv2.COLOR_RGB2HSV)
        hue = hsv_drawing[:, :, 0] 

        # Segment colours based on user drawing
        is_red = ((hue < 20) | (hue > 160)) & drawn_mask
        is_green = (hue > 35) & (hue < 85) & drawn_mask
        is_blue = (hue > 85) & (hue < 140) & drawn_mask

        red_mask = is_red.astype(np.uint8) * 255
        green_mask = is_green.astype(np.uint8) * 255
        
        has_red = np.any(is_red)
        has_green = np.any(is_green)
        
        # We still need the subject mask for scatter boundaries and removal zones
        subject_mask = self.generator.get_mask(original_image)
        
        # ==========================================
        # FACTOR A: THE NOISY BASELINE (Item 2 - regnoise)
        # ==========================================
        if has_red:
            noisy_base_img = self.generator.background_noise(original_image, red_mask)
            outputs.append({
                "image": noisy_base_img,
                "label": current_grade,
                "type": f"n-{original_path}",
                "folder": "baseline_noise" 
            })

        # ==========================================
        # FACTOR B: THE FOREGROUND EDITS (Items 3-12)
        # ==========================================
        if has_green:
            
            # --- SYNTHETIC INJECTION (Scatters) ---
            impurity_levels = [5, 15, 50]
            labels = ["s", "m", "l"]     
            folder_names = ["smallscatter", "mediumscatter", "largescatter"]
            
            for level, label, folder in zip(impurity_levels, labels, folder_names): 
                # 1. Generate the CLEAN scatter first
                clean_scatter = self.generator.destructive_scatter(original_image, subject_mask, green_mask, clones=level)
                outputs.append({ 
                    "image": clean_scatter, 
                    "label": current_grade,
                    "type": f"{label}r-{original_path}",
                    "folder": f"{folder}_regular" 
                })
                
                # 2. Apply noise directly to the CLEAN scatter image
                if has_red:
                    noisy_scatter = self.generator.background_noise(clean_scatter, red_mask)
                    outputs.append({ 
                        "image": noisy_scatter, 
                        "label": current_grade,
                        "type": f"{label}n-{original_path}",
                        "folder": f"{folder}_noise" 
                    })

            # --- SEMANTIC ERASURE (Remove) ---
            # 1. Generate the CLEAN remove first
            clean_remove = self.generator.remove_impurity(original_image, subject_mask, green_mask)
            outputs.append({
                "image": clean_remove,
                "label": current_grade,
                "type": f"rmr-{original_path}",
                "folder": "remove_regular"
            })
            
            # 2. Apply noise directly to the CLEAN remove image
            if has_red:
                noisy_remove = self.generator.background_noise(clean_remove, red_mask)
                outputs.append({
                    "image": noisy_remove,
                    "label": current_grade,
                    "type": f"rmn-{original_path}",
                    "folder": "remove_noise"
                })
            
            # --- FEATURE AMPLIFICATION (CLAHE) ---
            # 1. Generate the CLEAN CLAHE first
            clean_clahe = self.generator.apply_clahe(original_image, green_mask)
            outputs.append({
                "image": clean_clahe,
                "label": current_grade,
                "type": f"chr-{original_path}",
                "folder": "clahe_regular"
            })
            
            # 2. Apply noise directly to the CLEAN CLAHE image
            if has_red:
                noisy_clahe = self.generator.background_noise(clean_clahe, red_mask)
                outputs.append({
                    "image": noisy_clahe,
                    "label": current_grade,
                    "type": f"chn-{original_path}",
                    "folder": "clahe_noise"
                })

        # Blue mask logic placeholder
        if np.any(is_blue):
            print("Blue brush detected (Logic not connected yet)")
            
        return outputs


    # def generate_counterexamples(self, original_image, original_path, current_grade, editor_data):
    #     outputs = []
        
    #     # ==========================================
    #     # 1. The Baseline (Item 1)
    #     # ==========================================
    #     outputs.append({ 
    #         "image": original_image, 
    #         "label": current_grade,
    #         "type": f"o-{original_path}",
    #         "folder": "baseline_clean"
    #     })
             
    #     # If no annotations exist, stop here (Prevents Grade 0 Duplicates)
    #     if editor_data is None or not editor_data["layers"]:
    #         return outputs
        
    #     # --- Extract Brush Masks ---
    #     drawing_layer = editor_data["layers"][0]
    #     alpha = drawing_layer[:, :, 3] 
    #     drawn_mask = alpha > 10

    #     drawing_rgb = drawing_layer[:, :, :3]
    #     hsv_drawing = cv2.cvtColor(drawing_rgb, cv2.COLOR_RGB2HSV)
    #     hue = hsv_drawing[:, :, 0] 

    #     # Segment colours based on user drawing
    #     is_red = ((hue < 20) | (hue > 160)) & drawn_mask
    #     is_green = (hue > 35) & (hue < 85) & drawn_mask
    #     is_blue = (hue > 85) & (hue < 140) & drawn_mask

    #     red_mask = is_red.astype(np.uint8) * 255
    #     green_mask = is_green.astype(np.uint8) * 255
        
    #     has_red = np.any(is_red)
    #     has_green = np.any(is_green)
        
    #     # We still need the subject mask for scatter boundaries and removal zones
    #     subject_mask = self.generator.get_mask(original_image)
        
    #     # ==========================================
    #     # FACTOR A: THE NOISY BASELINE (Item 2 - regnoise)
    #     # ==========================================
    #     noisy_base_img = None
    #     if has_red:
    #         noisy_base_img = self.generator.background_noise(original_image, red_mask)
    #         outputs.append({
    #             "image": noisy_base_img,
    #             "label": current_grade,
    #             "type": f"rn-{original_path}",
    #             "folder": "baseline_noisy" 
    #         })

    #     # ==========================================
    #     # FACTOR B: THE FOREGROUND EDITS (Items 3-12)
    #     # ==========================================
    #     if has_green:
            
    #         # --- SYNTHETIC INJECTION (Scatters) ---
    #         impurity_levels = [5, 15, 50]
    #         labels = ["s", "m", "l"]     
    #         folder_names = ["smallscatter", "mediumscatter", "largescatter"]
            
    #         for level, label, folder in zip(impurity_levels, labels, folder_names): 
    #             # Items 3, 4, 5 (Clean Scatter)
    #             clean_scatter = self.generator.destructive_scatter(original_image, subject_mask, green_mask, clones=level)
    #             outputs.append({ 
    #                 "image": clean_scatter, 
    #                 "label": current_grade,
    #                 "type": f"{label}c-{original_path}",
    #                 "folder": f"clean_{folder}" 
    #             })
                
    #             # Items 6, 7, 8 (Noisy Scatter - Only if Red Mask exists)
    #             if has_red:
    #                 noisy_scatter = self.generator.destructive_scatter(noisy_base_img, subject_mask, green_mask, clones=level)
    #                 outputs.append({ 
    #                     "image": noisy_scatter, 
    #                     "label": current_grade,
    #                     "type": f"{label}n-{original_path}",
    #                     "folder": f"noisy_{folder}" 
    #                 })

    #         # --- SEMANTIC ERASURE (Remove) ---
    #         # Item 9 (Clean Remove)
    #         clean_remove = self.generator.remove_impurity(original_image, subject_mask, green_mask)
    #         outputs.append({
    #             "image": clean_remove,
    #             "label": current_grade,
    #             "type": f"rmc-{original_path}",
    #             "folder": "clean_remove"
    #         })
            
    #         # Item 10 (Noisy Remove - Only if Red Mask exists)
    #         if has_red:
    #             noisy_remove = self.generator.remove_impurity(noisy_base_img, subject_mask, green_mask)
    #             outputs.append({
    #                 "image": noisy_remove,
    #                 "label": current_grade,
    #                 "type": f"rmn-{original_path}",
    #                 "folder": "noisy_remove"
    #             })
            
    #         # --- FEATURE AMPLIFICATION (CLAHE) ---
    #         # Item 11 (Clean CLAHE)
    #         clean_clahe = self.generator.apply_clahe(original_image, green_mask)
    #         outputs.append({
    #             "image": clean_clahe,
    #             "label": current_grade,
    #             "type": f"chc-{original_path}",
    #             "folder": "clean_clahe"
    #         })
            
    #         # Item 12 (Noisy CLAHE - Only if Red Mask exists)
    #         if has_red:
    #             noisy_clahe = self.generator.apply_clahe(noisy_base_img, green_mask)
    #             outputs.append({
    #                 "image": noisy_clahe,
    #                 "label": current_grade,
    #                 "type": f"chn-{original_path}",
    #                 "folder": "noisy_clahe"
    #             })

    #     # Blue mask logic placeholder
    #     if np.any(is_blue):
    #         print("Blue brush detected (Logic not connected yet)")
            
    #     return outputs

    def save_corrections(self, corrections_data, labels):
        saved_count = self.dataset_manager.save_corrections_to_hf(corrections_data, labels)
        return saved_count