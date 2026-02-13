from core.inference import * 
from core.saliency import * 
from core.generator import * 
from config import * 
import uuid

class HITLPipeline: 
    
    def __init__(self, yolo_model, torch_model):
        self.yolo_model = yolo_model
        self.torch_model = torch_model
        self.generator = Generator()
    
    def process_prediction(self, image):
        result = predict_image(self.yolo_model, image)
        saliency_map, _ = generate_saliency(self.torch_model, image)
        segmentation_mask = self.generator.get_mask(image)
        
        return {
            'result': result,
            'saliency': saliency_map,
            'segmentation': segmentation_mask
        }
    
    def generate_counterexamples(self, original_image, current_grade, editor_data):
        outputs = [{ 
            "image": original_image, 
            "label": current_grade,
            "type": f"original"
        }]
             
        if editor_data is None or not editor_data["layers"]:
            return 
        
        # Extract mask  
        drawing_layer = editor_data["layers"][0]
        alpha = drawing_layer[:, :, 3] 
        drawn_mask = alpha > 10

        # Get hue channel  
        drawing_rgb = drawing_layer[:, :, :3]
        hsv_drawing = cv2.cvtColor(drawing_rgb, cv2.COLOR_RGB2HSV)
        hue = hsv_drawing[:, :, 0] 

        # Segment colours
        is_red = ((hue < 20) | (hue > 160)) & drawn_mask
        is_green = (hue > 35) & (hue < 85) & drawn_mask
        is_blue = (hue > 85) & (hue < 140) & drawn_mask

        red_mask = is_red.astype(np.uint8) * 255
        green_mask = is_green.astype(np.uint8) * 255
        blue_mask = is_blue.astype(np.uint8) * 255
        
        # Red 
        if np.any(is_red):
            noisy_base_img = self.generator.background_noise(original_image, red_mask)
        else:
            noisy_base_img = original_image.copy()

        # Green 
        if np.any(is_green):
            impurity_levels = [5, 15, 50]
            labels = ["Low", "Medium", "High"]
            for level, label in zip(impurity_levels, labels): 
                scatter_img = self.generator.destructive_scatter(noisy_base_img, green_mask, clones=level)
                outputs.append({ 
                    "image": scatter_img, 
                    "label": current_grade,
                    "type": f"synthetic_{label}"
                })

        elif np.any(is_red):
            outputs.append({ 
                "image": noisy_base_img, 
                "label": current_grade,
                "type": "synthetic_noise"
            })
             
        # Blue
        if np.any(is_blue):
            print("Blue brush detected (Logic not connected yet)")
        
        return outputs 

    def save_corrections(self, corrections_data, labels):
        saved_count = 0
        
        for i, item in enumerate(corrections_data):
            label = labels[i]
            if label is None:
                continue
            
            item["label"] = label
            
            folder_type = "corrected" if item["type"] == "original" else "counterexamples"
            target_dir = DATASET_GENERATED / folder_type / str(label)
            target_dir.mkdir(parents=True, exist_ok=True)
            
            unique_id = uuid.uuid4().hex[:8]
            filename = f"{item['type']}_{unique_id}.jpg"
            save_path = target_dir / filename
            
            img_bgr = cv2.cvtColor(item["image"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_path), img_bgr)
            
            saved_count += 1
        
        return saved_count