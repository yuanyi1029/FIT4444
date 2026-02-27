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
        saved_count = self.dataset_manager.save_corrections_to_hf(corrections_data, labels)
        return saved_count