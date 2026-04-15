import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from ultralytics.data.augment import classify_transforms

class YOLOWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        res = self.model(x)
        return res[0] if isinstance(res, (list, tuple)) else res

def generate_saliency(torch_model, image_path, target_class_idx=None):
    try:
        # Gradio image path (nd.array)
        if isinstance(image_path, np.ndarray):
            img_pil = Image.fromarray(image_path)
            img_rgb = image_path

        # Regular image path 
        elif isinstance(image_path, str):
            img_cv2 = cv2.imread(image_path)
            if img_cv2 is None:
                raise ValueError(f"Could not read image file: {image_path}")
            
            img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

        # Unsupported image  
        else:
            raise ValueError("Unsupported image source type")

        # Preprocessing
        device = next(torch_model.parameters()).device
        transform = classify_transforms(size=640)
        input_tensor = transform(img_pil).unsqueeze(0).to(device)

        # GradCAM setup 
        wrapped_model = YOLOWrapper(torch_model)
        target_layers = [torch_model.model[-2]]
        cam = GradCAM(model=wrapped_model, target_layers=target_layers)

        # Generate Heatmap 
        targets = [ClassifierOutputTarget(target_class_idx)] if target_class_idx is not None else None
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        # Prepare Background 
        img_resized = cv2.resize(img_rgb, (640, 640))
        rgb_background = np.float32(img_resized) / 255
        visualization = show_cam_on_image(rgb_background, grayscale_cam, use_rgb=True, image_weight=0.8)

        return visualization, grayscale_cam

    except Exception as e:
        print(f"Error generating saliency: {e}")
        return None, None

if __name__ == "__main__":
    pass