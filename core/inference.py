from config import * 
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import numpy as np
import copy
import torch

def load_model(model_path):
    try:
        # Standard yolo model 
        model = YOLO(model_path)

        # Pytorch model (for GradCam) 
        torch_model = copy.deepcopy(model.model)
        device = torch.device(DEVICE)
        torch_model.to(device)
        
        # Unfreeze gradients (Crucial for Grad-CAM!)
        for param in torch_model.parameters():
            param.requires_grad = True
        torch_model.eval()

        print("Model loaded successfully")
        return model, torch_model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def predict_image(model, image_path):
    try:
        # Filename safety name 
        filename = "uploaded_image.jpg"
        
        # Gradio image path 
        if isinstance(image_path, np.ndarray):
            image_path = Image.fromarray(image_path)

        # Regular image path  
        elif isinstance(image_path, (str, Path)):
            filename = Path(image_path).name

        # Unsupported image  
        else:
            raise ValueError("Unsupported image source type")
        
        # Inference
        results = model(image_path, verbose=False)
        result = results[0]

        # Probabilities 
        probs = result.probs.data.tolist()
        sorted_probs = sorted(probs, reverse=True)
        
        # Margin  
        top1_conf = sorted_probs[0]
        top2_conf = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        margin = top1_conf - top2_conf

        pred_idx = result.probs.top1
        pred_class = result.names[pred_idx]

        return {
            "filename": filename,
            "filepath": image_path,
            "class": pred_class,
            "confidence": top1_conf,
            "margin": margin,
            "probs": probs
        }

    except Exception as e:
        print(f"Prediction failed: {e}")
        return None

def test_model(model): 
    dataset_path = DATASET_HITL_LABELED
    metrics = model.val(
        data=dataset_path, 
        split="test", 
        plots=False, 
        seed=SEED, 
        deterministic=True,
        device=GPU_ID,
        # device="cpu"
    )

    cm = metrics.confusion_matrix.matrix
    if not isinstance(cm, np.ndarray):
        cm = np.array(cm)
    
    # Calculate per-class metrics
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    eps = 1e-9
    
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    
    # NEW: Filter for classes that actually exist in the test ground truth
    ground_truth_counts = np.sum(cm, axis=1)
    present_classes = ground_truth_counts > 0
    
    # Calculate macro averages ONLY for the present classes
    macro_precision = np.mean(precision[present_classes])
    macro_recall = np.mean(recall[present_classes])
    macro_f1 = np.mean(f1[present_classes])
    
    # Return comprehensive results
    return {
        'accuracy': metrics.results_dict.get('metrics/accuracy_top1', 0.0),
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'confusion_matrix': cm.tolist(),
        'raw_metrics': metrics.results_dict
    }
    
if __name__ == "__main__":
    pass