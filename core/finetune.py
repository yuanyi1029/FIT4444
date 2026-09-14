from core.inference import * 
from config import *
from ultralytics import YOLO
from pathlib import Path
import shutil

def finetune_model(model, version_number):
    BASE_DIR = Path.cwd()
    MODELS_DIR = BASE_DIR / "models"
    FINETUNED_DIR = MODELS_DIR / "finetuned"
    LOGS_DIR = MODELS_DIR / "training_logs"

    FINETUNED_DIR.mkdir(parents=True, exist_ok=True)
    
    temp_model_path = MODELS_DIR / "temp_trainer.pt"
    model.save(str(temp_model_path))
    model = YOLO(str(temp_model_path))
        
    results = model.train(
        data=str(BASE_DIR / "dataset_retrain"),
        epochs=15,              
        patience=FINETUNE_PATIENCE,             
        imgsz=640,
        seed=SEED,
        deterministic=True,
        lr0=0.0001,           
        lrf=0.01,             
        fliplr=0.3,            
        degrees=8.0,          
        hsv_h=0.015,          
        hsv_s=0.15,           
        hsv_v=0.15,            
        project=str(LOGS_DIR),      
        name=f'finetune_v{version_number}', 
        exist_ok=True,
        device=GPU_ID,
        verbose=True
    )

    training_output_path = LOGS_DIR / f'finetune_v{version_number}' / 'weights' / 'best.pt'
    versioned_model_path = FINETUNED_DIR / f"bests_v{version_number}.pt"
    
    if training_output_path.exists():
        shutil.copy2(training_output_path, versioned_model_path)

        current_model_path = MODELS_DIR / "current.pt"
        shutil.copy2(versioned_model_path, current_model_path)
    else:
        print(f"ERROR: Could not find trained weights at: {training_output_path}")

    if temp_model_path.exists():
        temp_model_path.unlink()

    return YOLO(str(versioned_model_path)), results

if __name__ == "__main__":
    pass
