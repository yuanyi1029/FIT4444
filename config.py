from pathlib import Path
import torch

# Paths
MODEL_PATH = "models/best_final.pt"
DATASET_GENERATED = Path("dataset_generated")
DATASET_RETRAIN = Path("dataset_retrain")
DATASET_HITL = Path("dataset_hitl")
DATASET_HITL_LABELED = Path("dataset_hitl/labeled")

# UI
ITEMS = 4
BRUSH_COLORS = ["#90EE9080", "#FF999980", "#87CEEB60"]
BRUSH_DEFAULT_SIZE = 15

# Training 
SEED = 42
FINETUNE_EPOCHS = 15
FINETUNE_PATIENCE = 5

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GPU_ID = 0 if torch.cuda.is_available() else 'cpu'