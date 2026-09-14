from config import *
from pathlib import Path
import shutil
import os

def display_dataset_distribution(root="dataset_hitl", classes=['0', '1', '2']):
    """
    Summarizes the distribution of images across labeled and unlabeled directories.
    """
    print(f"\n{'='*60}")
    print(f"DATASET DISTRIBUTION SUMMARY: {root}")
    print(f"{'='*60}\n")

    def get_counts(path, is_flat=False):
        counts = {cls: 0 for cls in classes}
        total = 0
        if not os.path.exists(path):
            return counts, 0

        if is_flat:
            # Parse filenames like (0)_image.jpg
            for f in os.listdir(path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    for cls in classes:
                        if f.startswith(f"({cls})"):
                            counts[cls] += 1
                            total += 1
        else:
            # Count images in subdirectories /0, /1, /2
            for cls in classes:
                subdir = os.path.join(path, cls)
                if os.path.exists(subdir):
                    img_count = len([f for f in os.listdir(subdir) 
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))])
                    counts[cls] = img_count
                    total += img_count
        return counts, total

    def print_row(title, counts, total):
        details = ", ".join([f"{k}: {v:>3}" for k, v in counts.items()])
        print(f"{title:<10} | {details}  (Total: {total})")

    # --- Process Labeled ---
    print("LABELED:")
    print("-" * 60)
    for split in ["train", "val", "test"]:
        counts, total = get_counts(os.path.join(root, "labeled", split))
        print_row(split.capitalize(), counts, total)

    # --- Process Unlabeled ---
    print("\nUNLABELED POOL:")
    print("-" * 60)
    u_counts, u_total = get_counts(os.path.join(root, "unlabeled"), is_flat=True)
    print_row("Pool Size", u_counts, u_total)
    
    print(f"\n{'='*60}\n")

def prepare_dataset_generated():
    print("Preparing dataset_generated")

    root = DATASET_GENERATED
    sub_dirs = ["corrected", "counterexamples"]

    for sub in sub_dirs:
        for i in range(3):
            target_path = root / sub / str(i)
            
            target_path.mkdir(parents=True, exist_ok=True)
            
            for item in target_path.iterdir():
                try:
                    if item.is_file() or item.is_symlink():
                        item.unlink() 
                    elif item.is_dir():
                        shutil.rmtree(item) 
                except Exception as e:
                    print(f"Failed to delete {item}. Reason: {e}")
            
    print("dataset_generated preparation complete.")

def prepare_dataset_retrain():
    print("Preparing dataset_retrain")
    
    src_hitl = DATASET_HITL_LABELED
    src_generated = DATASET_GENERATED
    dest_retrain = DATASET_RETRAIN
    
    if dest_retrain.exists():
        shutil.rmtree(dest_retrain)
    dest_retrain.mkdir(parents=True)

    for split in ["train", "val", "test"]:
        src_split = src_hitl / split
        if src_split.exists():
            shutil.copytree(src_split, dest_retrain / split)
        else:
            for i in range(3):
                (dest_retrain / split / str(i)).mkdir(parents=True, exist_ok=True)

    gen_mapping = {
        "corrected": "0c", 
        "counterexamples": "0e", 
    }
    
    for folder_name, prefix in gen_mapping.items():
        for i in range(3):
            src_folder = src_generated / folder_name / str(i)
            dest_folder = dest_retrain / "train" / str(i)
            
            if src_folder.exists():
                for img_path in src_folder.iterdir():
                    if img_path.is_file():
                        new_name = f"{prefix}_{img_path.name}"
                        shutil.copy2(img_path, dest_folder / new_name)

    print("dataset_retrain preparation complete")

def count_dataset(folder_path):
    root = Path(folder_path)
    splits = ["train", "val", "test"]
    classes = ["0", "1", "2"]
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    print(f"\n{root.name.upper()}:")
    print("-" * 60)

    for split in splits:
        counts = {}
        split_total = 0
        
        for cls in classes:
            subdir = root / split / cls
            count = len([f for f in subdir.iterdir() if f.suffix.lower() in extensions]) if subdir.exists() else 0
            counts[cls] = count
            split_total += count
        
        details = ", ".join([f"{k}: {v:>3}" for k, v in counts.items()])
        print(f"{split.capitalize():<10} | {details}  (Total: {split_total})")

if __name__ == "__main__": 
    # count_dataset("dataset_hitl/labeled")
    # count_dataset("dataset_retrain")
    display_dataset_distribution()
    # prepare_dataset_retrain()
    # prepare_dataset_generated() 