from huggingface_hub import HfApi
from pathlib import Path
from dotenv import load_dotenv
import os
import shutil
import cv2
import uuid
from config import *
import tempfile

load_dotenv()

class DatasetManager:
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        self.generated_repo = os.getenv("HF_DATASET_GENERATED")
        self.api = HfApi(token=self.hf_token)
        # Local temporary folder for building uploads
        self.temp_upload_dir = Path("./temp_upload_generated")
        
        self.counter_file = Path("session_counter.txt")
        self.count = self._load_count()
    
    def _load_count(self):
        """Reads the count from disk. Starts at 1 if no file exists."""
        if self.counter_file.exists():
            try:
                with open(self.counter_file, "r") as f:
                    return int(f.read().strip())
            except ValueError:
                return 1 
        return 1

    def _save_count(self):
        """Saves the current count to disk."""
        self.counter_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.counter_file, "w") as f:
            f.write(str(self.count))

    def _init_upload_structure(self):
        """Create temporary folder structure for upload"""
        if self.temp_upload_dir.exists():
            shutil.rmtree(self.temp_upload_dir)
        
        for folder in ["corrected", "counterexamples"]:
            for label in ["0", "1", "2"]:
                path = self.temp_upload_dir / folder / label
                path.mkdir(parents=True, exist_ok=True)
    
    def _download_existing(self):
        """Download existing dataset from HF to temp folder"""
        try:
            self.api.snapshot_download(
                repo_id=self.generated_repo,
                repo_type="dataset",
                local_dir=str(self.temp_upload_dir),
                token=self.hf_token
            )
            print(f"✓ Downloaded existing corrections from {self.generated_repo}")
        except Exception as e:
            print(f"⚠ No existing dataset found (first upload): {e}")
            self._init_upload_structure()

    # def save_corrections_to_hf(self, corrections_data, labels):
    #     print("="*60)
    #     print("SAVING NEW CORRECTIONS LOCALLY")
    #     print("="*60)
        
    #     saved_count = 0
    #     local_base_dir = Path("./dataset_generated")
        
    #     # Add ONLY the new corrections to the local folder
    #     for i, item in enumerate(corrections_data):
    #         label = labels[i]
    #         if label is None:
    #             continue
            
    #         # Determine folder based on type
    #         if item["type"][0] == "o":
    #             folder_type = "corrected"
    #         else:
    #             folder_type = "counterexamples"
            
    #         # Create target directory
    #         target_dir = local_base_dir / folder_type / str(label)
    #         target_dir.mkdir(parents=True, exist_ok=True)
            
    #         # Generate unique filename
    #         unique_id = uuid.uuid4().hex[:4]
    #         filename = f"{self.count}-{item['type']}-{unique_id}.jpg"
    #         save_path = target_dir / filename
            
    #         # Save image
    #         img_bgr = cv2.cvtColor(item["image"], cv2.COLOR_RGB2BGR)
    #         cv2.imwrite(str(save_path), img_bgr)
            
    #         print(f"✓ Saved: {folder_type}/{label}/{filename}")
    #         saved_count += 1
        
    #     if saved_count > 0:
    #         print(f"\n✓ Successfully saved {saved_count} corrections to {local_base_dir}")
            
    #     print("="*60)

    #     self.count += 1  
    #     self._save_count()
    #     return saved_count

    def save_corrections_to_hf(self, corrections_data, labels):
        print("="*60)   
        print("SAVING NEW CORRECTIONS LOCALLY")
        print("="*60)
        
        saved_count = 0
        local_base_dir = Path("./dataset_generated")
        
        for i, item in enumerate(corrections_data):
            label = labels[i]
            if label is None:
                continue
            
            folder_type = item.get("folder", "uncategorized") 
            target_dir = local_base_dir / folder_type / str(label)
            target_dir.mkdir(parents=True, exist_ok=True) 
            
            # Generate unique filename
            unique_id = uuid.uuid4().hex[:4]
            filename = f"{self.count}-{item['type']}-{unique_id}.jpg"
            save_path = target_dir / filename
            
            # Save image
            img_bgr = cv2.cvtColor(item["image"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_path), img_bgr)
            
            print(f"✓ Saved: {folder_type}/{label}/{filename}")
            saved_count += 1
        
        if saved_count > 0:
            print(f"\n✓ Successfully saved {saved_count} corrections to {local_base_dir}")
            
        print("="*60)

        self.count += 1  
        self._save_count()
        return saved_count

    # def save_corrections_to_hf(self, corrections_data, labels):
    #     """
    #     Save ONLY new corrections to HF dataset_generated using OS Temp directories.
    #     """
    #     print("="*60)
    #     print("SAVING NEW CORRECTIONS TO HF")
    #     print("="*60)
        
    #     saved_count = 0
        
    #     # Create a secure temporary directory managed by the OS
    #     with tempfile.TemporaryDirectory() as temp_dir:
    #         temp_upload_path = Path(temp_dir)
            
    #         # Add ONLY the new corrections to the temp folder
    #         for i, item in enumerate(corrections_data):
    #             label = labels[i]
    #             if label is None:
    #                 continue
                
    #             # Determine folder based on type
    #             if item["type"] == "o":
    #                 folder_type = "corrected"
    #             else:
    #                 folder_type = "counterexamples"
                
    #             # Create target directory
    #             target_dir = temp_upload_path / folder_type / str(label)
    #             target_dir.mkdir(parents=True, exist_ok=True)
                
    #             # Generate unique filename
    #             unique_id = uuid.uuid4().hex[:8]
    #             filename = f"{item['type']}_{unique_id}.jpg"
    #             save_path = target_dir / filename
                
    #             # Save image
    #             img_bgr = cv2.cvtColor(item["image"], cv2.COLOR_RGB2BGR)
    #             cv2.imwrite(str(save_path), img_bgr)
                
    #             print(f"✓ Staged: {folder_type}/{label}/{filename}")
    #             saved_count += 1
            
    #         # Upload ONLY the new files to HF
    #         if saved_count > 0:
    #             print(f"\nUploading {saved_count} new corrections to HF...")
    #             try:
    #                 self.api.upload_folder(
    #                     folder_path=str(temp_upload_path),
    #                     repo_id=self.generated_repo,
    #                     repo_type="dataset",
    #                     token=self.hf_token,
    #                     commit_message=f"Add {saved_count} new corrections via CAIPI"
    #                 )
    #                 print(f"✓ Successfully pushed to {self.generated_repo}")
    #             except Exception as e:
    #                 print(f"✗ Upload failed: {e}")
    #                 return 0
                
    #         # Note: No 'finally' block needed! 
    #         # When the 'with' block ends, Python safely deletes the temp folder automatically.

    #     print("="*60)
    #     return saved_count
    
    def prepare_dataset_retrain(self):
        """
        Build dataset_retrain locally by combining:
        1. dataset_hitl (already local in Space repo)
        2. dataset_generated (download from HF)
        
        Returns: Path to dataset_retrain
        """
        print("\n" + "="*60)
        print("BUILDING DATASET_RETRAIN")
        print("="*60)
        
        # Clean up old retrain folder
        if DATASET_RETRAIN.exists():
            shutil.rmtree(DATASET_RETRAIN)
        DATASET_RETRAIN.mkdir(parents=True)
        
        # Step 1: Copy dataset_hitl/labeled to dataset_retrain
        print("\nStep 1: Copying dataset_hitl (local)...")
        for split in ["train", "val", "test"]:
            src_split = DATASET_HITL_LABELED / split
            dest_split = DATASET_RETRAIN / split
            
            if src_split.exists():
                shutil.copytree(src_split, dest_split)
                
                # Count images
                count = 0
                for label in ["0", "1", "2"]:
                    count += len(list((dest_split / label).glob("*.jpg")))
                print(f"  {split}: {count} images")
            else:
                # Create empty structure
                for i in range(3):
                    (dest_split / str(i)).mkdir(parents=True, exist_ok=True)
                print(f"  {split}: 0 images (empty)")
        
        # Step 2: Download dataset_generated from HF
        print("\nStep 2: Downloading dataset_generated from HF...")
        temp_download = Path("./temp_download_generated")
        
        try:
            self.api.snapshot_download(
                repo_id=self.generated_repo,
                repo_type="dataset",
                local_dir=str(temp_download),
                token=self.hf_token
            )
            print(f"✓ Downloaded from {self.generated_repo}")
            
            # Step 3: Merge into dataset_retrain/train
            corrections_merged = 0
            
            # Process corrected folder
            corrected_path = temp_download / "corrected"
            if corrected_path.exists():
                for label in ["0", "1", "2"]:
                    src_dir = corrected_path / label
                    if src_dir.exists():
                        dest_dir = DATASET_RETRAIN / "train" / label
                        for img_file in src_dir.glob("*.jpg"):
                            # Copy with prefix to distinguish
                            new_name = f"hf_corrected_{img_file.name}"
                            shutil.copy2(img_file, dest_dir / new_name)
                            corrections_merged += 1
            
            # Process counterexamples folder
            counterex_path = temp_download / "counterexamples"
            if counterex_path.exists():
                for label in ["0", "1", "2"]:
                    src_dir = counterex_path / label
                    if src_dir.exists():
                        dest_dir = DATASET_RETRAIN / "train" / label
                        for img_file in src_dir.glob("*.jpg"):
                            # Copy with prefix to distinguish
                            new_name = f"hf_synthetic_{img_file.name}"
                            shutil.copy2(img_file, dest_dir / new_name)
                            corrections_merged += 1
            
            print(f"✓ Merged {corrections_merged} HF corrections into train split")
            
            # Cleanup temp download
            shutil.rmtree(temp_download)
            
        except Exception as e:
            print(f"⚠ No HF corrections found or error: {e}")
            print("✓ Using only dataset_hitl for retraining")
        
        # Display final stats
        self._display_retrain_stats()
        
        print("="*60 + "\n")
        return DATASET_RETRAIN
    
    def _display_retrain_stats(self):
        """Show what's in dataset_retrain"""
        print("\nFINAL DATASET_RETRAIN STATISTICS:")
        print("-" * 60)
        
        grand_total = 0
        for split in ["train", "val", "test"]:
            split_path = DATASET_RETRAIN / split
            if not split_path.exists():
                continue
                
            counts = {}
            split_total = 0
            for cls in ["0", "1", "2"]:
                cls_path = split_path / cls
                if cls_path.exists():
                    count = len(list(cls_path.glob("*.jpg")))
                    counts[cls] = count
                    split_total += count
            
            details = ", ".join([f"{k}: {v:>3}" for k, v in counts.items()])
            print(f"{split.capitalize():<10} | {details}  (Total: {split_total})")
            grand_total += split_total
        
        print("-" * 60)
        print(f"{'GRAND TOTAL':<10} | {grand_total} images")

    def clear_dataset(self):
        """
        Clear all images from HF dataset_generated while preserving
        .gitattributes and README.md
        
        Returns: Number of files deleted
        """
        print("\n" + "="*60)
        print("CLEARING DATASET_GENERATED ON HF")
        print("="*60)
        
        try:
            # Get all files in the repo
            files = self.api.list_repo_files(
                repo_id=self.generated_repo, 
                repo_type="dataset",
                token=self.hf_token
            )
            
            # Filter: keep .gitattributes and README.md, delete everything else
            files_to_keep = {".gitattributes", "README.md", ".gitignore"}
            files_to_delete = [
                f for f in files 
                if f not in files_to_keep
            ]
            
            if not files_to_delete:
                print("✓ Dataset is already empty!")
                return 0
            
            print(f"Found {len(files_to_delete)} files to delete")
            
            deleted_count = 0
            for file_path in files_to_delete:
                try:
                    self.api.delete_file(
                        path_in_repo=file_path,
                        repo_id=self.generated_repo,
                        repo_type="dataset",
                        token=self.hf_token,
                        commit_message=f"Clear dataset: delete {file_path}"
                    )
                    print(f"✓ Deleted: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"✗ Failed to delete {file_path}: {e}")
            
            print(f"\n✓ Successfully cleared {deleted_count} files from {self.generated_repo}")
            print("✓ Preserved: .gitattributes, README.md")
            print("="*60 + "\n")
            
            return deleted_count
            
        except Exception as e:
            print(f"✗ Error clearing dataset: {e}")
            print("="*60 + "\n")
            return 0
