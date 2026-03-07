from core.setup import * 
from core.finetune import *
from core.inference import * 
from config import *
from core.pipeline import HITLPipeline
import gradio as gr
import numpy as np
import cv2 

def create_callbacks(pipeline: HITLPipeline):

    def session_process(input_files):
        if not input_files:
            return gr.update(), gr.update(), *[gr.update()]*10 
            
        sorted_queue = pipeline.init_active_session(input_files)
        
        if not sorted_queue:
            return gr.update(), gr.update(), *[gr.update()]*10

        first_item = sorted_queue.pop(0)
        remaining = len(sorted_queue)
        
        ui_updates = predict_process(first_item["path"])
        
        return (
            sorted_queue,                                       
            gr.update(value=f"**Queue:** {remaining} images left", visible=True),
            *ui_updates                                            
        )
    
    def predict_process(input_image): 
        print("prediction process")
        if input_image is None: 
            return tuple([gr.update()]*10)
        
        clean_name = "unknown"
        if isinstance(input_image, str):
            img_bgr = cv2.imread(input_image)
            state_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            clean_name = "-".join(Path(input_image).name.split("-")[:2])
        else:   
            state_image = input_image

        prediction_data = pipeline.process_prediction(input_image)
        result = prediction_data["result"]

        output_text = (
            f"Filename: {result['filename']}\n"
            f"Prediction: {result['class']}\n"
            f"Confidence: {result['confidence']:.2%}\n"
            f"Margin:     {result['margin']:.4f}"
        )

        return (
            state_image, 
            result["class"], 
            clean_name, 
            output_text, 
            input_image,
            prediction_data["saliency"], 
            prediction_data["segmentation"], 
            gr.update(visible=True), 
            gr.update(value=prediction_data["saliency"]), 
            gr.update(value=result["class"]),
        )

    def generate_process(original_image, original_grade, original_path, editor_data, current_grade):
        print("generate counterexamples process")
        outputs = pipeline.generate_counterexamples(
            original_image,
            original_path, 
            current_grade,
            editor_data 
        )
        
        # Annotation Check
        is_annotated = False
        grade_changed = (original_grade != current_grade) if original_grade else False

        if editor_data and editor_data["layers"]:
            alpha = editor_data["layers"][0][:, :, 3]
            is_annotated = np.count_nonzero(alpha > 10) > 50

        # Logic
        if not grade_changed and not is_annotated:
            gr.Info("Right for Right Reasons (Verified)")
        elif grade_changed:
            gr.Info("Wrong for Wrong Reasons (Label + Explanation Correction)")        
        elif not grade_changed and is_annotated:
            gr.Info("Right for Wrong Reasons (Explanation Correction)")

        ui_updates = []

        for i in range(ITEMS):
            if i < len(outputs):
                item = outputs[i]
                ui_updates.append(gr.update(visible=True))       
                ui_updates.append(gr.update(value=item["image"])) 
                is_interactive = (i > 0)
                ui_updates.append(gr.update(value=item["label"], interactive=is_interactive, label=f"Grade ({item['type']})")) 
            else:
                ui_updates.append(gr.update(visible=False))
                ui_updates.append(gr.update(value=None))
                ui_updates.append(gr.update(value=None))

        ui_updates.append(gr.update(visible=True))
        return ui_updates + [outputs]

    def save_process(current_data, *current_labels):
        print("save process")        
        if not current_data:
            gr.Warning("No data to save.")
            return current_data

        saved_count = pipeline.save_corrections(current_data, current_labels)
        gr.Info(f"Successfully saved {saved_count} items to dataset_generated!")
        
        return current_data

    def next_process(current_queue): 
        print("next process")

        if not current_queue:
            gr.Info("Session Complete")
            
            predict_empties = [None, None, None, None, None, None, gr.update(visible=False), None, None]
            row4_resets = [gr.update(visible=False), None, None] * ITEMS + [gr.update(visible=False), None]
            
            return (
                current_queue, 
                gr.update(value="**Queue:** 0 images left"), 
                *predict_empties, 
                *row4_resets
            )

        next_item = current_queue.pop(0)
        remaining = len(current_queue)

        ui_updates = predict_process(next_item["path"])

        row4_resets = []
        for _ in range(ITEMS):
            row4_resets.append(gr.update(visible=False)) 
            row4_resets.append(gr.update(value=None))   
            row4_resets.append(gr.update(value=None))  
        row4_resets.append(gr.update(visible=False))  
        row4_resets.append(gr.update(value=None))    
        
        return (
            current_queue,
            gr.update(value=f"**Queue:** {remaining} images left"),
            *ui_updates,
            *row4_resets
        )

    def finetune_process(): 
        print("finetune process")
        dataset_path = pipeline.dataset_manager.prepare_dataset_retrain()  
        
        new_model, train_metrics = finetune_model(pipeline.yolo_model, 2)
        test_metrics = test_model(new_model)
        new_accuracy = test_metrics.get("accuracy")
        new_precision = test_metrics.get("macro_precision")
        new_recall = test_metrics.get("macro_recall")
        new_f1 = test_metrics.get("macro_f1")

        pipeline.yolo_model = new_model

        print("finetuning complete")

        return f"""
            # CAIPI Framework 
            ### Loaded Model: `{os.path.basename(MODEL_PATH)}_2` 
            Accuracy: **{new_accuracy}**<br>
            Precision: **{new_precision}**<br>
            Recall: **{new_recall}** <br>
            F1 Score: **{new_f1}** <br> 
        """
    
    def clear_process():
        print("clear process")
        deleted_count = pipeline.dataset_manager.clear_dataset()
        
        if deleted_count > 0:
            gr.Info(f"Successfully cleared {deleted_count} files from dataset_generated!")
        else:
            gr.Info("Dataset was already empty or clear failed.")
        
        return deleted_count
    
    return { 
        "session": session_process,
        "predict": predict_process,
        "generate": generate_process,
        "save": save_process,
        "next": next_process,
        "finetune": finetune_process,
        "clear": clear_process
    }