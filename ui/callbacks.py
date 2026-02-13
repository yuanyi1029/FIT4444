from core.setup import * 
from core.finetune import *
from core.inference import * 
from config import *
from core.pipeline import HITLPipeline
import gradio as gr
import numpy as np

def create_callbacks(pipeline: HITLPipeline):

    def predict_process(input_image): 
        print("prediction process")
        if input_image is None: 
            return "Please upload an image."
        
        prediction_data = pipeline.process_prediction(input_image)
        result = prediction_data["result"]

        output_text = (
            f"Prediction: {result['class']}\n"
            f"Confidence: {result['confidence']:.2%}\n"
            f"Margin:     {result['margin']:.4f}"
        )

        return (
            input_image, 
            result["class"], 
            output_text, 
            prediction_data["saliency"], 
            prediction_data["segmentation"], 
            gr.update(visible=True), 
            gr.update(value=prediction_data["saliency"]), 
            gr.update(value=result["class"])
        )

    def generate_process(original_image, original_grade, editor_data, current_grade):
        print("generate counterexamples process")
        outputs = pipeline.generate_counterexamples(
            original_image,
            original_grade,
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

    def finetune_process(): 
        print("finetune process")
        prepare_dataset_retrain()
        new_model, train_metrics = finetune_model(pipeline.yolo_model, 2)
        test_metrics = test_model(new_model)
        new_accuracy = test_metrics.get('metrics/accuracy_top1', 0.0)

        pipeline.yolo_model = new_model

        return f"""
            # CAIPI Framework 
            ### Loaded Model: `{os.path.basename(MODEL_PATH)}_2` | Accuracy: **{new_accuracy}**
        """
    
    return { 
        "predict": predict_process,
        "generate": generate_process,
        "save": save_process,
        "finetune": finetune_process,
    }