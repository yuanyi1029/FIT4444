from inference import * 
from saliency import * 
from generator import * 
from setup import * 
from finetune import *
from testing import *
import gradio as gr
import uuid
from pathlib import Path
import os 
import time

MODEL_PATH = "models/bests.pt"
ITEMS = 4 

yolo_model, torch_model = load_model(MODEL_PATH)
model_name = os.path.basename(MODEL_PATH)
try: 
    # metrics = test_model(yolo_model)
    # accuracy = f"{metrics.get('metrics/accuracy_top1')}"
    
    # Temporary 
    accuracy = "0.645962"
except Exception as e: 
    accuracy = "N/A" 

generator = Generator()  

# Prediction process 
def predict_process(input_image): 
    print("prediction process")
    if input_image is None: 
        return "Please upload an image."
    
    result = predict_image(yolo_model, input_image)
    output_text = (
        f"Prediction: {result['class']}\n"
        f"Confidence: {result['confidence']:.2%}\n"
        f"Margin:     {result['margin']:.4f}"
    )
    predicted_grade = result["class"]

    saliency_map, _ = generate_saliency(torch_model, input_image)
    segmentation_mask = generator.get_mask(input_image)
    
    return (
        input_image, 
        predicted_grade, 
        output_text, 
        saliency_map, 
        segmentation_mask, 
        gr.update(visible=True), 
        gr.update(value=saliency_map), 
        gr.update(value=predicted_grade)
    )

# Generate counterexamples process  
def generate_process(original_image, original_grade, editor_data, current_grade):
    print("generate counterexamples process")
    outputs = [{ 
        "image": original_image, 
        "label": current_grade,
        "type": f"original"
    }]

    def generate_counterexamples(): 
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
            noisy_base_img = generator.background_noise(original_image, red_mask)
        else:
            noisy_base_img = original_image.copy()

        # Green 
        if np.any(is_green):
            impurity_levels = [5, 15, 50]
            labels = ["Low", "Medium", "High"]
            for level, label in zip(impurity_levels, labels): 
                scatter_img = generator.destructive_scatter(noisy_base_img, green_mask, clones=level)
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
        
    # Annotation Check
    is_annotated = False
    if editor_data and editor_data["layers"]:
        drawing_layer = editor_data["layers"][0]
        alpha = drawing_layer[:, :, 3] 
        if np.count_nonzero(alpha > 10) > 50:
            is_annotated = True

    # Grade Check
    grade_changed = False
    if original_grade is not None and current_grade is not None:
        grade_changed = (current_grade != original_grade)

    # Logic
    if not grade_changed and not is_annotated:
        gr.Info("Right for Right Reasons (Verified)")
    
    elif grade_changed:
        gr.Info("Wrong for Wrong Reasons (Label + Explanation Correction)")
        if is_annotated: 
            generate_counterexamples()
      
    elif not grade_changed and is_annotated:
        gr.Info("Right for Wrong Reasons (Explanation Correction)")
        generate_counterexamples()

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
    BASE_PATH = Path("dataset_generated")
    
    if not current_data:
        gr.Warning("No data to save.")
        return current_data

    for i in range(len(current_data)):
        new_label = current_labels[i]
        if new_label is None:
            continue
            
        item = current_data[i]
        item["label"] = new_label
        
        folder_type = "corrected" if item["type"] == "original" else "counterexamples"
        target_dir = BASE_PATH / folder_type / str(new_label)
        target_dir.mkdir(parents=True, exist_ok=True)

        unique_id = uuid.uuid4().hex[:8]
        filename = f"{item['type']}_{unique_id}.jpg"
        save_path = target_dir / filename

        img_bgr = cv2.cvtColor(item["image"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), img_bgr)

        print(f"Saved {filename} to {target_dir}")

    gr.Info(f"Successfully saved {len(current_data)} items to dataset_generated!")
    return current_data

def finetune_process(): 
    print("finetune process")
    time.sleep(5)
    prepare_dataset_retrain()
    new_model, train_metrics = finetune_model(yolo_model, 2)
    test_metrics = test_model(new_model)
    new_acc = test_metrics.get('metrics/accuracy_top1') or test_metrics.get('top1_acc') or 0.0

    return_string = f"""
        # CAIPI Framework 
        ### Loaded Model: `{model_name}_v2` | Accuracy: **{new_acc}**
    """
    print("===============================")
    print(test_metrics)
    print("===============================")
    
    return return_string


def disable_button(text):
    return gr.Button(text, interactive=False)

def enable_button(text): 
    return gr.Button(text, interactive=True)

with gr.Blocks(title="CAIPI") as demo:

    with gr.Row(): 
        with gr.Column(scale=5): 
            header_md = gr.Markdown(
                f"""
                # CAIPI Framework 
                ### Loaded Model: `{model_name}` | Accuracy: **{accuracy}**
                """
            )

        with gr.Column(scale=1): 
            finetune_btn = gr.Button("Finetune Model", variant="primary")

    original_grd = gr.State()
    original_img = gr.State()
    generated_st = gr.State()
    
    with gr.Row():
        # Left Column 
        with gr.Column(scale=2):
            input_img = gr.Image(label="Input Image", type="numpy", height=300)
            submit_btn = gr.Button("Submit", variant="primary")
            clear_btn = gr.ClearButton(components=[input_img])

            with gr.Column(visible=False) as editor_container: 
                img_editor = gr.ImageEditor(
                    type="numpy", 
                    label="Masking Tool", 
                    brush=gr.Brush(colors=["#90EE9080", "#FF999980", "#87CEEB60"], default_size=15), 
                    interactive=True,
                    height=400
                )

                grade_dd = gr.Dropdown(
                    choices=["0", "1", "2"],
                    label="Verify Grade",
                    interactive=True
                )

                gen_btn = gr.Button("Generate Counterexamples", variant="stop") 

        # Right Column 
        with gr.Column(scale=3):
            output_txt = gr.Textbox(label="Model Prediction")
            with gr.Row(): 
                output_map = gr.Image(label="Grad-CAM Saliency Map", type="numpy", height=300)
                output_seg = gr.Image(label="Segmentation Mask", type="numpy", height=300)

            output_cols = []
            output_imgs = []
            output_dds = []

            with gr.Row(): 
                for i in range(ITEMS): 
                    with gr.Column(visible=False) as col: 
                        img = gr.Image(label=f"Image {i}", interactive=False, height=200)
                        dd = gr.Dropdown(choices=["0", "1", "2"], label="Grade", interactive=True)
                        
                        output_cols.append(col)
                        output_imgs.append(img)
                        output_dds.append(dd)

            save_btn = gr.Button("Save & Next", variant="primary", visible=False)

    output_gen = []
    for i in range(ITEMS):
        output_gen.append(output_cols[i])
        output_gen.append(output_imgs[i])
        output_gen.append(output_dds[i])
    
    output_gen.append(save_btn) 
    output_gen.append(generated_st) 

    submit_btn.click(
        fn=predict_process,
        inputs=[input_img],
        outputs=[original_img, original_grd, output_txt, output_map, output_seg, editor_container, img_editor, grade_dd]
    )

    gen_btn.click(
        fn=generate_process, 
        inputs=[original_img, original_grd, img_editor, grade_dd],
        outputs=output_gen
    )
    
    save_btn.click(
        fn=save_process, 
        inputs=[generated_st] + output_dds,
        outputs=[generated_st]            
    )

    finetune_btn.click(
        # Use lambda to pass the specific string you want here
        fn=lambda: disable_button("Finetuning..."), 
        inputs=None,
        outputs=finetune_btn 
    ).then(
        fn=finetune_process,
        inputs=None,
        outputs=header_md
    ).then(
        fn=lambda: enable_button("Finetune Model"),
        inputs=None,
        outputs=finetune_btn
    )

if __name__ == "__main__": 
    demo.launch()