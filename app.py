import gradio as gr
from inference import * 
from saliency import * 
from generator import * 

yolo_model, torch_model = load_model("models/bests.pt")
generator = Generator()  
ITEMS = 4 

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

# Generate counterexamples process
def save_process():
    print("generate counterexamples process")
    gr.Info("Save button clicked")

# Generate counterexamples process
def save_process(current_data, *current_labels):
    print("generate counterexamples process")
    
    if not current_data:
        gr.Warning("No data to save.")
        return current_data

    # Iterate through the state and update the labels
    for i in range(len(current_data)):
        new_label = current_labels[i]
        
        if new_label is not None:
            old_label = current_data[i]["label"]
            current_data[i]["label"] = new_label
            
            # Print for verification
            if old_label != new_label:
                print(f"Item {i} ({current_data[i]['type']}): Grade updated from {old_label} -> {new_label}")
            else:
                print(f"Item {i} ({current_data[i]['type']}): Grade confirmed as {new_label}")

    gr.Info(f"State updated with {len(current_data)} verified items!")
    return current_data

with gr.Blocks(title="CAIPI") as demo:
    gr.Markdown("CAIPI Framework")

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

if __name__ == "__main__": 
    demo.launch()