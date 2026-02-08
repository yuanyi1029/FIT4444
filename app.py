import gradio as gr
from inference import * 
from saliency import * 
from generator import * 

yolo_model, torch_model = load_model("models/bests.pt")
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
    
    return input_image, predicted_grade, output_text, saliency_map, gr.update(visible=True), gr.update(value=saliency_map), gr.update(value=predicted_grade)

# Generate counterexamples process  
def generate_process(original_image, original_grade, editor_data, current_grade):
    print("generate counterexamples process")

    # Annotation Checking
    is_annotated = False
    if editor_data and editor_data["layers"]:
        drawing_layer = editor_data["layers"][0]
        alpha = drawing_layer[:, :, 3] 
        # If there are enough drawn pixels, consider it annotated
        if np.count_nonzero(alpha > 10) > 50:
            is_annotated = True

    # Grade Checking 
    grade_changed = False
    if original_grade is not None and current_grade is not None:
        grade_changed = (current_grade != original_grade)

    # Logic
    if grade_changed and is_annotated:
        gr.Info("Wrong for Wrong Reasons (Label + Explanation Correction)")
    
    elif not grade_changed and not is_annotated:
        gr.Info("Right for Right Reasons (Verified)")
        return gr.update(visible=False, value=[])
        
    elif not grade_changed and is_annotated:
        gr.Info("Right for Wrong Reasons (Explanation Correction)")

    # Proceed
    if editor_data is None or not editor_data["layers"]:
        return None
    
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
    
    outputs = []

    # Red 
    if np.any(is_red):
        noisy_base_img = generator.background_noise(original_image, red_mask)
    else:
        noisy_base_img = original_image.copy()

    # Green 
    if np.any(is_green):
        impurity_levels = [5, 15, 30]
        labels = ["Low", "Medium", "High"]
        for level, label in zip(impurity_levels, labels): 
            scatter_img = generator.destructive_scatter(noisy_base_img, green_mask, clones=level)
            outputs.append((scatter_img, label))
        
    # Blue
    if np.any(is_blue):
        print("Blue brush detected (Logic not connected yet)")
        
    if not outputs: 
        return gr.update(value=[], visible=False)

    # gr.Info(f"The grade is bruh {original_grade} but current is {current_grade}")
    return gr.update(value=outputs, visible=True)

with gr.Blocks(title="CAIPI") as demo:
    gr.Markdown("CAIPI Framework")

    original_grd = gr.State()
    original_img = gr.State()
    
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

                grd_dropdown = gr.Dropdown(
                    choices=["0", "1", "2"],
                    label="Verify Grade",
                    interactive=True
                )

                gen_btn = gr.Button("Generate Counterexamples", variant="stop") 

        # Right Column 
        with gr.Column(scale=3):
            output_txt = gr.Textbox(label="Model Prediction")
            output_map = gr.Image(label="Grad-CAM Saliency Map", type="numpy", height=300)
            output_gallery = gr.Gallery(
                label="Generated Counterexamples",
                show_label=True,
                columns=2,
                rows=2,
                object_fit="contain",
                height=400,
                visible=False
            )            

    submit_btn.click(
        fn=predict_process,
        inputs=[input_img],
        outputs=[original_img, original_grd, output_txt, output_map, editor_container, img_editor, grd_dropdown]
    )

    gen_btn.click(
        fn=generate_process, 
        inputs=[original_img, original_grd, img_editor, grd_dropdown],
        outputs=[output_gallery]
    )

if __name__ == "__main__": 
    demo.launch()