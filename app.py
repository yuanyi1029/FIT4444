import gradio as gr
from inference import * 
from saliency import * 

yolo_model, torch_model = load_model("models/bests.pt")

def gradio_process(input_image): 
    if input_image is None: 
        return "Please upload an image."
    
    result = predict_image(yolo_model, input_image)
    saliency_map, _ = generate_saliency(torch_model, input_image)

    output_text = (
        f"Prediction: {result['class']}\n"
        f"Confidence: {result['confidence']:.2%}\n"
        f"Margin:     {result['margin']:.4f}"
    )
    
    return output_text, saliency_map, gr.update(value=saliency_map, visible=True)

with gr.Blocks(title="CAIPI") as demo:
    gr.Markdown("CAIPI Framework")

    with gr.Row():
        # Left Column 
        with gr.Column(scale=2):
            input_img = gr.Image(label="Input Image", type="numpy", height=300)
            submit_btn = gr.Button("Submit", variant="primary")
            clear_btn = gr.ClearButton(components=[input_img])

            image_editor = gr.ImageEditor(
                type="numpy", 
                label="Masking Tool", 
                brush=gr.Brush(colors=["#90EE9080", "#FF999980", "#87CEEB60"], default_size=15), 
                interactive=True,
                visible=False, 
                height=400
            )

        # Right Column 
        with gr.Column(scale=3):
            output_txt = gr.Textbox(label="Model Prediction")
            output_map = gr.Image(label="Grad-CAM Saliency Map", type="numpy", height=300)

    submit_btn.click(
        fn=gradio_process,
        inputs=[input_img],
        outputs=[output_txt, output_map, image_editor]
    )

if __name__ == "__main__": 
    demo.launch()