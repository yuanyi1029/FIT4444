from config import * 
import gradio as gr
import os

def create_layout(callbacks): 
    with gr.Blocks(title="CAIPI") as demo:
        # --- Header ---
        with gr.Row(): 
            with gr.Column(scale=5): 
                header_md = gr.Markdown(
                    f"""
                    # CAIPI Framework 
                    ### Loaded Model: `{os.path.basename(MODEL_PATH)}`
                    Accuracy: **0.7639**<br>
                    Precision: **0.7654**<br>
                    Recall: **0.7649**<br>
                    F1 Score: **0.7635**<br> 
                    """
                )

            with gr.Column(scale=1): 
                with gr.Row(): 
                    finetune_btn = gr.Button("Finetune Model", variant="primary")
                    clear_btn = gr.Button("Clear Dataset", variant="stop")

        # --- Global States ---
        original_grd = gr.State()
        original_img = gr.State()
        generated_st = gr.State([]) # Initialize as empty list
        
        # --- Active Learning States ---
        session_queue = gr.State([])      
        original_pth = gr.State("")   
 
        # ==========================================
        # ROW 1: Files Upload and Session Control
        # ==========================================
        with gr.Row():
            with gr.Column():
                folder_uploader = gr.File(
                    file_count="multiple", 
                    label="Upload Unlabeled Directory", 
                    file_types=["image"],
                    height=200
                )
                
                with gr.Row():
                    start_session_btn = gr.Button("Start Active Session", variant="primary")
                
                queue_status = gr.Markdown("**Queue:** 0 images ready", visible=False)

        # ==========================================
        # ROW 2: Prediction & Visualizations 
        # ==========================================
        # gr.Markdown("## Model Predictions")
        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("## Model Predictions")
            with gr.Column(scale=1):
                skip_btn = gr.Button("Skip Image", variant="secondary")

        with gr.Row(): 
            with gr.Column(scale=1):
                output_img = gr.Image(label="Grad-CAM Saliency Map", type="numpy", height=300)
            with gr.Column(scale=1):
                output_map = gr.Image(label="Grad-CAM Saliency Map", type="numpy", height=300)
            with gr.Column(scale=1):
                output_seg = gr.Image(label="Segmentation Mask", type="numpy", height=300)

        with gr.Row():
            output_txt = gr.Textbox(label="Model Prediction", lines=4)
            
        # ==========================================
        # ROW 3: Image Annotation (Correction Tools)
        # ==========================================
        with gr.Row():
            with gr.Column(visible=False) as editor_container: 
                gr.Markdown("## Annotation Tools")
                img_editor = gr.ImageEditor(
                    type="numpy", 
                    label="Masking Tool", 
                    brush=gr.Brush(colors=BRUSH_COLORS, default_size=BRUSH_DEFAULT_SIZE), 
                    interactive=True,
                    height=400
                )
                grade_dd = gr.Dropdown(
                    choices=["0", "1", "2"],
                    label="Verify Grade",
                    interactive=True
                )
                gen_btn = gr.Button("Generate Counterexamples", variant="stop") 

        # We group these outputs now so they can be referenced inside the render block
        predict_outputs = [
            original_img, original_grd, original_pth, output_txt, output_img, 
            output_map, output_seg, editor_container, img_editor, grade_dd
        ]

        # ==========================================
        # ROW 4: Dynamic Counterexamples via Rendering
        # ==========================================
        @gr.render(inputs=generated_st)
        def render_counterexamples(data):
            if not data:
                return # Renders nothing (hiding the row) if state is empty

            with gr.Row(): 
                for i, item in enumerate(data): 
                    with gr.Column(): 
                        img = gr.Image(value=item["image"], label=f"Image {i}", interactive=False, height=200)
                        dd = gr.Dropdown(
                            choices=["0", "1", "2"], 
                            value=item["label"], 
                            label=f"Grade ({item['type']})", 
                            interactive=True
                        )
                        
                        # When a dropdown changes, we update the specific item in the state
                        def update_grade(new_grade, index=i):
                            new_data = list(data) # List copy forces state change recognition
                            new_data[index]["label"] = new_grade
                            return new_data

                        dd.change(fn=update_grade, inputs=[dd], outputs=[generated_st])
                        
            with gr.Row():
                save_btn = gr.Button("Save & Next", variant="primary")
                
                # The save button is dynamically bound whenever the UI rebuilds
                save_btn.click(
                    fn=lambda: gr.Button("Saving...", interactive=False),
                    inputs=None,
                    outputs=save_btn, 
                    queue=False
                ).then(
                    fn=callbacks.get("save"), 
                    inputs=[generated_st],
                    outputs=[generated_st]            
                ).then( 
                    fn=callbacks.get("next"), 
                    inputs=[session_queue],
                    outputs=[session_queue, queue_status] + predict_outputs + [generated_st]
                )

        # ==========================================
        # EVENT WIRING (Connecting static UI to Logic)
        # ==========================================

        # 1. Active Learning Initialization
        start_session_btn.click(
            fn=lambda: gr.Button("Starting Session...", interactive=False),
            inputs=None,
            outputs=start_session_btn
        ).then(
            fn=callbacks["session"], 
            inputs=[folder_uploader],
            outputs=[
                session_queue, 
                queue_status,   
                *predict_outputs 
            ]
        ).then(
            fn=lambda: gr.Button("Start Active Session", interactive=True),
            inputs=None,
            outputs=start_session_btn
        )

        # 2. Generate Counterexamples
        gen_btn.click(
            fn=lambda: gr.Button("Generating...", interactive=False),
            inputs=None,
            outputs=gen_btn
        ).then(
            fn=callbacks.get("generate"), 
            inputs=[original_img, original_grd, original_pth, img_editor, grade_dd],
            outputs=[generated_st]
        ).then(
            fn=lambda: gr.Button("Generate Counterexamples", interactive=True),
            inputs=None,
            outputs=gen_btn
        )

        # 3. Finetune
        finetune_btn.click(
            fn=lambda: gr.Button("Finetuning...", interactive=False), 
            inputs=None,
            outputs=finetune_btn 
        ).then(
            fn=callbacks.get("finetune"),
            inputs=None,
            outputs=header_md
        ).then(
            fn=lambda: gr.Button("Finetune Model", interactive=True), 
            inputs=None,
            outputs=finetune_btn
        )

        # 4. Clear Dataset 
        clear_btn.click(
            fn=lambda: gr.Button("Clearing...", interactive=False),
            inputs=None,
            outputs=clear_btn
        ).then(
            fn=callbacks.get("clear"),
            inputs=None,
            outputs=None
        ).then(
            fn=lambda: gr.Button("Clear Dataset", interactive=True),
            inputs=None,
            outputs=clear_btn
        )

        skip_btn.click(
            fn=lambda: gr.Button("Skipping...", interactive=False),
            inputs=None,
            outputs=skip_btn,
            queue=False
        ).then(
            fn=callbacks.get("next"), 
            inputs=[session_queue],
            outputs=[session_queue, queue_status] + predict_outputs + [generated_st]
        ).then(
            fn=lambda: gr.Button("Skip Image", interactive=True),
            inputs=None,
            outputs=skip_btn,
            queue=False
        )

    return demo