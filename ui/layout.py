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
        generated_st = gr.State()
        
        # --- Active Learning States ---
        session_queue = gr.State([])      
        current_filepath = gr.State("")   
 
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
                    # next_img_btn = gr.Button("Skip to Next Image", variant="secondary", visible=False)
                
                queue_status = gr.Markdown("**Queue:** 0 images ready", visible=False)

        # ==========================================
        # ROW 2: Prediction & Visualizations (Saliency & Mask)
        # ==========================================
        gr.Markdown("## Model Predictions")
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

        # ==========================================
        # ROW 4: Counterexamples Placeholders
        # ==========================================
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
                    
        with gr.Row():
            save_btn = gr.Button("Save & Next", variant="primary", visible=False)

        # ==========================================
        # EVENT WIRING (Connecting UI to Logic)
        # ==========================================

        # Helper arrays for outputs
        output_gen = []
        for i in range(ITEMS):
            output_gen.append(output_cols[i])
            output_gen.append(output_imgs[i])
            output_gen.append(output_dds[i])
        
        output_gen.append(save_btn) 
        output_gen.append(generated_st) 
        
        # Predict outputs
        predict_outputs = [
            original_img, original_grd, output_txt, output_map, output_img,
            output_seg, editor_container, img_editor, grade_dd
        ]

        # 1. Active Learning Initialization
        # start_session_btn.click(
        #     fn=callbacks["session"], 
        #     inputs=[folder_uploader],
        #     outputs=[
        #         session_queue, 
        #         queue_status,   
        #         original_img, original_grd, output_txt, output_img, 
        #         output_map, output_seg, editor_container, img_editor, grade_dd
        #     ]
        # )

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
                original_img, original_grd, output_txt, output_img, 
                output_map, output_seg, editor_container, img_editor, grade_dd
            ]
        ).then(
            fn=lambda: gr.Button("Start Active Session", interactive=True),
            inputs=None,
            outputs=start_session_btn
        )

        # 2. Generate Counterexamples
        # gen_btn.click(
        #     fn=callbacks.get("generate"), 
        #     inputs=[original_img, original_grd, img_editor, grade_dd],
        #     outputs=output_gen
        # )
        gen_btn.click(
            fn=lambda: gr.Button("Generating...", interactive=False),
            inputs=None,
            outputs=gen_btn
        ).then(
            fn=callbacks.get("generate"), 
            inputs=[original_img, original_grd, img_editor, grade_dd],
            outputs=output_gen
        ).then(
            fn=lambda: gr.Button("Generate Counterexamples", interactive=True),
            inputs=None,
            outputs=gen_btn
        )

        # 3. Save & Automatically Load Next Image
        # save_btn.click(
        #     fn=callbacks.get("save"), 
        #     inputs=[generated_st] + output_dds,
        #     outputs=[generated_st]            
        # ).then( 
        #     fn=callbacks.get("next"), 
        #     inputs=[session_queue],
        #     outputs=[session_queue, queue_status] + predict_outputs + output_gen
        # )
        
        save_btn.click(
            fn=lambda: gr.Button("Saving...", interactive=False),
            inputs=None,
            outputs=save_btn, 
            queue=False
        ).then(
            fn=callbacks.get("save"), 
            inputs=[generated_st] + output_dds,
            outputs=[generated_st]            
        ).then( 
            fn=callbacks.get("next"), 
            inputs=[session_queue],
            outputs=[session_queue, queue_status] + predict_outputs + output_gen
        ).then(
            fn=lambda: gr.Button("Save & Next", interactive=True),
            inputs=None,
            outputs=save_btn, 
            queue=False
        )

        # 4. Finetune
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

        # 5. Clear Dataset 
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

    return demo