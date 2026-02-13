
# def create_layout(callbacks): 
#     with gr.Blocks(title="CAIPI") as demo:
#         with gr.Row(): 
#             with gr.Column(scale=5): 
#                 header_md = gr.Markdown(
#                     f"""
#                     # CAIPI Framework 
#                     ### Loaded Model: `{os.path.basename(MODEL_PATH)}` | Accuracy: **0.6459**
#                     """
#                 )

#             with gr.Column(scale=1): 
#                 finetune_btn = gr.Button("Finetune Model", variant="primary")

#         original_grd = gr.State()
#         original_img = gr.State()
#         generated_st = gr.State()
        
#         with gr.Row():
#             # Left Column 
#             with gr.Column(scale=2):
#                 input_img = gr.Image(label="Input Image", type="numpy", height=300)
#                 submit_btn = gr.Button("Submit", variant="primary")
#                 clear_btn = gr.ClearButton(components=[input_img])

#                 with gr.Column(visible=False) as editor_container: 
#                     img_editor = gr.ImageEditor(
#                         type="numpy", 
#                         label="Masking Tool", 
#                         brush=gr.Brush(colors=BRUSH_COLORS, default_size=BRUSH_DEFAULT_SIZE), 
#                         interactive=True,
#                         height=400
#                     )

#                     grade_dd = gr.Dropdown(
#                         choices=["0", "1", "2"],
#                         label="Verify Grade",
#                         interactive=True
#                     )

#                     gen_btn = gr.Button("Generate Counterexamples", variant="stop") 

#             # Right Column 
#             with gr.Column(scale=3):
#                 output_txt = gr.Textbox(label="Model Prediction")
#                 with gr.Row(): 
#                     output_map = gr.Image(label="Grad-CAM Saliency Map", type="numpy", height=300)
#                     output_seg = gr.Image(label="Segmentation Mask", type="numpy", height=300)

#                 output_cols = []
#                 output_imgs = []
#                 output_dds = []

#                 with gr.Row(): 
#                     for i in range(ITEMS): 
#                         with gr.Column(visible=False) as col: 
#                             img = gr.Image(label=f"Image {i}", interactive=False, height=200)
#                             dd = gr.Dropdown(choices=["0", "1", "2"], label="Grade", interactive=True)
                            
#                             output_cols.append(col)
#                             output_imgs.append(img)
#                             output_dds.append(dd)

#                 save_btn = gr.Button("Save & Next", variant="primary", visible=False)

#         output_gen = []
#         for i in range(ITEMS):
#             output_gen.append(output_cols[i])
#             output_gen.append(output_imgs[i])
#             output_gen.append(output_dds[i])
        
#         output_gen.append(save_btn) 
#         output_gen.append(generated_st) 

#         submit_btn.click(
#             fn=callbacks["predict"],
#             inputs=[input_img],
#             outputs=[original_img, original_grd, output_txt, output_map, output_seg, editor_container, img_editor, grade_dd]
#         )

#         gen_btn.click(
#             fn=callbacks["generate"], 
#             inputs=[original_img, original_grd, img_editor, grade_dd],
#             outputs=output_gen
#         )
        
#         save_btn.click(
#             fn=callbacks["save"], 
#             inputs=[generated_st] + output_dds,
#             outputs=[generated_st]            
#         )

#         finetune_btn.click(
#             fn=lambda: gr.Button("Finetuning...", interactive=False), 
#             inputs=None,
#             outputs=finetune_btn 
#         ).then(
#             fn=callbacks["finetune"],
#             inputs=None,
#             outputs=header_md
#         ).then(
#             fn=lambda: gr.Button("Finetune Model", interactive=False), 
#             inputs=None,
#             outputs=finetune_btn
#         )

#     return demo