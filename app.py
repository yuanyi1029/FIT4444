from core.pipeline import *
from ui.callbacks import * 
from ui.layout import * 

yolo_model, torch_model = load_model(MODEL_PATH)
pipeline = HITLPipeline(yolo_model, torch_model)
callbacks = create_callbacks(pipeline)
demo = create_layout(callbacks)

def main():
    print("Launching Gradio interface...")
    demo.launch()

if __name__ == "__main__":
    main()