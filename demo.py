import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load the custom trained YOLO model
# Assuming the best model weights are located in the checkpoints directory
try:
    model = YOLO("checkpoints/best.pt")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def predict_image(img):
    if model is None:
        return img  # Return original image if model failed to load

    # The input img is a PIL Image
    # Perform inference
    results = model(img)
    
    # Results is a list, we take the first item
    res = results[0]
    
    # Plot the results on the image
    # res.plot() returns a numpy array with BGR format
    annotated_img_bgr = res.plot()
    
    # Convert BGR back to RGB for Gradio to display correctly
    # res.plot() gives us a numpy array in BGR (which OpenCV uses), we reverse the channels for RGB
    annotated_img_rgb = annotated_img_bgr[..., ::-1]
    
    return annotated_img_rgb

# Build the Gradio interface
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Image(type="numpy", label="Annotated Output"),
    title="Custom YOLO Model Demo",
    description="Upload an image to perform object detection using our custom trained YOLO model.",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
