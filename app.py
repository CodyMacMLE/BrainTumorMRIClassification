import gradio as gr
import torch
from pathlib import Path
import numpy as np
from PIL import Image

# Internal
from src.Model.Resnet.mri_resnet import build_resnet
from src.Model.Resnet.predict import predict as predict_resnet
from src.Model.Unet.Unet import UNetModel, predict as predict_unet
from src.Model.persistance import load_weights

"""
Loading Models with trained weights
"""
device = "cuda" if torch.cuda.is_available() else "cpu"

resnet_model = build_resnet(num_classes=2)
load_weights(resnet_model, Path("models/resnet18_2026-03-21_11h03-39s.pth"), device)
resnet_model.to(device)

unet_model = UNetModel(in_channels=3, out_channels=1)
load_weights(unet_model, Path("models/unet_2026-03-30_01h03-26s.pth"), device)
unet_model.to(device)

"""
UI Display
"""
with gr.Blocks() as demo:
    """
    Unet States
    """
    mri_state = gr.State()
    unet_mask_state = gr.State()

    gr.Markdown("# MRI Brain Tumor Classification & Segmentation")
    gr.Markdown("To run, upload a sliced, top down mri segment and click **submit**")
    gr.Markdown("---")
    with gr.Row():
        original_output = gr.Image(label="Original", sources=None)
        with gr.Column():
            pred_class_out = gr.Label(label="Predicted Class")
            pred_prob_out = gr.Label(label="Predicted Probability")
    with gr.Column():
        with gr.Row() as output_segmentation:
            overlay_output = gr.Image(label="Overlay", sources=None)
            pred_mask_out = gr.Image(label="Mask", sources=None)
        alpha_slider = gr.Slider(minimum=0, maximum=1, value=0.3, step=0.05, label="Overlay Alpha")

    gr.Markdown("---")
    mri_segment = gr.Image(label="Upload")
    submit_btn = gr.Button("Submit")

    @submit_btn.click(inputs=(mri_segment, alpha_slider), outputs=(original_output, pred_class_out, pred_prob_out, overlay_output, pred_mask_out, unet_mask_state, mri_state))
    def inference(mri_segment, alpha_slider):
        # Handle input
        mri_pil = Image.fromarray(mri_segment)
        mri_pil = mri_pil.resize((224, 224))  # Input size
        mri_array = np.array(mri_pil)


        # Handle Resnet
        resnet_class, resnet_confidence, cam = predict_resnet(resnet_model, mri_array, device)
        resnet_label = "Tumor Detected" if resnet_class == 1 else "No Tumor Detected"
        resnet_confidence = f"{resnet_confidence:.2f}%"

        # Handle Unet
        unet_pred_mask = predict_unet(unet_model, mri_segment, device)

        overlaid = mri_array.copy()
        overlaid[unet_pred_mask == 1] = [255, 0, 0]  # red where tumor
        blended = ((1 - alpha_slider) * mri_array + alpha_slider * overlaid).astype(np.uint8)

        return mri_array, resnet_label, resnet_confidence, blended, unet_pred_mask, unet_pred_mask, mri_array

    @alpha_slider.change(inputs=(alpha_slider, mri_state, unet_mask_state), outputs=overlay_output)
    def alpha_slider_change(alpha_slider, mri_state, unet_mask_state):
        if mri_state is not None:
            overlaid = mri_state.copy()
            overlaid[unet_mask_state == 1] = [255, 0, 0]  # red where tumor
            blended = ((1 - alpha_slider) * mri_state + alpha_slider * overlaid).astype(np.uint8)
            return blended
        return None


demo.launch()