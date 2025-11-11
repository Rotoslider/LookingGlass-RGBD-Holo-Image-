import gradio as gr
import torch
from PIL import Image
import numpy as np
from distillanydepth.modeling.archs.dam.dam import DepthAnything
from distillanydepth.utils.image_util import chw2hwc, colorize_depth_maps
from distillanydepth.midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import cv2
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from gradio_imageslider import ImageSlider
from gradio import File as GrFile
import tempfile
from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import Tuple
import sys
import time
from unittest.mock import MagicMock
sys.modules['spaces'] = MagicMock()   # pretend it exists

# Helper function to load model from Hugging Face
def load_model_by_name(arch_name, checkpoint_path, device):
    model = None
    if arch_name == 'depthanything': 
        model_weights = load_file(checkpoint_path)        
        model = DepthAnything(checkpoint_path=None).to(device)
        model.load_state_dict(model_weights)  

        model = model.to(device) 
    else:
        raise NotImplementedError(f"Unknown architecture: {arch_name}")
    return model

# Image processing function
def process_image(image, model, device):
    if model is None:
        return None, None, None, None
    
    # Preprocess the image
    image_np = np.array(image)[..., ::-1] / 255
    
    transform = Compose([
        Resize(756, 756, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet()
    ])
    
    image_tensor = transform({'image': image_np})['image']
    image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_disp, _ = model(image_tensor)
    torch.cuda.empty_cache()

    # Convert depth map to numpy
    pred_disp_np = pred_disp.cpu().detach().numpy()[0, 0, :, :]
    
    # Normalize depth map
    pred_disp_normalized = (pred_disp_np - pred_disp_np.min()) / (pred_disp_np.max() - pred_disp_np.min())
    
    # Colorized depth map
    cmap = "Spectral_r"
    depth_colored = colorize_depth_maps(pred_disp_normalized[None, ..., None], 0, 1, cmap=cmap).squeeze()
    depth_colored = (depth_colored * 255).astype(np.uint8)
    depth_colored_hwc = chw2hwc(depth_colored)
    
    # Gray depth map
    depth_gray = (pred_disp_normalized * 255).astype(np.uint8)
    depth_gray_hwc = np.stack([depth_gray] * 3, axis=-1)  # Convert to 3-channel grayscale
    
    # Save raw depth map as a temporary npy file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as temp_file:
        np.save(temp_file.name, pred_disp_normalized)
        depth_raw_path = temp_file.name
    
    # Resize outputs to match original image size
    h, w = image_np.shape[:2]
    depth_colored_hwc = cv2.resize(depth_colored_hwc, (w, h), cv2.INTER_LINEAR)
    depth_gray_hwc = cv2.resize(depth_gray_hwc, (w, h), cv2.INTER_LINEAR)
    
    # Convert to PIL images
    return image, Image.fromarray(depth_colored_hwc), Image.fromarray(depth_gray_hwc), depth_raw_path



# Gradio interface function with GPU support

def build_output_paths(original_name: str) -> Tuple[str, str, str]:
    """
    original_name : original uploaded file name (can be empty string for examples)
    returns       : depth_png, npy, rgbd_png  (absolute paths)
    """
    if not original_name:                       # cached examples – use temp dir
        tmp   = Path(tempfile.gettempdir())
        stem  = "example"
    else:                                       # real upload – use original folder
        p     = Path(original_name)
        tmp   = p.parent
        stem  = p.stem

    depth_png = str(tmp / f"{stem}_depth.png")
    npy_path  = str(tmp / f"{stem}.npy")
    rgbd_png  = str(tmp / f"{stem}_rgbd.png")
    return depth_png, npy_path, rgbd_png


# ----------  updated interface  ----------
def gradio_interface(image_path: str, _file_name: str = "") -> tuple:
    """
    image_path : str  – path to uploaded file (from Gradio type="filepath")
    _file_name : unused, kept for signature compatibility
    """
    timestamp = str(int(time.time() * 1000))          # simple unique token
    tmp_root  = Path(tempfile.gettempdir())

    depth_png_path = str(tmp_root / f"{timestamp}_depth.png")
    npy_path       = str(tmp_root / f"{timestamp}.npy")
    rgbd_png_path  = str(tmp_root / f"{timestamp}_rgbd.png")
    p = Path(image_path)          # original file path
    real_name = str(p)            # full path for naming outputs
    image = Image.open(image_path)  # open as PIL
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_kwargs = dict(
        vitb=dict(
            encoder='vitb', features=128, out_channels=[96, 192, 384, 768],
        ),
        vitl=dict(
            encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024],
            use_bn=False, use_clstoken=False, max_depth=150.0, mode='disparity',
            pretrain_type='dinov2', del_mask_token=False
        )
    )
    model = DepthAnything(**model_kwargs['vitl']).to(device)
    checkpoint_path = hf_hub_download(
        repo_id="xingyang1/Distill-Any-Depth",
        filename="large/model.safetensors",
        repo_type="model"
    )
    model_weights = load_file(checkpoint_path)
    model.load_state_dict(model_weights)
    model.to(device)

    # -----  depth inference (unchanged)  -----
    image_np = np.array(image)[..., ::-1] / 255.0
    transform = Compose([
        Resize(756, 756, resize_target=False, keep_aspect_ratio=True,
               ensure_multiple_of=14, resize_method='lower_bound',
               image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet()
    ])
    image_tensor = transform({'image': image_np})['image']
    image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_disp, _ = model(image_tensor)
    torch.cuda.empty_cache()

    pred_disp_np = pred_disp.cpu().detach().numpy()[0, 0, :, :]
    pred_disp_norm = (pred_disp_np - pred_disp_np.min()) / (pred_disp_np.max() - pred_disp_np.min())

    # coloured depth (for slider)
    depth_colored = colorize_depth_maps(pred_disp_norm[None, ..., None], 0, 1, cmap="Spectral_r").squeeze()
    depth_colored = (depth_colored * 255).astype(np.uint8)
    depth_colored_hwc = chw2hwc(depth_colored)
    h, w = image_np.shape[:2]
    depth_colored_hwc = cv2.resize(depth_colored_hwc, (w, h), cv2.INTER_LINEAR)

    # grey-scale depth (8-bit PNG)
    depth_gray = (pred_disp_norm * 255).astype(np.uint8)
    depth_gray_hwc = np.stack([depth_gray] * 3, axis=-1)  # 3-ch grey for PIL
    depth_gray_hwc = cv2.resize(depth_gray_hwc, (w, h), cv2.INTER_LINEAR)

    # -----  build output file paths  -----
    depth_png_path, npy_path, rgbd_png_path = build_output_paths(real_name)

    # save grey PNG
    grey_pil = Image.fromarray(depth_gray_hwc)
    grey_pil.save(depth_png_path, format="PNG")
    grey_pil.close()

    # save raw NumPy array
    np.save(npy_path, pred_disp_norm)

    # build side-by-side RGBD image
    rgbd = np.hstack([np.array(image), depth_gray_hwc])
    Image.fromarray(rgbd).save(rgbd_png_path, format="PNG")

    return (
        image,                                # left  slider
        Image.fromarray(depth_colored_hwc)    # right slider
    ), \
    GrFile(depth_png_path, label="Grey-depth PNG"), \
    GrFile(npy_path,        label="Raw depth (NumPy)"), \
    GrFile(rgbd_png_path,   label="RGBD side-by-side PNG")


# ----------  Gradio interface (updated)  ----------
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Image(type="filepath", label="Input image"),  # <-- gives path string
        gr.Textbox(value="", visible=False)
    ],
    outputs=[
        ImageSlider(label="Depth slider", type="pil", slider_color="pink"),
        gr.Image(type="pil", label="Grey-depth PNG"),
        gr.File(label="Raw depth (NumPy)"),
        gr.File(label="RGBD side-by-side PNG")
    ],
    title="RGBD Holo Image for Looking Glass Portrait",
    description="Upload an image to get depth maps and a combined RGBD PNG.",
    examples=[["1.jpg"], ["2.jpg"], ["4.png"], ["6.jpg"]],
    cache_examples=False,
    allow_flagging="never"
)

iface.launch()