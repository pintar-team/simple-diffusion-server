import os
import io
import base64
import logging
from typing import Optional, Tuple, List, Dict, Any

import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from diffusers import (
    UNet2DConditionModel,
    StableDiffusionXLPipeline,
    StableDiffusionXLInpaintPipeline,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL
)

from utils import parse_args, Args, is_local_file, process_image_data, compose_images

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if running with Gunicorn
is_gunicorn = "gunicorn" in os.environ.get("SERVER_SOFTWARE", "")

# Set device (CPU or CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load configuration
if not is_gunicorn:
    args = parse_args()
else:
    args = Args(
        model=os.getenv('MODEL_NAME', 'stabilityai/stable-diffusion-xl-base-1.0'),
        unet=os.getenv('UNET_MODEL', ''),
        lora_dirs=os.getenv('LORA_DIRS', ''),
        lora_scales=os.getenv('LORA_SCALES', ''),
        scheduler=os.getenv('SCHEDULER', 'euler_a'),
        host=os.getenv('HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', 8001)),
        vae=os.getenv('VAE_MODEL', 'madebyollin/sdxl-vae-fp16-fix')
    )

def load_models() -> StableDiffusionXLInpaintPipeline:
    """
    Load and configure the Stable Diffusion XL models.
    """
    logger.info("Loading models...")
    
    if args.vae == '' or args.vae == 'baked':
        vae = None
    else:
        vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=torch.bfloat16, variant="fp16")

    if args.unet:
        unet = UNet2DConditionModel.from_pretrained(args.unet, torch_dtype=torch.bfloat16, variant="fp16")
    else:
        unet = None

    if is_local_file(args.model):
        pipe = StableDiffusionXLInpaintPipeline.from_single_file(
            args.model, vae=vae, unet=unet, torch_dtype=torch.bfloat16, variant="fp16",
            use_safetensors=True, num_in_channels=4, ignore_mismatched_sizes=True
        )
    else:
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            args.model, vae=vae, unet=unet, torch_dtype=torch.bfloat16, variant="fp16"
        )

    load_and_fuse_lora(pipe)
    set_scheduler(pipe)

    pipe.to(device)
    # Uncomment the following lines if you want to enable these optimizations
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()

    logger.info("Models loaded successfully")
    return pipe

def load_and_fuse_lora(pipe: StableDiffusionXLInpaintPipeline) -> None:
    """
    Load and fuse LoRA weights to the pipeline.
    """
    lora_dirs = args.lora_dirs.split(':') if args.lora_dirs else []
    lora_scales = [float(scale) for scale in args.lora_scales.split(':')] if args.lora_scales else []

    if len(lora_dirs) != len(lora_scales):
        raise ValueError("The number of LoRA directories must match the number of scales")

    for ldir, lsc in zip(lora_dirs, lora_scales):
        pipe.load_lora_weights(ldir)
        pipe.fuse_lora(lora_scale=lsc)

def set_scheduler(pipe: StableDiffusionXLInpaintPipeline) -> None:
    """
    Set the appropriate scheduler for the pipeline.
    """
    if args.scheduler == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler == "euler_a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Load models
pipe = load_models()

# Initialize Flask app
app = Flask(__name__)

@app.route('/generate-image', methods=['POST'])
def generate_image():
    try:
        data = request.json
        image_params = parse_image_params(data)
        
        init_image, init_mask = create_init_image_and_mask(image_params['width'], image_params['height'])
        
        generated_image = generate_image_with_pipe(pipe, image_params, init_image, init_mask)
        
        return jsonify({"image": encode_image(generated_image, image_params['format'])})

    except Exception as e:
        logger.exception("Error generating image")
        return jsonify({"error": str(e)}), 500

@app.route('/generate-img2img', methods=['POST'])
def generate_img2img():
    try:
        data = request.json
        image_params = parse_image_params(data)
        
        composite_image, composite_mask = process_input_images(data, image_params)
        
        generated_image = generate_image_with_pipe(pipe, image_params, composite_image, composite_mask)
        
        if image_params['extract_mask'] and composite_mask is not None:
            generated_image = extract_masked_content(generated_image, composite_mask, image_params['extract_color'])
        
        return jsonify({"image": encode_image(generated_image, image_params['format'])})

    except Exception as e:
        logger.exception("Error generating img2img")
        return jsonify({"error": str(e)}), 500

def parse_image_params(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse and validate image generation parameters from request data.
    """
    params = {
        "prompt": data["prompt"],
        "negative_prompt": data.get("negative_prompt"),
        "num_inference_steps": data.get("num_inference_steps", 30),
        "guidance_scale": data.get("guidance_scale", 7.5),
        "seed": data.get("seed"),
        "format": data.get("format", "jpeg").lower(),
        "width": ((data.get("width", 1024) + 7) // 8) * 8,
        "height": ((data.get("height", 1024) + 7) // 8) * 8,
        "strength": data.get("strength", 0.8),
        "extract_mask": data.get("extract_mask", False),
        "apply_mask": data.get("apply_mask", True),
        "extract_color": parse_extract_color(data.get("extract_color", (0, 0, 0, 0))),
    }
    
    if params["format"] not in ["jpeg", "png"]:
        raise ValueError("Invalid image format. Choose 'jpeg' or 'png'.")
    
    return params

def parse_extract_color(extract_color: Any) -> Tuple[int, int, int, int]:
    """
    Parse the extract_color parameter into a tuple of integers.
    """
    if isinstance(extract_color, list):
        return tuple(extract_color)
    elif isinstance(extract_color, str):
        return tuple(map(int, extract_color.split(",")))
    elif isinstance(extract_color, tuple):
        return extract_color
    else:
        return (0, 0, 0, 0)  # Default to transparent black if invalid format

def create_init_image_and_mask(width: int, height: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create initial image and mask tensors for image generation.
    """
    init_image = Image.new("RGB", (width, height))
    init_image_tensor = torch.from_numpy(np.array(init_image)).float() / 255.0
    init_image_tensor = init_image_tensor.permute(2, 0, 1).unsqueeze(0).half().to(device)

    white_mask = Image.new("L", (width, height), 255)
    mask_tensor = torch.from_numpy(np.array(white_mask)).float() / 255.0
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).half().to(device)

    return init_image_tensor, mask_tensor

def process_input_images(data: Dict[str, Any], image_params: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Process input images and masks for img2img generation.
    """
    images = process_image_data(data.get("images", []))
    masks = process_image_data(data.get("masks", [])) if data.get("masks") else None

    composite_image = compose_images(images, image_params['width'], image_params['height'])
    composite_mask = compose_images(masks, image_params['width'], image_params['height']) if masks else None

    composite_image_tensor = image_to_tensor(composite_image)
    composite_mask_tensor = mask_to_tensor(composite_mask) if composite_mask else create_white_mask_tensor(image_params['width'], image_params['height'])

    return composite_image_tensor, composite_mask_tensor

def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL Image to a PyTorch tensor.
    """
    tensor = torch.from_numpy(np.array(image)).float() / 255.0
    return tensor.permute(2, 0, 1).unsqueeze(0).half().to(device)

def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    """
    Convert a PIL mask image to a PyTorch tensor.
    """
    tensor = torch.from_numpy(np.array(mask)).float() / 255.0
    return tensor.unsqueeze(0).unsqueeze(0).half().to(device)

def create_white_mask_tensor(width: int, height: int) -> torch.Tensor:
    """
    Create a white mask tensor.
    """
    white_mask = Image.new("L", (width, height), 255)
    return mask_to_tensor(white_mask)

def generate_image_with_pipe(
    pipe: StableDiffusionXLInpaintPipeline,
    params: Dict[str, Any],
    image: torch.Tensor,
    mask: torch.Tensor
) -> Image.Image:
    """
    Generate an image using the Stable Diffusion XL pipeline.
    """
    generator = torch.manual_seed(params['seed']) if params['seed'] is not None else None

    generated_image = pipe(
        prompt=params['prompt'],
        negative_prompt=params['negative_prompt'],
        image=image,
        mask_image=mask,
        height=params['height'],
        width=params['width'],
        strength=params['strength'],
        num_inference_steps=params['num_inference_steps'],
        guidance_scale=params['guidance_scale'],
        generator=generator
    ).images[0]

    return crop_image(generated_image, params['width'], params['height'])

def crop_image(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """
    Crop the generated image to the target dimensions.
    """
    width, height = image.size
    if (width != target_width) or (height != target_height):
        left = (width - target_width) // 2
        top = (height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        return image.crop((left, top, right, bottom))
    return image

def extract_masked_content(
    generated_image: Image.Image,
    mask: torch.Tensor,
    extract_color: Tuple[int, int, int, int]
) -> Image.Image:
    """
    Extract the generated content using the mask.
    """
    mask_image = Image.fromarray((mask.squeeze().cpu().numpy() * 255).astype(np.uint8), mode='L')
    return Image.composite(generated_image.convert("RGBA"), Image.new("RGBA", generated_image.size, extract_color), mask_image)

def encode_image(image: Image.Image, format: str) -> str:
    """
    Encode the image to a base64 data URI.
    """
    buffer = io.BytesIO()
    if format == "jpeg":
        image = image.convert("RGB")
    image.save(buffer, format=format)
    mime_type = f"image/{format}"
    return f"data:{mime_type};base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"

if __name__ == '__main__':
    app.run(host=args.host, port=args.port)