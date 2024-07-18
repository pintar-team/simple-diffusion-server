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

from utils import parse_args, Args, is_local_file

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
        vae=os.getenv('VAE_MODEL', '')
    )

def load_models() -> StableDiffusionXLInpaintPipeline:
    """
    Load and configure the Stable Diffusion XL models.
    """
    logger.info("Loading models...")
    
    pipeline_args = {
        "torch_dtype": torch.bfloat16,
        "variant": "fp16",
        "use_safetensors": True,
        "num_in_channels": 4,
        "ignore_mismatched_sizes": True
    }

    if args.vae and args.vae != 'baked':
        vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=torch.bfloat16, variant="fp16")
        pipeline_args["vae"] = vae

    if args.unet:
        unet = UNet2DConditionModel.from_pretrained(args.unet, torch_dtype=torch.bfloat16, variant="fp16")
        pipeline_args["unet"] = unet

    if is_local_file(args.model):
        pipe = StableDiffusionXLInpaintPipeline.from_single_file(args.model, **pipeline_args)
    else:
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(args.model, **pipeline_args)

    load_and_fuse_lora(pipe)
    set_scheduler(pipe)

    pipe.to(device)
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
        
        generation_args = {
            "prompt": image_params['prompt'],
            "negative_prompt": image_params['negative_prompt'],
            "image": init_image,
            "mask_image": init_mask,
            "height": image_params['height'],
            "width": image_params['width'],
            "strength": 1.0,  # For text-to-image, we use full strength
            "num_inference_steps": image_params['num_inference_steps'],
            "guidance_scale": image_params['guidance_scale'],
            "seed": image_params['seed']
        }
        
        generated_image = generate_image_with_pipe(pipe, **generation_args)
        
        return jsonify({"image": encode_image(generated_image, image_params['format'])})

    except Exception as e:
        logger.exception("Error generating image")
        return jsonify({"error": str(e)}), 500

@app.route('/generate-img2img', methods=['POST'])
def generate_img2img():
    try:
        data = request.json
        image_params = parse_image_params(data)
        
        images_data = data.get("images", [])
        masks_data = data.get("masks", [])

        images = process_image_data(images_data)
        masks = process_image_data(masks_data) if masks_data else None

        composite_image, composite_mask = compose_images(images, masks, image_params)
        
        generation_args = {
            "prompt": image_params['prompt'],
            "negative_prompt": image_params['negative_prompt'],
            "image": composite_image,
            "height": image_params['height'],
            "width": image_params['width'],
            "strength": image_params['strength'],
            "num_inference_steps": image_params['num_inference_steps'],
            "guidance_scale": image_params['guidance_scale'],
            "seed": image_params['seed']
        }
        
        if image_params['apply_mask'] and composite_mask is not None:
            # Ensure the mask is a PIL Image
            if isinstance(composite_mask, np.ndarray):
                composite_mask = Image.fromarray(composite_mask.astype(np.uint8))
            elif isinstance(composite_mask, torch.Tensor):
                composite_mask = Image.fromarray(composite_mask.cpu().numpy().astype(np.uint8))
            generation_args["mask_image"] = composite_mask
        
        generated_image = generate_image_with_pipe(pipe, **generation_args)
        
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
    params = {}

    # Required parameters
    if "prompt" not in data:
        raise ValueError("Prompt is required")
    params["prompt"] = data["prompt"][:300]  # Trim to 300 symbols

    # Optional parameters with default values
    params["negative_prompt"] = data.get("negative_prompt", "")[:300]  # Trim to 300 symbols
    params["num_inference_steps"] = int(data.get("num_inference_steps", 30))
    params["guidance_scale"] = float(data.get("guidance_scale", 7.5))
    params["seed"] = int(data.get("seed")) if data.get("seed") is not None else None
    params["format"] = data.get("format", "jpeg").lower()
    params["width"] = ((int(data.get("width", 1024)) + 7) // 8) * 8
    params["height"] = ((int(data.get("height", 1024)) + 7) // 8) * 8
    params["strength"] = float(data.get("strength", 0.8))
    params["extract_mask"] = bool(data.get("extract_mask", False))
    params["apply_mask"] = bool(data.get("apply_mask", True))
    params["extract_color"] = parse_extract_color(data.get("extract_color", (0, 0, 0, 0)))

    # Validate parameters
    if params["format"] not in ["jpeg", "png"]:
        raise ValueError("Invalid image format. Choose 'jpeg' or 'png'.")
    
    if params["num_inference_steps"] < 1 or params["num_inference_steps"] > 150:
        raise ValueError("num_inference_steps must be between 1 and 150")
    
    if params["guidance_scale"] < 0 or params["guidance_scale"] > 20:
        raise ValueError("guidance_scale must be between 0 and 20")
    
    if params["width"] < 128 or params["width"] > 2048:
        raise ValueError("Width must be between 128 and 2048")
    
    if params["height"] < 128 or params["height"] > 2048:
        raise ValueError("Height must be between 128 and 2048")
    
    if params["strength"] < 0 or params["strength"] > 1:
        raise ValueError("Strength must be between 0 and 1")

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

def create_init_image_and_mask(width: int, height: int) -> Tuple[Image.Image, Image.Image]:
    """
    Create initial image and mask for image generation.
    """
    init_image = Image.new("RGB", (width, height))
    white_mask = Image.new("L", (width, height), 255)
    return init_image, white_mask

def process_image_data(image_data):
    if isinstance(image_data, list):
        return [process_image_data(img) for img in image_data]
    elif isinstance(image_data, dict):
        image = base64.b64decode(image_data["image"].split(",")[1])
        image = Image.open(io.BytesIO(image)).convert("RGBA")
        return {
            "x": image_data.get("x", 0),
            "y": image_data.get("y", 0),
            "sx": image_data.get("sx", 1),
            "sy": image_data.get("sy", 1),
            "image": image
        }
    else:
        image = base64.b64decode(image_data.split(",")[1])
        return Image.open(io.BytesIO(image)).convert("RGBA")

def compose_images(images, masks, image_params):
    width = image_params['width']
    height = image_params['height']
    offset_x = (width - image_params.get("original_width", width)) // 2
    offset_y = (height - image_params.get("original_height", height)) // 2

    composite_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for image_data in images:
        if isinstance(image_data, dict):
            image = image_data["image"]
            x = image_data["x"]
            y = image_data["y"]
            sx = image_data["sx"]
            sy = image_data["sy"]
            if (sx != 1) or (sy != 1):
                image = image.resize((int(image.width * sx), int(image.height * sy)))
            composite_image.paste(image, (offset_x + x, offset_y + y), image)
        else:
            composite_image.paste(image_data, (offset_x, offset_y))
    
    composite_image = composite_image.convert("RGB")
    
    if masks:
        composite_mask = Image.new("L", (width, height), 0)
        for mask_data in masks:
            if isinstance(mask_data, dict):
                mask = mask_data["image"].convert("L")
                x = mask_data["x"]
                y = mask_data["y"]
                sx = mask_data["sx"]
                sy = mask_data["sy"]
                if (sx != 1) or (sy != 1):
                    mask = mask.resize((int(mask.width * sx), int(mask.height * sy)))
                composite_mask.paste(mask, (offset_x + x, offset_y + y), mask)
            else:
                composite_mask.paste(mask_data.convert("L"), (offset_x, offset_y))
    else:
        composite_mask = Image.new("L", (width, height), 255)

    return composite_image, composite_mask



def generate_image_with_pipe(
    pipe: StableDiffusionXLInpaintPipeline,
    **kwargs
) -> Image.Image:
    """
    Generate an image using the Stable Diffusion XL pipeline.
    """
    seed = kwargs.pop('seed', None)
    if seed is not None:
        generator = torch.manual_seed(int(seed))
        kwargs['generator'] = generator
    else:
        kwargs['generator'] = None

    # Ensure image is a PIL Image
    if isinstance(kwargs['image'], np.ndarray):
        kwargs['image'] = Image.fromarray(kwargs['image'].astype(np.uint8))
    elif isinstance(kwargs['image'], torch.Tensor):
        kwargs['image'] = Image.fromarray(kwargs['image'].cpu().numpy().astype(np.uint8))

    # Ensure mask_image is a PIL Image if it exists
    if 'mask_image' in kwargs and kwargs['mask_image'] is not None:
        if isinstance(kwargs['mask_image'], np.ndarray):
            kwargs['mask_image'] = Image.fromarray(kwargs['mask_image'].astype(np.uint8))
        elif isinstance(kwargs['mask_image'], torch.Tensor):
            kwargs['mask_image'] = Image.fromarray(kwargs['mask_image'].cpu().numpy().astype(np.uint8))

    generated_image = pipe(**kwargs).images[0]
    return crop_image(generated_image, kwargs['width'], kwargs['height'])

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
    mask: Image.Image,
    extract_color: Tuple[int, int, int, int]
) -> Image.Image:
    """
    Extract the generated content using the mask.
    """
    return Image.composite(generated_image.convert("RGBA"), Image.new("RGBA", generated_image.size, extract_color), mask)

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