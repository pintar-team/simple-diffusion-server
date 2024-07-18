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
        
        init_image_tensor = image_to_tensor(init_image)
        init_mask_tensor = mask_to_tensor(init_mask, image_params)
        image_params['strength'] = 1.0  # Reset strength for single image generation
        generated_image = generate_image_with_pipe(pipe, image_params, init_image_tensor, init_mask_tensor)
        
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

        composite_image = compose_images(images, image_params['width'], image_params['height'], 
                                         image_params['offset_x'], image_params['offset_y']).convert("RGB")
        composite_mask = compose_images(masks, image_params['width'], image_params['height'], 
                                        image_params['offset_x'], image_params['offset_y']).convert("L") if masks else None

        composite_image_tensor = image_to_tensor(composite_image)
        composite_mask_tensor = mask_to_tensor(composite_mask, image_params) if image_params['apply_mask'] else create_white_mask_tensor(image_params)

        generated_image = generate_image_with_pipe(pipe, image_params, composite_image_tensor, composite_mask_tensor)

        if image_params['extract_mask'] and composite_mask is not None:
            generated_image = extract_masked_content(generated_image, composite_mask, image_params['extract_color'])

        generated_image = crop_image(generated_image, image_params['original_width'], image_params['original_height'])

        return jsonify({"image": encode_image(generated_image, image_params['format'])})

    except Exception as e:
        logger.exception("Error generating img2img")
        return jsonify({"error": str(e)}), 500

def parse_image_params(data: Dict[str, Any]) -> Dict[str, Any]:
    params = {}
    params["prompt"] = data.get("prompt", "")[:300]  # Trim to 300 symbols
    params["negative_prompt"] = data.get("negative_prompt", "")[:300]  # Trim to 300 symbols
    params["num_inference_steps"] = int(data.get("num_inference_steps", 30))
    params["guidance_scale"] = float(data.get("guidance_scale", 7.5))
    params["seed"] = int(data.get("seed")) if data.get("seed") is not None else None
    params["format"] = data.get("format", "jpeg").lower()
    params["original_width"] = int(data.get("width", 1024))
    params["original_height"] = int(data.get("height", 1024))
    params["width"] = ((params["original_width"] + 7) // 8) * 8
    params["height"] = ((params["original_height"] + 7) // 8) * 8
    params["offset_x"] = (params["width"] - params["original_width"]) // 2
    params["offset_y"] = (params["height"] - params["original_height"]) // 2
    params["strength"] = float(data.get("strength", 1.0))
    params["extract_mask"] = bool(data.get("extract_mask", False))
    params["apply_mask"] = bool(data.get("apply_mask", True))
    params["extract_color"] = parse_extract_color(data.get("extract_color", (0, 0, 0, 0)))
    return params

def parse_extract_color(extract_color: Any) -> Tuple[int, int, int, int]:
    if isinstance(extract_color, list):
        return tuple(extract_color)
    elif isinstance(extract_color, str):
        return tuple(map(int, extract_color.split(",")))
    elif isinstance(extract_color, tuple):
        return extract_color
    else:
        return (0, 0, 0, 0)  # Default to transparent black if invalid format

def create_init_image_and_mask(width: int, height: int) -> Tuple[Image.Image, Image.Image]:
    init_image = Image.new("RGB", (width, height))
    init_mask = Image.new("L", (width, height), 255)
    return init_image, init_mask

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

def compose_images(images, width, height, offset_x=0, offset_y=0):
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
    return composite_image

def image_to_tensor(image):
    tensor = torch.from_numpy(np.array(image)).float() / 255.0
    return tensor.permute(2, 0, 1).unsqueeze(0).half().to(device)

def mask_to_tensor(mask, params):
    if mask is not None:
        tensor = torch.from_numpy(np.array(mask)).float() / 255.0
        return tensor.unsqueeze(0).unsqueeze(0).half().to(device)
    else:
        return create_white_mask_tensor(params)

def create_white_mask_tensor(params):
    white_mask = Image.new("L", (params['width'], params['height']), 255)
    tensor = torch.from_numpy(np.array(white_mask)).float() / 255.0
    return tensor.unsqueeze(0).unsqueeze(0).half().to(device)

def generate_image_with_pipe(pipe, params, image_tensor, mask_tensor):
    generator = torch.manual_seed(params['seed']) if params['seed'] is not None else None
    
    return pipe(
        params['prompt'],
        negative_prompt=params['negative_prompt'],
        image=image_tensor,
        mask_image=mask_tensor,
        height=params['height'],
        width=params['width'],
        strength=params['strength'],
        num_inference_steps=params['num_inference_steps'],
        guidance_scale=params['guidance_scale'],
        generator=generator
    ).images[0]

def extract_masked_content(generated_image, mask, extract_color):
    return Image.composite(generated_image.convert("RGBA"), Image.new("RGBA", generated_image.size, extract_color), mask)

def crop_image(image, target_width, target_height):
    width, height = image.size
    if (width != target_width) or (height != target_height):
        left = (width - target_width) // 2
        top = (height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        return image.crop((left, top, right, bottom))
    return image

def encode_image(image, format):
    buffer = io.BytesIO()
    if format == "jpeg":
        image = image.convert("RGB")
    image.save(buffer, format=format)
    mime_type = f"image/{format}"
    return f"data:{mime_type};base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"

if __name__ == '__main__':
    app.run(host=args.host, port=args.port)