import os
from flask import Flask, request, jsonify
from diffusers import UNet2DConditionModel, StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, AutoencoderKL
import torch
from PIL import Image, ImageOps
import io
import base64
import logging
import numpy as np

from utils import parse_args, Args, is_local_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

is_gunicorn = "gunicorn" in os.environ.get("SERVER_SOFTWARE", "")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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



#vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.bfloat16)

def load_models():
    print("Loading models...")

    if args.vae == '' or args.vae == 'baked':
        if args.unet == '':
            if is_local_file(args.model):
                pipe = StableDiffusionXLInpaintPipeline.from_single_file(args.model, torch_dtype=torch.bfloat16, variant="fp16", use_safetensors=True, num_in_channels=4, ignore_mismatched_sizes=True)
            else:    
                pipe = StableDiffusionXLInpaintPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, variant="fp16")
        else:
            unet = UNet2DConditionModel.from_pretrained(args.unet, torch_dtype=torch.bfloat16, variant="fp16")
            if is_local_file(args.model):
                pipe = StableDiffusionXLInpaintPipeline.from_single_file(args.model, unet=unet, torch_dtype=torch.bfloat16, variant="fp16", use_safetensors=True, num_in_channels=4, ignore_mismatched_sizes=True)
            else:
                pipe = StableDiffusionXLInpaintPipeline.from_pretrained(args.model, unet=unet, torch_dtype=torch.bfloat16, variant="fp16")
    else:
        vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=torch.bfloat16, variant="fp16")
        if args.unet == '':
            if is_local_file(args.model):
                pipe = StableDiffusionXLInpaintPipeline.from_single_file(args.model, vae=vae, torch_dtype=torch.bfloat16, variant="fp16", use_safetensors=True, num_in_channels=4, ignore_mismatched_sizes=True)
            else:    
                pipe = StableDiffusionXLInpaintPipeline.from_pretrained(args.model, vae=vae, torch_dtype=torch.bfloat16, variant="fp16")
        else:
            unet = UNet2DConditionModel.from_pretrained(args.unet, torch_dtype=torch.bfloat16, variant="fp16")
            if is_local_file(args.model):
                pipe = StableDiffusionXLInpaintPipeline.from_single_file(args.model, vae=vae, unet=unet, torch_dtype=torch.bfloat16, variant="fp16", use_safetensors=True, num_in_channels=4, ignore_mismatched_sizes=True)
            else:
                pipe = StableDiffusionXLInpaintPipeline.from_pretrained(args.model, vae=vae, unet=unet, torch_dtype=torch.bfloat16, variant="fp16")

    lora_dirs = args.lora_dirs.split(':') if args.lora_dirs else []
    lora_scales = [float(scale) for scale in args.lora_scales.split(':')] if args.lora_scales else []

    if len(lora_dirs) != len(lora_scales):
        raise ValueError("The number of LoRA directories must match the number of scales")

    for ldir, lsc in zip(lora_dirs, lora_scales):
        pipe.load_lora_weights(ldir)
        pipe.fuse_lora(lora_scale=lsc)

    if args.scheduler == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler == "euler_a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    
    pipe.to(device)
    #pipe.enable_vae_slicing()
    #pipe.enable_attention_slicing()

    print("Models loaded")
    return pipe

pipe = load_models()

app = Flask(__name__)

@app.route('/generate-image', methods=['POST'])
def generate_image():
    try:
        data = request.json
        prompt = data.get("prompt")
        negative_prompt = data.get("negative_prompt", None)
        num_inference_steps = data.get("num_inference_steps", 30)
        guidance_scale = data.get("guidance_scale", 7.5)
        seed = data.get("seed", None)
        image_format = data.get("format", "jpeg").lower()

        original_width = data.get("width", 1024)
        original_height = data.get("height", 1024)

        width = ((original_width + 7) // 8) * 8
        height = ((original_height + 7) // 8) * 8

        if seed is not None:
            generator = torch.manual_seed(seed)
        else:
            generator = None

        if image_format not in ["jpeg", "png"]:
            return jsonify({"error": "Invalid image format. Choose 'jpeg' or 'png'."}), 400

        init_image = Image.new("RGB", (width, height))
        init_image_tensor = torch.from_numpy(np.array(init_image)).float() / 255.0
        init_image_tensor = init_image_tensor.permute(2, 0, 1).unsqueeze(0)
        init_image_tensor = init_image_tensor.half().to(device)

        white_mask = Image.new("L", (width, height), 255)
        while_mask_tensor = torch.from_numpy(np.array(white_mask)).float() / 255.0
        while_mask_tensor = while_mask_tensor.unsqueeze(0).unsqueeze(0)
        while_mask_tensor = while_mask_tensor.half().to(device)

        generated_image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            image=init_image_tensor,
            mask_image=while_mask_tensor,
            height=height,
            width=width,
            strength=1,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        gen_width, gen_height = generated_image.size
        if (gen_width != original_width) or (gen_width != original_height):
            left = (gen_width - original_width) // 2
            top = (gen_height - original_height) // 2
            right = left + original_width
            bottom = top + original_height
            generated_image = generated_image.crop((left, top, right, bottom))

        buffer = io.BytesIO()
        if image_format == "jpeg":
            # Convert the image to RGB color mode for JPEG format
            generated_image = generated_image.convert("RGB")

        generated_image.save(buffer, format=image_format)
        mime_type = "image/jpeg" if image_format == "jpeg" else "image/png"
        data_uri = "data:" + mime_type + ";base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify({"image": data_uri})

    except Exception as e:
        logger.exception("Error generating image")
        return jsonify({"error": str(e)}), 500
    
@app.route('/generate-img2img', methods=['POST'])
def generate_img2img():
    try:
        data = request.json
        prompt = data.get("prompt")
        negative_prompt = data.get("negative_prompt", None)
        num_inference_steps = data.get("num_inference_steps", 30)
        guidance_scale = data.get("guidance_scale", 7.5)
        seed = data.get("seed", None)
        image_format = data.get("format", "jpeg").lower()
        strength = data.get("strength", 0.8)
        extract_mask = data.get("extract_mask", False)
        apply_mask = data.get("apply_mask", True)
        # Parse the extract_color parameter
        extract_color = data.get("extract_color", (0, 0, 0, 0))
        if isinstance(extract_color, list):
            extract_color = tuple(extract_color)
        elif isinstance(extract_color, str):
            extract_color = tuple(map(int, extract_color.split(",")))
        else:
            extract_color = (0, 0, 0, 0)  # Default to transparent black if invalid format


        original_width = data.get("width", 1024)
        original_height = data.get("height", 1024)

        width = ((original_width + 7) // 8) * 8
        height = ((original_height + 7) // 8) * 8

        offset_x = (width - original_width) // 2
        offset_y = (height - original_height) // 2

        if seed is not None:
            generator = torch.manual_seed(seed)
        else:
            generator = None

        if image_format not in ["jpeg", "png"]:
            return jsonify({"error": "Invalid image format. Choose 'jpeg' or 'png'."}), 400

        images_data = data.get("images", [])
        masks_data = data.get("masks", [])

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

        images = process_image_data(images_data)
        masks = process_image_data(masks_data) if masks_data else None

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

        composite_image = compose_images(images, width, height, offset_x, offset_y).convert("RGB")
        composite_mask = compose_images(masks, width, height, offset_x, offset_y).convert("L") if masks else None
        #Convert to tensor
        composite_image_tensor = torch.from_numpy(np.array(composite_image)).float() / 255.0
        composite_image_tensor = composite_image_tensor.permute(2, 0, 1).unsqueeze(0)

        if composite_mask is not None and apply_mask:
            composite_mask_tensor = torch.from_numpy(np.array(composite_mask)).float() / 255.0
            composite_mask_tensor = composite_mask_tensor.unsqueeze(0).unsqueeze(0)
        else:
            #create a white mask
            white_mask = Image.new("L", (width, height), 255)
            composite_mask_tensor = torch.from_numpy(np.array(white_mask)).float() / 255.0
            composite_mask_tensor = composite_mask_tensor.unsqueeze(0).unsqueeze(0)

        # Convert to fp16 and move to CUDA
        composite_image_tensor = composite_image_tensor.half().to(device)
        if composite_mask_tensor is not None:
            composite_mask_tensor = composite_mask_tensor.half().to(device)

        # Print size
        #print(composite_image_tensor.size())
        #print(composite_mask_tensor.size() if composite_mask_tensor is not None else None)

        # Generate the image using the composite image and mask
        generated_image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            image=composite_image_tensor,
            mask_image=composite_mask_tensor,
            height=height,
            width=width,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator).images[0]
        
        #print("generated_image size:", generated_image.size)
        #print("composite_mask size:", composite_mask.size if composite_mask is not None else None)
        #print("composite_mask_tensor size:", composite_mask_tensor.size() if composite_mask_tensor is not None else None)

        if extract_mask and composite_mask_tensor is not None:
            # Extract the generated content using the mask
            generated_image = Image.composite(generated_image.convert("RGBA"), Image.new("RGBA", generated_image.size, extract_color), composite_mask)
        else:
            generated_image = generated_image

        gen_width, gen_height = generated_image.size
        if (gen_width != original_width) or (gen_width != original_height):
            left = (gen_width - original_width) // 2
            top = (gen_height - original_height) // 2
            right = left + original_width
            bottom = top + original_height
            generated_image = generated_image.crop((left, top, right, bottom))

        buffer = io.BytesIO()
        if image_format == "jpeg":
            # Convert the image to RGB color mode for JPEG format
            generated_image = generated_image.convert("RGB")

        generated_image.save(buffer, format=image_format)
        mime_type = "image/jpeg" if image_format == "jpeg" else "image/png"
        data_uri = "data:" + mime_type + ";base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify({"image": data_uri})

    except Exception as e:
        logger.exception("Error generating img2img")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host=args.host, port=args.port)