import os
import argparse
from typing import List, Dict, Any
import base64
from PIL import Image
import io

def parse_args():
    parser = argparse.ArgumentParser(description='Stable Diffusion Server')
    parser.add_argument('--model', type=str, default='stabilityai/stable-diffusion-xl-base-1.0', help='Model name for DiffusionPipeline')
    parser.add_argument('--unet', type=str, default='', help='UNet model name')
    parser.add_argument('--lora_dirs', type=str, default='', help='Colon-separated list of LoRA directories')
    parser.add_argument('--lora_scales', type=str, default='', help='Colon-separated list of LoRA scales')
    parser.add_argument('--scheduler', type=str, default='euler_a', help='Scheduler')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host')
    parser.add_argument('--port', type=int, default=8001, help='Port')
    parser.add_argument('--vae', type=str, default='madebyollin/sdxl-vae-fp16-fix', help='Model name for VAE')
    return parser.parse_args()

def is_local_file(path):
    return os.path.isfile(path)

class Args:
    def __init__(self, model, unet, lora_dirs, lora_scales, scheduler, host, port, vae):
        self.model = model
        self.unet = unet
        self.lora_dirs = lora_dirs
        self.lora_scales = lora_scales
        self.scheduler = scheduler
        self.host = host
        self.port = port
        self.vae = vae

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