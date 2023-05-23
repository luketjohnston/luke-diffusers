
from utils import slugify, image_grid
import torch
from diffusers import StableDiffusionPipeline
import os
from PIL import ImageDraw
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.training_utils import EMAModel


height=512
width=512

#testdir = 'adobe_may11_lr1e-4_b64_r256'
#testdir= 'adobe_may12_lr1e-5_b16_512x768'
#testdir= 'adobe_may12_lr1e-5_b16_512x768'
testdir= 'may18_lr1e-5_b16_r512'

output_dir = f'{testdir}/images_400000'
os.makedirs(output_dir, exist_ok=True)

checkpoint_dir = f'{testdir}/saves/checkpoint-400000'


MODEL_NAME="CompVis/stable-diffusion-v1-4"


# Make models
tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer", revision=None)
text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder", revision=None)
vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae", revision=None)
unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet", revision=None)

# Freeze vae and text_encoder
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

ema_unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet", revision=None)
ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

# Load ema params
load_model = EMAModel.from_pretrained(os.path.join(checkpoint_dir, "unet_ema"), UNet2DConditionModel)
ema_unet.load_state_dict(load_model.state_dict())
#ema_unet.to(accelerator.device)
del load_model


#load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
#model.register_to_config(**load_model.config)
#model.load_state_dict(load_model.state_dict())

# Copy ema params to unet
ema_unet.copy_to(unet.parameters())

def dummy(images, **kwargs):
        return images, [False for i in images]


pipeline = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
    revision=None,
)


original_pipeline = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME)

pipeline.to('cuda')
original_pipeline.to('cuda')

pipeline.safety_checker = dummy
original_pipeline.safety_checker = dummy

num_images=8


def get_pipeline_args():
     pipeline_args =  {
             'guidance_scale': 7.5,
             'generator': torch.Generator("cuda").manual_seed(1024),
             'num_inference_steps': 50,
         }
     return pipeline_args

prompts = [
          "A robot and a ninja swordfighting in front of mt fuji in woodblock style",
          "A panda sleeping in a hammock",
          "A scenic Background of Cape Town South africa, childrens book style, illustration, --ar 3:2, capetown, south africa, backdrop, landscape, horizontal",
          "pixelated pickle in the style of an epic video game, depicted as pixel art, video game items, octane render, colors, holy, full body, 8 k, unreal engine, highly detailed, artgem",
          "Sniper, capybara, detailed magical gun with mystical tree anime style, anime eyes, game art, anime, anime, dynamic pose, ultra-detailed, white background, HD, design art",
          "Jesus is eating hot dogs as a polaroid photo",
          "Cabin in the center of a lake surrounded by forest, landscape photography, 4k",
          "logo for shakes and cakes",
          "Buddha statue in the middle of the street of a crowned Tokyo city",
          "An intricate landscape trapped in a bottle, atmospheric lighting, intricate detail, bright in the style of greek mythology, gods and goddesses, mythic creatures, graphic novel style, dragons, embellished crystal and gemstones, volumetric rays, by greg rutkowski, deviantart, artstation, fantasy art, highly detailed, 8k, concept art, sharp focus",
          "modern architecture",
          "cat",
          ]

while True:
    if prompts:
        prompt = prompts.pop()
    else:
        prompt = input('prompt?')

    filename = slugify(prompt)
    filename = filename[:240]

    orig_images = original_pipeline(prompt=[prompt]*num_images, height=height, width=width, **get_pipeline_args()).images
    # Original works at 512 by 512
    #if width != 512 or height != 512:
    #    for i,im in enumerate(orig_images):
    #        orig_images[i] = im.resize((width,height))

    images = pipeline(prompt=[prompt]*num_images, height=height, width=width, **get_pipeline_args()).images

    all_images = orig_images + images

    grid = image_grid(all_images, rows=2, cols=num_images)

    draw = ImageDraw.Draw(grid)
     
    # Add Text to an image
    draw.text((28, 36), "original", fill=(255, 255, 255))
    draw.text((28, height+36), "finetuned", fill=(255, 255, 255))

    if not filename:
        filename = 'filename'

    grid.save(f"{output_dir}/{filename}.png")
