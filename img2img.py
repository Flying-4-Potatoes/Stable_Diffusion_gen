import os
import inspect
import warnings
from typing import List, Optional, Union

import torch
from torch import autocast
from tqdm.auto import tqdm
from PIL import Image

from diffusers import StableDiffusionImg2ImgPipeline

import matplotlib.pyplot as plt

if __name__=="__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    device = "cuda"
    model_path = "CompVis/stable-diffusion-v1-4"

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path,
        revision="fp16", 
        torch_dtype=torch.float16,
        use_auth_token=True
    )
    pipe.safety_checker = lambda images, **kwargs: (images, False)
    pipe = pipe.to(device)

    prompt = "A tank parked on the side of the road, trending on artstation"

    img_dir = r"D:\SKTAI\KoDALLE\data\coco\test2014\COCO_test2014_000000000001.jpg"
    init_img = Image.open(img_dir).resize((512, 512))

    generator = torch.Generator(device=device).manual_seed(1024)
    with autocast("cuda"):
        image = pipe(prompt=prompt, init_image=init_img, strength=0.75, guidance_scale=7.5, generator=generator).images[0]

    plt.subplot(1, 2, 1)
    plt.imshow(init_img)
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.show()