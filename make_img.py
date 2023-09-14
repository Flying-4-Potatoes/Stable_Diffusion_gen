import os
from PIL import Image#, ImageDraw
# import cv2
# import numpy as np
# from base64 import b64encode

import torch
from torch import autocast
from torch.nn import functional as F
from diffusers import StableDiffusionPipeline#, AutoencoderKL
# from diffusers import UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler
# from diffusers.schedulers.scheduling_ddim import DDIMScheduler
# from transformers import CLIPTextModel, CLIPTokenizer
# from tqdm.auto import tqdm
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd

# def image_grid(imgs, rows, cols):
#     assert len(imgs) == rows*cols

#     w, h = imgs[0].size
#     grid = Image.new('RGB', size=(cols*w, rows*h))
#     grid_w, grid_h = grid.size
    
#     for i, img in enumerate(imgs):
#         grid.paste(img, box=(i%cols*w, i//cols*h))
#     return grid

# def multi_imgen(prompts, n_images=3):
#     with autocast(device):
#         images = pipe(prompts, num_inference_steps=50)['sample']
#     return image_grid(images, rows=1, cols=3)


def get_tsv_eng(dir:str):
    df = pd.read_csv(dir, sep='\t', encoding='utf-8')
    return df['en'].tolist()


def add_postfix(prompt='', start=0, end=-1):
    template = ["fairy tale", "illustration", "cartoon style", "Ghibli style", "Disney style artwork", "procreate", "adobe illustrator", "hand drawn", 'digital illustration', '4k', 'detailed', "trending on artstation", "art by greg rutkowski", 'fantasy vivid colors', '']
    return prompt + ', ' + ', '.join(template[start:end])

if __name__=="__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    device = 'cuda'
    f_root = "D:/SKTAI/Grimm Fairy Tale preprocess/"


    # make sure you're logged in with `huggingface-cli login`
    pipe = StableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', revision='fp16',
        torch_dtype=torch.float16, use_auth_token=True)
    
    pipe.safety_checker = lambda images, **kwargs: (images, False)
    pipe = pipe.to(device)
    
    prompt = "Human heart is strapped around with three iron bands"
    prompts = get_tsv_eng(f_root+"fairy_tale_grimm_dialog_removed.tsv")
    for prompt in tqdm(prompts):
        prompt = add_postfix(prompt, -4)
        print('\n', prompt)
        with autocast(device):
            image = pipe(prompt)['sample'][0]
        plt.imshow(image)
        plt.show()
        # image.save("./{num:06i}img.jpg")