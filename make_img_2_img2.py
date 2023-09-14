import os
from PIL import Image#, ImageDraw
# import cv2
# import numpy as np
# from base64 import b64encode

import torch
from torch import autocast
from torch.nn import functional as F
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline#, AutoencoderKL
# from diffusers import UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler
# from diffusers.schedulers.scheduling_ddim import DDIMScheduler
# from transformers import CLIPTextModel, CLIPTokenizer
# from tqdm.auto import tqdm
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
from transformers import pipeline

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
    

    pipe2 = StableDiffusionImg2ImgPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', revision='fp16',
        torch_dtype=torch.float16, use_auth_token=True)
    pipe2.safety_checker = lambda images, **kwargs: (images, False)
    pipe2 = pipe2.to(device)
    
    summarizer = pipeline("summarization", model="linydub/bart-large-samsum")
    
    init_img = Image.open(r"C:\Users\PC\Downloads\캡3처.JPG").resize((512, 512))
    # df = pd.read_csv(f_root+"fairy_tale_grimm_dialog_removed.tsv", sep='\t', encoding='utf-8')
    df = pd.read_csv(f_root+"fairy_tale_grimm.tsv", sep='\t', encoding='utf-8')
    st_curr = -1
    for idx, n, prompt, _ in tqdm(df.itertuples(index=False, name=None)):
        # generate initial image
        if n!=st_curr:
            st_curr = n
            with autocast(device):
                init_img = pipe(add_postfix("Beautiful landscape"))['sample'][0]
            init_img.save(f_root + f"init_imgs/{st_curr:03d}.jpg")
                
        # generate image
        prompt = add_postfix(prompt, -4)
        generator = torch.Generator(device=device).manual_seed(1024)
        with autocast(device):
            image = pipe2(prompt=prompt, init_image=init_img, num_inference_steps=60, strength=0.9, guidance_scale=7.5, generator=generator).images[0]
        print(prompt)
        
        # image.save(f_root + f"imageset/{idx:06d}.jpg")
        plt.subplot(1, 3, 1), plt.imshow(init_img)
        plt.subplot(1, 3, 2), plt.imshow(image)
        
        prompt = summarizer(prompt, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        prompt = add_postfix(prompt, -4)
        generator = torch.Generator(device=device).manual_seed(1024)
        with autocast(device):
            image = pipe2(prompt=prompt, init_image=init_img, num_inference_steps=60, strength=0.9, guidance_scale=7.5, generator=generator).images[0]
        print(prompt)
        
        plt.subplot(1, 3, 3), plt.imshow(image)
        plt.show()
        