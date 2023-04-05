import torch
import argparse
import gradio as gr
from functools import partial
from my.config import BaseConf, dispatch_gradio
from run_3DFuse import SJC_3DFuse
import numpy as np
from PIL import Image
from pc_project import point_e
from diffusers import UnCLIPPipeline, DiffusionPipeline
from pc_project import point_e_gradio
import numpy as np
import plotly.graph_objs as go
from my.utils.seed import seed_everything
import os

def gen_pc_from_image(image, prompt='a dog', keyword='dog', bg_preprocess=True, seed=2023) :
    
    seed_everything(seed=seed)
    if keyword not in prompt:
        raise gr.Error("Prompt should contain keyword!")
    elif " " in keyword:
        raise gr.Error("Keyword should be one word!")
    
    if bg_preprocess:
        import cv2
        from carvekit.api.high import HiInterface
        interface = HiInterface(object_type="object",
                        batch_size_seg=5,
                        batch_size_matting=1,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                        matting_mask_size=2048,
                        trimap_prob_threshold=231,
                        trimap_dilation=30,
                        trimap_erosion_iters=5,
                        fp16=False)
        
        img_without_background = interface([image])
        mask = np.array(img_without_background[0]) > 127
        image = np.array(image)
        image[~mask] = [255., 255., 255.]
        image = Image.fromarray(np.array(image))
    
    
    points = point_e_gradio(image,'cuda')
    

    return image

from PIL import Image
img = Image.open('dog.jpeg')
gen_pc_from_image(img)
