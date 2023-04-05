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

class Intermediate:
    def __init__(self):
        self.images = None
        self.points = None
        self.is_generating = False


def gen_3d(model, intermediate, prompt, keyword, seed, ti_step, pt_step) :
    print('run gen_3d','='*50)
    if intermediate is not None:
        intermediate.is_generating = True
    images, points = intermediate.images, intermediate.points
    if images is None or points is None :
        raise gr.Error("Please generate point cloud first")
    del model
    torch.cuda.empty_cache()
    
    seed_everything(seed)
    model = dispatch_gradio(SJC_3DFuse, prompt, keyword, ti_step, pt_step, seed)
    setting = model.dict()
    exp_dir = os.path.join(setting['exp_dir'],keyword)
    initial_images_dir = os.path.join(exp_dir, 'initial_image')
    os.makedirs(initial_images_dir,exist_ok=True)
    
    for idx,img in enumerate(images) :
        img.save( os.path.join(initial_images_dir, f"instance{idx}.png") )
    
    torch.cuda.empty_cache()
    yield from model.run_gradio(points)
    torch.cuda.empty_cache()
    
    intermediate.is_generating = False
    


def gen_pc_from_prompt(intermediate, num_initial_image, prompt, keyword, type, bg_preprocess, seed) :
    
    seed_everything(seed=seed)
    if keyword not in prompt:
        raise gr.Error("Prompt should contain keyword!")
    elif " " in keyword:
        raise gr.Error("Keyword should be one word!")
    
    images = gen_init(num_initial_image=num_initial_image, prompt=prompt, type=type,  bg_preprocess=bg_preprocess)
    points = point_e_gradio(images[0],'cuda')
    torch.cuda.empty_cache()
    
    intermediate.images = images
    intermediate.points = points
    
    coords = np.array(points.coords)
    trace = go.Scatter3d(x=coords[:,0], y=coords[:,1], z=coords[:,2], mode='markers', marker=dict(size=2))

    layout = go.Layout(
        scene=dict(
            xaxis=dict(
                title="",
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks='',
                showticklabels=False
            ),
            yaxis=dict(
                title="",
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks='',
                showticklabels=False
            ),
            zaxis=dict(
                title="",
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks='',
                showticklabels=False
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False
    )

    fig = go.Figure(data=[trace], layout=layout)

    return images[0], fig


def gen_pc_from_image(intermediate, image, prompt, keyword, bg_preprocess, seed) :
    
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
    torch.cuda.empty_cache()
    
    if intermediate is not None:
        intermediate.images = [image]
        intermediate.points = points
    
    coords = np.array(points.coords)
    trace = go.Scatter3d(x=coords[:,0], y=coords[:,1], z=coords[:,2], mode='markers', marker=dict(size=2))

    layout = go.Layout(
        scene=dict(
            xaxis=dict(
                title="",
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks='',
                showticklabels=False
            ),
            yaxis=dict(
                title="",
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks='',
                showticklabels=False
            ),
            zaxis=dict(
                title="",
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks='',
                showticklabels=False
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False
    )

    fig = go.Figure(data=[trace], layout=layout)
    del interface

    return image, fig

def gen_init(num_initial_image, prompt, type="Karlo",  bg_preprocess=False):
    pipe = UnCLIPPipeline.from_pretrained("kakaobrain/karlo-v1-alpha", torch_dtype=torch.float16) if type=="Karlo (Recommended)" \
        else DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = pipe.to('cuda')
    
    view_prompt=["front view of ","overhead view of ","side view of ", "back view of "]
    
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

    images = []
    for i in range(num_initial_image):
        t=", white background" if bg_preprocess else ", white background"
        if i==0:
            prompt_ = f"{view_prompt[i%4]}{prompt}{t}"
        else:
            prompt_ = f"{view_prompt[i%4]}{prompt}"

        image = pipe(prompt_).images[0]
        
        if bg_preprocess:
            # motivated by NeuralLift-360 (removing bg)
            # NOTE: This option was added during the code orgranization process.
            # The results reported in the paper were obtained with [bg_preprocess: False] setting.
            img_without_background = interface([image])
            mask = np.array(img_without_background[0]) > 127
            image = np.array(image)
            image[~mask] = [255., 255., 255.]
            image = Image.fromarray(np.array(image))
        images.append(image)
            
    return images
            
        

if __name__ == '__main__':
    image_input = Image.open('./dog.jpeg')
    prompt_input_2 = 'a dog'
    word_input_2 = 'dog'
    preprocess_choice_2 = True
    intermediate = Intermediate()
    seed_2 = 2023
    model = None
    print('step 1: gen_pc_from_image')
    init_output, pc_plot = gen_pc_from_image(intermediate, image_input, prompt_input_2, word_input_2,
                                   preprocess_choice_2, seed_2)
    torch.cuda.empty_cache()
    
    print('step 2: gen_3d')
    opt_step_2 = 1
    pivot_step_2 = 1
    intermediate_output,logs,video_result = gen_3d(model,intermediate,prompt_input_2, word_input_2, seed_2, opt_step_2, pivot_step_2)
    

