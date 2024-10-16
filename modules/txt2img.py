import os
import json
import time  # Import the time module to measure generation time
from contextlib import closing

import modules.scripts
from modules import processing, infotext_utils
from modules.infotext_utils import create_override_settings_dict, parse_generation_parameters
from modules.shared import opts
import modules.shared as shared
from modules.ui import plaintext_to_html
from PIL import Image
import gradio as gr
import torch
import gc

# Modify this path to specify where the GPU memory information will be saved.
MEMORY_LOG_PATH = "benchmarking_results.txt"

def measure_memory_usage(func):
    def wrapper(*args, **kwargs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        max_memory = None
        try:
            with torch.no_grad():
                result = func(*args, **kwargs)

            max_memory = torch.cuda.max_memory_allocated() / 1024**3  # Convert MB to GB
        except torch.cuda.OutOfMemoryError:
            print("Out of memory error during inference.")
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        return result, max_memory

    return wrapper

@measure_memory_usage
def process_images_with_memory_measurement(p):
    return processing.process_images(p)

def log_memory_and_dimensions_to_file(memory_info, width, height, batch_size, prompt, generation_time):
    """Writes memory info, image dimensions, batch size, prompt, and generation time to a specified file."""
    try:
        with open(MEMORY_LOG_PATH, "a") as f:
            f.write(f"Prompt: {prompt}\nImage dimensions: {width}x{height}, Batch size: {batch_size}, Generation time: {generation_time:.2f} seconds, {memory_info}\n")
    except Exception as e:
        print(f"Failed to write memory info to file: {e}")

def txt2img_create_processing(id_task: str, request: gr.Request, prompt: str, negative_prompt: str, prompt_styles, n_iter: int, batch_size: int, cfg_scale: float, height: int, width: int, enable_hr: bool, denoising_strength: float, hr_scale: float, hr_upscaler: str, hr_second_pass_steps: int, hr_resize_x: int, hr_resize_y: int, hr_checkpoint_name: str, hr_sampler_name: str, hr_scheduler: str, hr_prompt: str, hr_negative_prompt, override_settings_texts, *args, force_enable_hr=False):
    override_settings = create_override_settings_dict(override_settings_texts)

    if force_enable_hr:
        enable_hr = True

    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=prompt_styles,
        negative_prompt=negative_prompt,
        batch_size=batch_size,
        n_iter=n_iter,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        enable_hr=enable_hr,
        denoising_strength=denoising_strength,
        hr_scale=hr_scale,
        hr_upscaler=hr_upscaler,
        hr_second_pass_steps=hr_second_pass_steps,
        hr_resize_x=hr_resize_x,
        hr_resize_y=hr_resize_y,
        hr_checkpoint_name=None if hr_checkpoint_name == 'Use same checkpoint' else hr_checkpoint_name,
        hr_sampler_name=None if hr_sampler_name == 'Use same sampler' else hr_sampler_name,
        hr_scheduler=None if hr_scheduler == 'Use same scheduler' else hr_scheduler,
        hr_prompt=hr_prompt,
        hr_negative_prompt=hr_negative_prompt,
        override_settings=override_settings,
    )

    p.scripts = modules.scripts.scripts_txt2img
    p.script_args = args

    p.user = request.username

    if shared.opts.enable_console_prompts:
        print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)

    return p

def txt2img_upscale(id_task: str, request: gr.Request, gallery, gallery_index, generation_info, *args):
    assert len(gallery) > 0, 'No image to upscale'
    assert 0 <= gallery_index < len(gallery), f'Bad image index: {gallery_index}'

    p = txt2img_create_processing(id_task, request, *args, force_enable_hr=True)
    p.batch_size = 1
    p.n_iter = 1
    p.txt2img_upscale = True

    geninfo = json.loads(generation_info)

    image_info = gallery[gallery_index] if 0 <= gallery_index < len(gallery) else gallery[0]
    p.firstpass_image = infotext_utils.image_from_url_text(image_info)

    parameters = parse_generation_parameters(geninfo.get('infotexts')[gallery_index], [])
    p.seed = parameters.get('Seed', -1)
    p.subseed = parameters.get('Variation seed', -1)

    p.override_settings['save_images_before_highres_fix'] = False

    start_time = time.time()  # Record the start time

    with closing(p):
        processed = modules.scripts.scripts_txt2img.run(p, *p.script_args)

        if processed is None:
            processed, max_memory = process_images_with_memory_measurement(p)
            generation_time = time.time() - start_time  # Calculate generation time
            if max_memory is not None:
                memory_info = f"Max GPU memory used during upscale: {max_memory:.2f} GB"
                print(memory_info)
                log_memory_and_dimensions_to_file(memory_info, p.width, p.height, p.batch_size, p.prompt, generation_time)  # Log memory, dimensions, batch size, prompt, generation time
                if isinstance(processed.comments, list):
                    processed.comments.append(memory_info)
                elif isinstance(processed.comments, str):
                    processed.comments += f"\n{memory_info}"
                else:
                    processed.comments = memory_info

    shared.total_tqdm.clear()

    new_gallery = []
    for i, image in enumerate(gallery):
        if i == gallery_index:
            geninfo["infotexts"][gallery_index: gallery_index+1] = processed.infotexts
            new_gallery.extend(processed.images)
        else:
            fake_image = Image.new(mode="RGB", size=(1, 1))
            fake_image.already_saved_as = image["name"].rsplit('?', 1)[0]
            new_gallery.append(fake_image)

    geninfo["infotexts"][gallery_index] = processed.info

    return new_gallery, json.dumps(geninfo), plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")

def txt2img(id_task: str, request: gr.Request, *args):
    p = txt2img_create_processing(id_task, request, *args)

    start_time = time.time()  # Record the start time

    with closing(p):
        processed = modules.scripts.scripts_txt2img.run(p, *p.script_args)

        if processed is None:
            processed, max_memory = process_images_with_memory_measurement(p)
            generation_time = time.time() - start_time  # Calculate generation time
            if max_memory is not None:
                memory_info = f"Max GPU memory used during txt2img: {max_memory:.2f} GB"
                print(memory_info)
                log_memory_and_dimensions_to_file(memory_info, p.width, p.height, p.batch_size, p.prompt, generation_time)  # Log memory, dimensions, batch size, prompt, generation time
                if isinstance(processed.comments, list):
                    processed.comments.append(memory_info)
                elif isinstance(processed.comments, str):
                    processed.comments += f"\n{memory_info}"
                else:
                    processed.comments = memory_info

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return processed.images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")
