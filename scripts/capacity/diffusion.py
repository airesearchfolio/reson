import gc, torch
def cleanup_pipe(pipe):
    try: del pipe
    except Exception: pass
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def load_sdxl_txt2img(config):
    from diffusers import StableDiffusionXLPipeline
    kw=dict(torch_dtype=torch.float16,use_safetensors=True,variant="fp16")
    try: pipe=StableDiffusionXLPipeline.from_pretrained(config.sdxl_model_id,**kw)
    except Exception:
        kw.pop("variant",None)
        pipe=StableDiffusionXLPipeline.from_pretrained(config.sdxl_model_id,**kw)
    pipe.to(config.get_device()); pipe.enable_vae_slicing(); pipe.set_progress_bar_config(disable=True)
    return pipe

@torch.inference_mode()
def generate_from_latent(pipe,latent,prompt,config):
    return pipe(prompt=str(prompt),latents=latent.half(),height=config.image_size,width=config.image_size,
                num_inference_steps=config.generation_steps,guidance_scale=config.guidance).images[0].convert("RGB")
