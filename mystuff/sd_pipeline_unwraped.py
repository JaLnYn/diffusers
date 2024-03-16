from diffusers import StableDiffusionPipeline
import torch
import numpy as np
from PIL import Image
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

@torch.no_grad()
def run(prompt, n_prompt):
    batch_size = 1
    num_images_per_prompt = 1
    width = pipe.unet.config.sample_size * pipe.vae_scale_factor
    height = pipe.unet.config.sample_size * pipe.vae_scale_factor 
    device = pipe._execution_device
    num_inference_steps = 8
    output_type = "pil"
    pipe._guidance_scale = 7.5

    pipe.check_inputs(
        prompt,
        height,
        width,
        None,
        n_prompt,
        None,
        None,
        None,
        None,
        None,
    )

    pipe._guidance_rescale = 0 
    pipe._cross_attention_kwargs = None 
    pipe._interrupt = False

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        pipe._guidance_scale > 1 and pipe.unet.config.time_cond_proj_dim is None,
        n_prompt,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        lora_scale=None,
        clip_skip=None,
        )

    if pipe.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    print(pipe.unet.config.in_channels)
    latents = pipe.prepare_latents(
        batch_size * num_images_per_prompt,
        pipe.unet.config.in_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        None,
        None,
    )

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    print(latents.shape)

    num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
    pipe._num_timesteps = len(timesteps)
    print(num_warmup_steps)
    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if pipe._interrupt: 
                continue

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=None,
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
            )[0]

            # perform guidance
            if pipe.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # call the callback, if provided
            if i == len(timesteps) - 1 or (i + 1) % pipe.scheduler.order == 0:
                progress_bar.update()

    print("here", latents.shape)
    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False, generator=None)[0]

    image = pipe.image_processor.postprocess(image, output_type=output_type, do_denormalize=[True] * image.shape[0])
    print("post")

    print(type(image))

    pipe.maybe_free_model_hooks()
    return StableDiffusionPipelineOutput(image, None)



image = run("a photo of an astronaut riding a horse on mars", None).images[0]
    
image.save("astronaut_rides_horse.png")