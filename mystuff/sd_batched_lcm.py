from diffusers import DiffusionPipeline
import torch
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from pathlib import Path
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
Path("./images/parallel").mkdir(parents=True, exist_ok=True)

torch.manual_seed(0)

pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
pipe = pipe.to("cuda")

noise_level = 1



@torch.no_grad()
def run(prompt, pipe):
    def _encode_prompt(
        pipe,
        prompt,
        device,
        num_images_per_prompt,
        prompt_embeds: None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """

        if prompt is not None and isinstance(prompt, str):
            pass
        elif prompt is not None and isinstance(prompt, list):
            len(prompt)
        else:
            prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = pipe.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = pipe.tokenizer.batch_decode(
                    untruncated_ids[:, pipe.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {pipe.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(pipe.text_encoder.config, "use_attention_mask") and pipe.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = pipe.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        if pipe.text_encoder is not None:
            prompt_embeds_dtype = pipe.text_encoder.dtype
        elif pipe.unet is not None:
            prompt_embeds_dtype = pipe.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # Don't need to get uncond prompt embedding because of LCM Guided Distillation
        return prompt_embeds



    batch_size = 1
    num_images_per_prompt = 8
    width = pipe.unet.config.sample_size * pipe.vae_scale_factor
    height = pipe.unet.config.sample_size * pipe.vae_scale_factor 
    num_inference_steps = 16
    output_type = "pil"
    pipe._guidance_scale = 7.5
    lcm_origin_steps = 50

    pipe._guidance_rescale = 0 
    pipe._cross_attention_kwargs = None 
    pipe._interrupt = False

     # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)

    device = pipe._execution_device
    # do_classifier_free_guidance = guidance_scale > 0.0  # In LCM Implementation:  cfg_noise = noise_cond + cfg_scale * (noise_cond - noise_uncond) , (cfg_scale > 0.0 using CFG)

    # 3. Encode input prompt
    prompt_embeds = _encode_prompt(
        pipe,
        prompt,
        device,
        num_images_per_prompt,
        None
    )

    # 4. Prepare timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, lcm_origin_steps)
    timesteps = pipe.scheduler.timesteps

    # 5. Prepare latent variable
    num_channels_latents = pipe.unet.config.in_channels
    latents = pipe.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        latents,
    )
    bs = batch_size * num_images_per_prompt

    # 6. Get Guidance Scale Embedding
    w = torch.tensor(guidance_scale).repeat(bs)
    w_embedding = pipe.get_w_embedding(w, embedding_dim=256).to(device=device, dtype=latents.dtype)

    # 7. LCM MultiStep Sampling Loop:
    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            ts = torch.full((bs,), t, device=device, dtype=torch.long)
            latents = latents.to(prompt_embeds.dtype)

            # model prediction (v-prediction, eps, x)
            model_pred = pipe.unet(
                latents,
                ts,
                timestep_cond=w_embedding,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # compute the previous noisy sample x_t -> x_t-1
            latents, denoised = pipe.scheduler.step(model_pred, i, t, latents, return_dict=False)

            # # call the callback, if provided
            # if i == len(timesteps) - 1:
            progress_bar.update()

    denoised = denoised.to(prompt_embeds.dtype)
    if not output_type == "latent":
        image = pipe.vae.decode(denoised / pipe.vae.config.scaling_factor, return_dict=False)[0]
        image, has_nsfw_concept = pipe.run_safety_checker(image, device, prompt_embeds.dtype)
    else:
        image = denoised
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

    image = pipe.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    if not return_dict:
        return (image, has_nsfw_concept)

    return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


images = run("a photo of an astronaut riding a horse on mars", pipe).images
counter = 0
for i in images:
    i.save("./images/parallel/astronaut_rides_horse{}.png".format(counter))
    counter += 1
    
