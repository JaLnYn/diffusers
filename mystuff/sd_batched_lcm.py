from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineFast
import torch
from functools import partial

def perturb_latents_callback(pipeline, i, t, callback_kwargs, step, max_images):
    latents = callback_kwargs["latents"]

    batch_size = latents.shape[0]
    others = {}
    if batch_size < max_images:
        # TODO: Prohibit cloning and perturbing at the very last iteration
        print(f"End of iteration {i}: Current batch size is {batch_size}, doubling the batch size to {batch_size * 2}")
        latents = latents.repeat(2, 1, 1, 1)
        others = {key: torch.cat([value] * 2) for key, value in callback_kwargs.items() if value is not None}

        for j in range(batch_size, batch_size * 2):
            print(f"End of iteration {i}: Peturbing the latents in batch dimension {j}")
            latents[j] = latents[j] + torch.randn_like(latents[j]) * 0.1

    return {"latents": latents, **others}

@torch.no_grad()
def latents_to_images(pipe, latents):
    needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
    del pipe.unet

    if needs_upcasting:
        pipe.upcast_vae()
        latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)

    has_latents_mean = hasattr(pipe.vae.config, "latents_mean") and pipe.vae.config.latents_mean is not None
    has_latents_std = hasattr(pipe.vae.config, "latents_std") and pipe.vae.config.latents_std is not None
    if has_latents_mean and has_latents_std:
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean).view(1, 4, 1, 1).to(l.device, l.dtype)
        )
        latents_std = (
            torch.tensor(pipe.vae.config.latents_std).view(1, 4, 1, 1).to(l.device, l.dtype)
        )
        latents = latents * latents_std / pipe.vae.config.scaling_factor + latents_mean
    else:
        latents = latents / pipe.vae.config.scaling_factor
    
    print(latents.shape)

    counter = 0
    # for i in range(latents.shape[0]):
    #     l = latents[i].unsqueeze(0)
    #     print(l.shape)
    #     image = pipe.vae.decode(latents, return_dict=False)[0]
    #     image = pipe.image_processor.postprocess(image, output_type="pil")
    #     image[0].save("./images/parallel/astronaut_rides_horse{}.png".format(counter))
    #     counter += 1
    #     del image
    #     del l

    print(latents.shape[0])
    for i in range(latents.shape[0]):
        print("decoding image {}".format(i))
        image = pipe.vae.decode(latents[i].unsqueeze(0), return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")
        image[0].save("./images/parallel/astronaut_rides_horse{}.png".format(counter))
        counter += 1

    if needs_upcasting:
        pipe.vae.to(dtype=torch.float16)

unet = UNet2DConditionModel.from_pretrained(
    "latent-consistency/lcm-sdxl",
    torch_dtype=torch.float16,
    variant="fp16",
)

pipe = StableDiffusionXLPipelineFast.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float8
).to("cuda")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

generator = torch.manual_seed(0)

print(type(pipe))

latents = pipe(
    prompt=prompt, num_inference_steps=4, generator=generator, 
    guidance_scale=8.0, num_images_per_prompt=8, output_type="latent",
    return_dict=False,
    # callback_on_step_end=partial(perturb_latents_callback, step=1, max_images=8), 
    # callback_on_step_end_tensor_inputs=pipe._callback_tensor_inputs
)[0]
#print("a", latents)

latents_to_images(pipe, latents)

print("here")

