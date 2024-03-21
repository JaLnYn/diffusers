from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineFast
import torch


@torch.no_grad()
def latents_to_images(pipe, latents):
    needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast

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
            torch.tensor(pie.vae.config.latents_std).view(1, 4, 1, 1).to(l.device, l.dtype)
        )
        latents = latents * latents_std / pipe.vae.config.scaling_factor + latents_mean
    else:
        latents = latents / pipe.vae.config.scaling_factor


    print(latents.shape)

    counter = 0
    for i in range(latents.shape[0]):

        l = latents[i].unsqueeze(0)

        print(l.shape)

        image = pipe.vae.decode(latents, return_dict=False)[0]

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
    "stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16
).to("cuda")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

generator = torch.manual_seed(0)

print(type(pipe))

latents = pipe(
    prompt=prompt, num_inference_steps=4, generator=generator, guidance_scale=8.0, num_images_per_prompt=1, output_type="latent"
, return_dict=False)[0]
print("a", latents)

latents_to_images(pipe, latents)

print("here")

