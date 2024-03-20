from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
import torch

unet = UNet2DConditionModel.from_pretrained(
    "latent-consistency/lcm-sdxl",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16
).to("cuda")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

generator = torch.manual_seed(0)

print(type(pipe))

images = pipe(
    prompt=prompt, num_inference_steps=4, generator=generator, guidance_scale=8.0, num_images_per_prompt=8
).images

counter = 0

for i in images:
    print(type(i))
    i.save("./images/parallel/astronaut_rides_horse{}.png".format(counter))
    counter += 1
    
