from diffusers import DiffusionPipeline
from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img import LatentConsistencyModelPipeline
import torch

pipe = LatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")

# To save GPU memory, torch.float16 can be used, but it may compromise image quality.
pipe.to(torch_device="cuda", torch_dtype=torch.float16)
print(type(pipe))

prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

# Can be set to 1~50 steps. LCM support fast inference even <= 4 steps. Recommend: 1~8 steps.
num_inference_steps = 4 

images = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=8.0, lcm_origin_steps=4, output_type="pil", num_images_per_prompt=8).images

counter = 0

for i in images:
    print(type(i))
    i.save("./images/parallel/astronaut_rides_horse{}.png".format(counter))
    counter += 1
    
