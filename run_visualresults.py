import os
from itertools import product

steps = [4, 6]
# models = ["lcm-sdxl", "lcm-sd-controlnet", "lcm-dreamshaper", "lcm-pixart-alpha"]
# models = ["lcm-sdxl"]
models = ["lcm-sd-controlnet", "lcm-dreamshaper", "lcm-pixart-alpha"]
batch_sizes = [8]
# initial_latents = [1, 2] Run only with 1 for now. Do an experiment with away_from_average and different starting latents in another file.
initial_latents = [1]

for step, model, batch_size, initial_latent_num in product(steps, models, batch_sizes, initial_latents):

    if batch_size > (2 ** (step - 1)):
        continue

    base_cmd = f"python experiment.py --use_wandb --model={model} --num_inference_steps={step} --max_batch_size={batch_size} --num_images_per_prompt={initial_latent_num}"

    print("Testing cloning only")
    cmd = base_cmd + f" --callback='cloning_only'"
    print(f"Running: {cmd}")
    os.system(cmd)

    # # Reduce the guidance for lcm-sdxl for the remaining tests
    # if model == "lcm-sdxl":
    #     base_cmd += " --guidance_scale=0.5"
    #     base_cmd += " --comment='reduced_guidance'"

    #     print("Testing cloning only with reduced guidance")
    #     cmd = base_cmd + f" --callback='cloning_only'" 
    #     print(f"Running: {cmd}")
    #     os.system(cmd)

    print("Testing random_noise callback")
    for scalar_range in [0.1, 0.3, 0.5]:
        cmd = base_cmd + f" --callback='random_noise' --scalar_range={scalar_range}"
        print(f"Running: {cmd}")
        os.system(cmd)

    print("Testing away_from_average callback")
    for noise_const in [0.1, 0.2, 0.25]:
        cmd = base_cmd + f" --callback='away_from_average' --noise_const={noise_const}"
        print(f"Running: {cmd}")
        os.system(cmd)

    print("Testing modify_colors callback")
    cmd = base_cmd + f" --callback='modify_colors'" 
    print(f"Running: {cmd}")
    os.system(cmd)
