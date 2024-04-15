import os
from itertools import product

# steps = [4, 6]
# # models = ["lcm-sdxl", "lcm-sd-controlnet", "lcm-dreamshaper", "lcm-pixart-alpha"]
# # models = ["lcm-sdxl"]
# models = ["lcm-sd-controlnet", "lcm-dreamshaper", "lcm-pixart-alpha"]
# batch_sizes = [8]
# # initial_latents = [1, 2] Run only with 1 for now. Do an experiment with away_from_average and different starting latents in another file.
# initial_latents = [1]
model = "lcm-pixart-alpha"
step = 6
batch_size = 8
prompts = [
    # "an astronaut sitting in a diner, eating fries, cinematic, analog film",
    # "Pirate ship trapped in a cosmic maelstrom nebula, rendered in cosmic beach whirlpool engine, volumetric lighting, spectacular, ambient lights, light pollution, cinematic atmosphere, art nouveau style, illustration art artwork by SenseiJaye, intricate detail.",
    "stars, water, brilliantly, gorgeous large scale scene, a little girl, in the style of dreamy realism, light gold and amber, blue and pink, brilliantly illuminated in the background.",
    "beautiful lady, freckles, big smile, blue eyes, short ginger hair, dark makeup, wearing a floral blue vest top, soft light, dark grey background",
    # "Spectacular Tiny World in the Transparent Jar On the Table, interior of the Great Hall, Elaborate, Carved Architecture, Anatomy, Symetrical, Geometric and Parameteric Details, Precision Flat line Details, Pattern, Dark fantasy, Dark errie mood and ineffably mysterious mood, Technical design, Intricate Ultra Detail, Ornate Detail, Stylized and Futuristic and Biomorphic Details, Architectural Concept, Low contrast Details, Cinematic Lighting, 8k, by moebius, Fullshot, Epic, Fullshot, Octane render, Unreal ,Photorealistic, Hyperrealism",
    # "anthropomorphic profile of the white snow owl Crystal priestess , art deco painting, pretty and expressive eyes, ornate costume, mythical, ethereal, intricate, elaborate, hyperrealism, hyper detailed, 3D, 8K, Ultra Realistic, high octane, ultra resolution, amazing detail, perfection, In frame, photorealistic, cinematic lighting, visual clarity, shading , Lumen Reflections, Super-Resolution, gigapixel, color grading, retouch, enhanced, PBR, Blender, V-ray, Procreate, zBrush, Unreal Engine 5, cinematic, volumetric, dramatic, neon lighting, wide angle lens ,no digital painting blur",
    # "The parametric hotel lobby is a sleek and modern space with plenty of natural light. The lobby is spacious and open with a variety of seating options. The front desk is a sleek white counter with a parametric design. The walls are a light blue color with parametric patterns. The floor is a light wood color with a parametric design. There are plenty of plants and flowers throughout the space. The overall effect is a calm and relaxing space. occlusion, moody, sunset, concept art, octane rendering, 8k, highly detailed, concept art, highly detailed, beautiful scenery, cinematic, beautiful light, hyperreal, octane render, hdr, long exposure, 8K, realistic, fog, moody, fire and explosions, smoke, 50mm f2.8"

]
initial_latent_num = 1

def _subprocess_run(cmd):
    import subprocess

    try:
        print(f"Running: {cmd}")
        subprocess.run(cmd, check=True, shell=True)
    except subprocess.CalledProcessError:
        print("Command failed, exiting.")
        exit(1)

for prompt in prompts:

    base_cmd = f"python experiment.py --use_wandb --model={model} --num_inference_steps={step} --wandb_name='visualresults-pixart'"

    if prompt != "original":
        base_cmd += f" --prompt='{prompt}'"

    # print("Testing baseline")
    # cmd = base_cmd + f" --comment='baseline' --num_images_per_prompt={batch_size}"
    # _subprocess_run(cmd)

    base_cmd += f" --max_batch_size={batch_size} --num_images_per_prompt={initial_latent_num}"

    print("Testing cloning only")
    cmd = base_cmd + f" --callback='cloning_only'"
    _subprocess_run(cmd)

    # # Reduce the guidance for lcm-sdxl for the remaining tests
    # if model == "lcm-sdxl":
    #     base_cmd += " --guidance_scale=0.5"
    #     base_cmd += " --comment='reduced_guidance'"

    #     print("Testing cloning only with reduced guidance")
    #     cmd = base_cmd + f" --callback='cloning_only'" 
    #     print(f"Running: {cmd}")
    #     os.system(cmd)

    print("Testing random_noise callback")
    scalar_range = 0.3
    cmd = base_cmd + f" --callback='random_noise' --scalar_range={scalar_range}"
    _subprocess_run(cmd)

    print("Testing away_from_average callback")
    noise_const = 0.1
    cmd = base_cmd + f" --callback='away_from_average' --noise_const={noise_const}"
    _subprocess_run(cmd)

    # print("Testing modify_colors callback")
    # cmd = base_cmd + f" --callback='modify_colors'" 
    # _subprocess_run(cmd)

    # print("Testing away_plus_colors callback")
    # cmd = base_cmd + f" --callback='away_plus_colors'" 
    # _subprocess_run(cmd)
