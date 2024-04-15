import os
from itertools import product

model = "lcm-pixart-alpha"
steps = [4, 6]
batch_size = 8
precisions = ["fp16", "fp32"]

initial_latent_num = 1

def _subprocess_run(cmd):
    import subprocess

    try:
        print(f"Running: {cmd}")
        subprocess.run(cmd, check=True, shell=True)
    except subprocess.CalledProcessError:
        print("Command failed, exiting.")
        exit(1)

for step, precision in product(steps, precisions):

    base_cmd = f"python experiment.py --model={model} --num_inference_steps={step} --wandb_benchmark_name='pixart-benchmark-rerun-cuda' --benchmark_mode --precision={precision}"

    print("Testing baseline")
    cmd = base_cmd + f" --comment='baseline' --num_images_per_prompt={batch_size}"
    _subprocess_run(cmd)

    base_cmd += f" --max_batch_size={batch_size} --num_images_per_prompt={initial_latent_num}"

    # print("Testing cloning only")
    # cmd = base_cmd + f" --callback='cloning_only'"
    # _subprocess_run(cmd)

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

    # print("Testing away_from_average callback")
    # noise_const = 0.1
    # cmd = base_cmd + f" --callback='away_from_average' --noise_const={noise_const}"
    # _subprocess_run(cmd)

    # print("Testing modify_colors callback")
    # cmd = base_cmd + f" --callback='modify_colors'" 
    # _subprocess_run(cmd)

    # print("Testing away_plus_colors callback")
    # cmd = base_cmd + f" --callback='away_plus_colors'" 
    # _subprocess_run(cmd)
