#!/bin/bash

# Array of model names
# models=("lcm-sdxl" "lcm-sd-controlnet" "lcm-dreamshaper" "lcm-pixart-alpha")
models=("lcm-sd-controlnet")

# Loop over the model names
for model in "${models[@]}"; do
    # Run the experiment.py script with the current model
    echo "Running experiment with model: $model"
    python experiment.py --benchmark_mode --model="$model" --device="cuda" --precision="fp16" \
        --num_inference_steps=4 --num_images_per_prompt=8 --comment="baseline"
    python experiment.py --benchmark_mode --model="$model" --device="cuda" --precision="fp32" \
        --num_inference_steps=4 --num_images_per_prompt=8 --comment="baseline"
    python experiment.py --benchmark_mode --model="$model" --device="cpu" --precision="fp32" \
        --num_inference_steps=4 --num_images_per_prompt=8 --comment="baseline"
    python experiment.py --benchmark_mode --model="$model" --device="cuda" --precision="fp16" \
        --num_inference_steps=6 --num_images_per_prompt=8 --comment="baseline"
    python experiment.py --benchmark_mode --model="$model" --device="cuda" --precision="fp32" \
        --num_inference_steps=6 --num_images_per_prompt=8 --comment="baseline"
    python experiment.py --benchmark_mode --model="$model" --device="cpu" --precision="fp32" \
        --num_inference_steps=6 --num_images_per_prompt=8 --comment="baseline"
done
