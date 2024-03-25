import abc
import datetime
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, List, Type

import PIL
import simple_parsing
import torch
from natsort import natsorted
from simple_parsing import Serializable
from simple_parsing.helpers import list_field
from simple_parsing.helpers.serialization import save as sp_save
from torch.distributions import Uniform
from tqdm import tqdm

import wandb
from diffusers import DiffusionPipeline
from diffusers.utils.testing_utils import enable_full_determinism


def _save_images_helper(images, output_dir: Path):
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(pbar := tqdm(images)):
        dst = output_dir / f"images_{i}.png"
        pbar.set_description(f"Saving image {i} to {dst}")
        img.save(dst)

def sp_target(lambda_target):
    """Auto-create the necessary simple_parsing.field() call for _target value, with a given class.

    Notes:
    - cmd=False is to prevent the field from being added to the command line arguments.
    - to_dict=False is to prevent the field from being added to the dictionary when serializing the object.
    """
    return simple_parsing.field(
        default_factory=lambda_target, cmd=False, to_dict=False
    )

@dataclass
class BaseConfig(Serializable):
    """Config class for instantiating the class specified in the _target attribute.

    Equipped with a .setup() method that returns the instantiated object using the config.
    """

    _target: Type = simple_parsing.field(cmd=False, to_dict=False)

    def setup(self, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(self, **kwargs)

# TODO: Keeping things simple by having everything in the same file. Can be split into multiple files later.
# ===========================================================
# ======================== MODELS ===========================
# ===========================================================

@dataclass
class DiffusionModelConfig(BaseConfig):
    num_inference_steps: int
    guidance_scale: float
    prompt: str
    num_images_per_prompt: int
    save_inter_images: bool = False
    """Whether to save intermediate images or not (before modifying latents)."""
    save_inter_images_modified: bool = False
    """Whether to save intermediate images after modifying the latents or not."""

@dataclass
class DiffusionModel:
    config: DiffusionModelConfig

    def run_inference(self, callback_func, output_dir: Path) -> List[PIL.Image.Image]:
        # Wrap the callback function to save intermediate images if necessary
        if self.config.save_inter_images or self.config.save_inter_images_modified:
            callback_func = self.wrap_callback_func(callback_func, output_dir)

        return self._run_inference(callback_func)

    @abc.abstractmethod
    def _run_inference(self, callback_func) -> List[PIL.Image.Image]:
        raise NotImplementedError

    def wrap_callback_func(self, org_callback: callable, output_dir: Path):
        return partial(
            self._callback_wrapper,
            org_callback=org_callback,
            save_images_func=self._decode_latents_and_save_images,
            save_inter_images=self.config.save_inter_images,
            save_inter_images_modified=self.config.save_inter_images_modified,
            output_dir=output_dir
        )

    @staticmethod
    def _callback_wrapper(
        pipeline: DiffusionPipeline, i: int, t: int, callback_kwargs: dict,
        org_callback: callable,
        save_images_func: callable,
        save_inter_images: bool,
        save_inter_images_modified: bool,
        output_dir: Path
    ) -> dict:

        if save_inter_images:
            save_images_func(pipeline, callback_kwargs, output_dir / f"step_{i}")

        outputs = org_callback(pipeline, i, t, callback_kwargs)

        if save_inter_images_modified:
            save_images_func(pipeline, outputs, output_dir / f"step_{i}_modified")

        return outputs

    @staticmethod
    @abc.abstractmethod
    def _decode_latents_and_save_images(pipe, model_outputs, output_dir: Path):
        raise NotImplementedError

@dataclass
class LCM_SDXL_DiffusionModelConfig(DiffusionModelConfig):
    _target: Type = sp_target(lambda: LCM_SDXL_DiffusionModel)
    num_inference_steps: int = 4
    guidance_scale: float = 8.0
    prompt: str = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
    num_images_per_prompt: int = 1

@dataclass
class LCM_SDXL_DiffusionModel(DiffusionModel):
    config: LCM_SDXL_DiffusionModelConfig

    def _run_inference(self, callback_func) -> List[PIL.Image.Image]:
        from diffusers import LCMScheduler, UNet2DConditionModel, StableDiffusionXLPipeline
        # from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineFast

        unet = UNet2DConditionModel.from_pretrained(
            "latent-consistency/lcm-sdxl",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16
        ).to("cuda")
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        pipe.enable_vae_slicing()

        # TODO: Apparently, LCM needs at least two steps to generate (sufficiently - at least
        # easily noticeable) different images after perturbing the latents?? (hence num_inference_steps=4 -> 5
        # NOTE: If num_inference_steps=4, you will notice that images_0 & images_4,
        # images_1 & images_5, images_2 & are (almost) exactly the same?
        images = pipe(
            prompt=self.config.prompt,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            num_images_per_prompt=self.config.num_images_per_prompt,
            callback_on_step_end=callback_func,
            callback_on_step_end_tensor_inputs=pipe._callback_tensor_inputs
        ).images

        return images

    @staticmethod
    def _decode_latents_and_save_images(pipe, model_outputs, output_dir: Path):
        latents = model_outputs["latents"]
        # Copied directly from pipeline_stable_diffusion_xl.py
        # pipe.enable_vae_slicing() should not be necessary as it should have already been applied.

        # make sure the VAE is in float32 mode, as it overflows in float16
        self = pipe
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        if needs_upcasting:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        # unscale/denormalize the latents
        # denormalize with the mean and std if available and not None
        has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
        has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
        if has_latents_mean and has_latents_std:
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            )
            latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
        else:
            latents = latents / self.vae.config.scaling_factor

        image = self.vae.decode(latents, return_dict=False)[0]

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        image = self.image_processor.postprocess(image, output_type="pil")

        # Save the images to the output dir
        _save_images_helper(image, output_dir)

@dataclass
class LCM_Dreamshaper_DiffusionModelConfig(DiffusionModelConfig):
    _target: Type = sp_target(lambda: LCM_Dreamshaper_DiffusionModel)
    num_inference_steps: int = 4
    guidance_scale: float = 8.0
    prompt: str = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
    num_images_per_prompt: int = 1

    def __post_init__(self):
        # LCM_Dreamshaper_DiffusionModel does not support saving
        # intermediate images after modification
        # as it uses outputs["denoised"] to generate images (which we do not modify)
        # (i.e. it's pointless as it would be same image as the original before modification)
        if self.save_inter_images_modified:
            print("NOTE: LCM_Dreamshaper_DiffusionModel does not support saving intermediate images after modification.")
            self.save_inter_images_modified = False

@dataclass
class LCM_Dreamshaper_DiffusionModel(DiffusionModel):

    def _run_inference(self, callback_func) -> List[PIL.Image.Image]:
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
        # To save GPU memory, torch.float16 can be used, but it may compromise image quality.
        pipe.to(torch_device="cuda", torch_dtype=torch.float32)

        # Can be set to 1~50 steps. LCM support fast inference even <= 4 steps. Recommend: 1~8 steps.
        images = pipe(
            prompt=self.config.prompt,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            num_images_per_prompt=self.config.num_images_per_prompt,
            callback_on_step_end=callback_func,
            callback_on_step_end_tensor_inputs=pipe._callback_tensor_inputs
        ).images

        return images

    @staticmethod
    def _decode_latents_and_save_images(pipe, model_outputs, output_dir: Path):
        denoised = model_outputs["denoised"]

        # Taken from src/diffusers/pipelines/latent_consistency_models/pipeline_latent_consistency_text2img.py
        self = pipe

        image = self.vae.decode(denoised / self.vae.config.scaling_factor, return_dict=False)[0]
        do_denormalize = [True] * image.shape[0]

        image = self.image_processor.postprocess(image, output_type="pil", do_denormalize=do_denormalize)

        # Save the images to the output dir
        _save_images_helper(image, output_dir)

# ===========================================================
# ======================== CALLBACKS ========================
# ===========================================================

@dataclass
class CallbackConfig(BaseConfig):
    pass

class Callback:
    config: CallbackConfig

    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def callback_func(pipeline: DiffusionPipeline, i: int, t: int, callback_kwargs: dict, *args, **kwargs) -> dict:
        """ Callback functions may have additional arguments/keyword arguments (that are specified
        when wrapped in partial) but MUST take the first four arguments.

        Args:
            pipeline (object): The pipeline object (is simply 'self' in the pipeline class).
            i (int): the current iteration.
            t (float): The current time.
            callback_kwargs (dict): all inputs passed back from the pipeline
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: The updated callback_kwargs.
        """
        raise NotImplementedError


@dataclass
class IdentifyInputsCallbackConfig(CallbackConfig):
    _target: Type = sp_target(lambda: IdentifyInputsCallback)

@dataclass
class IdentifyInputsCallback(Callback):
    config: IdentifyInputsCallbackConfig

    def __call__(self):
        return partial(self.callback_func)

    @staticmethod
    def callback_func(pipeline: DiffusionPipeline, i: int, t: int, callback_kwargs: dict):
        for key, value in callback_kwargs.items():
            if value is not None:
                print(f"End of iteration {i}: {key} shape is {value.shape}")

        return {}


@dataclass(kw_only=True)
class RandomNoiseCallbackConfig(CallbackConfig):
    _target: Type = sp_target(lambda: RandomNoiseCallback)
    max_batch_size: int
    """Maximum batch size to reach with cloning."""
    cloning_mode: str = simple_parsing.choice(
        "doubling", "linear",
        default="doubling"
    )
    """Rate of which to clone the batch elements."""
    scalar_range: float = 0.1
    """Range of the scalar to multiply the noise with. Sampled with Uniform(-scalar_range, scalar_range)."""
    skip_modify: bool = False
    """Whether to skip modifying the latents or not."""
    steps: List[int] = None
    """List of steps to apply the callback on. Defaults to all steps until max_batch_size is reached."""
    to_modify: List[str] = list_field("latents")
    """List of keys to modify. Defaults to latents in the callback_kwargs."""


@dataclass
class RandomNoiseCallback(Callback):
    config: RandomNoiseCallbackConfig

    def __call__(self):
        return partial(
            self.callback_func,
            max_batch_size=self.config.max_batch_size,
            skip_modify=self.config.skip_modify,
            cloning_mode=self.config.cloning_mode,
            scalar_range=self.config.scalar_range,
            steps=self.config.steps,
            to_modify=self.config.to_modify
        )

    @staticmethod
    def callback_func(
        pipeline: DiffusionPipeline, i: int, t: int, callback_kwargs: dict,
        max_batch_size: int,
        skip_modify: bool,
        cloning_mode: str,
        scalar_range: float,
        steps: List[int] = None,
        to_modify: List[str] = None,
    ):
        latents = callback_kwargs["latents"]

        batch_size = latents.shape[0]

        # Do nothing if the current iteration is not in the steps list
        if steps is not None and i not in steps:
            return callback_kwargs

        outputs = {}
        if batch_size < max_batch_size:
            if cloning_mode == "doubling":
                print(f"End of iteration {i}, timestep {t}: Current batch size is {batch_size}, doubling the batch size to {batch_size * 2}")
                outputs = {key: torch.cat([value] * 2) for key, value in callback_kwargs.items() if value is not None}
                mask = torch.Tensor([False] * batch_size + [True] * batch_size).bool()
            elif cloning_mode == "linear":
                print(f"End of iteration {i}, timestep {t}: Current batch size is {batch_size}, cloning the batch size to {batch_size + 1}")
                outputs = {key: torch.cat([value] * (max_batch_size // batch_size)) for key, value in callback_kwargs.items() if value is not None}
                mask = torch.Tensor([True] * 2 + [False] * (batch_size - 1)).bool()

            if not skip_modify:
                for key in to_modify:
                    tensor = outputs[key]
                    mask_batch_size = tensor[mask].shape[0]

                    scalar = Uniform(-scalar_range, scalar_range).sample([mask_batch_size]).to(tensor.device, dtype=tensor.dtype)
                    print(f"End of iteration {i}, timestep {t}: Peturbing the {key} in batch dimension with {mask=} using {scalar=}")
                    tensor[mask] = (
                        tensor[mask] +
                        torch.randn_like(tensor[mask]) * scalar.reshape([mask_batch_size] + [1] * (tensor.ndim - 1))
                    )

                    outputs[key] = tensor

        return outputs

# ===========================================================
# ======================== MAIN =============================
# ===========================================================

@dataclass
class ExperimentConfig(Serializable):
    seed: int = 0
    """Random seed to use for reproducibility."""
    use_wandb: bool = simple_parsing.flag(default=False, negative_option="--disable_wandb")
    """Whether to use wandb or not."""
    wandb_offline: bool = simple_parsing.flag(default=False, negative_option="--no_wandb_offline")
    """Whether to use wandb in offline mode or not."""
    use_full_determinism: bool = simple_parsing.flag(default=False, negative_option="--no_full_determinism")
    """Whether to use full determinism mode or not."""
    output_dir: Path = Path("./experiment_runs")
    """Root folder to where indiviudal experiment runs will be saved."""
    model: DiffusionModelConfig = simple_parsing.subgroups(
        {
            "lcm-sdxl": LCM_SDXL_DiffusionModelConfig,
            "lcm_dreamshaper": LCM_Dreamshaper_DiffusionModelConfig,
        },
        default="lcm-sdxl",
    )
    """Model configuration."""
    callback: CallbackConfig = simple_parsing.subgroups(
        {
            "identify_inputs": IdentifyInputsCallbackConfig,
            "random_noise": RandomNoiseCallbackConfig,
        },
        default="identify_inputs",
    )
    """Callback configuration."""
    comment: str = ""
    """Additional comment to add to the experiment ID (used mainly to distinguish wandb runs)."""
    # TODO: Implement a benchmark/timing mode just to gather runtime statistics.

def parse_args():
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(ExperimentConfig, "exp_config")

    args = parser.parse_args()
    return args

def main():
    def _should_skip_saving(exp_config):
        if isinstance(exp_config.callback, IdentifyInputsCallbackConfig):
            return True
        # Add more conditions later
        return False

    args = parse_args()
    exp_config = args.exp_config

    # Individual runs will be identified by a timestamp + model + callback
    _timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    _model_name = args.subgroups["exp_config.model"]
    _callback_name = args.subgroups["exp_config.callback"]
    _comment = "_" + exp_config.comment if exp_config.comment != "" else ""
    exp_id = f"{_timestamp}_{_model_name}_{_callback_name}{_comment}"
    exp_dir = exp_config.output_dir / exp_id

    # Skip all file-saving as well as Weights&Biases calls if we're just identifying inputs.
    skip_saving = _should_skip_saving(exp_config)
    if skip_saving:
        exp_config.use_wandb = False
        # Force off saving intermediate images
        exp_config.model.save_inter_images = False
        exp_config.model.save_inter_images_modified = False

    if exp_config.use_wandb:
        # Start wandb
        wandb.init(
            project="diffusion",
            entity="kwand_jalnyn_csc2231", # Make sure you are logged in to wandb and have access to this team
            name=exp_id,
            config=exp_config.to_dict(save_dc_types=True),
            mode="offline" if exp_config.wandb_offline else "online",
        )

    # Construct all the objects from the configs
    callback = exp_config.callback.setup()
    callback_func = callback()
    model = exp_config.model.setup()

    if exp_config.use_full_determinism:
        enable_full_determinism(exp_config.seed)
    else:
        torch.manual_seed(exp_config.seed)

    with torch.inference_mode():
        images = model.run_inference(callback_func, exp_dir)

    # Create dirs, save config, and save images only at the end - to prevent unnecessary files
    # from being saved in case of early termination (crashes)
    if not skip_saving:
        # Create the output directory
        os.makedirs(exp_dir, exist_ok=True)
        print("Saving files to", exp_dir)

        # Serialize and save the config
        sp_save(exp_config, exp_dir / "config.yaml", save_dc_types=True)
        print("Saved experiment config to", exp_dir / "config.yaml")

        # Save all the images
        img_output_dir = exp_dir / "final"
        os.makedirs(img_output_dir, exist_ok=True)
        print("Saving images to", img_output_dir)
        _save_images_helper(images, img_output_dir)

    # Save the image results to a table and finish the wandb run
    if exp_config.use_wandb:
        columns = ["exp_id", "model_config", "callback_config", "stage"]
        for i in range(exp_config.callback.max_batch_size):
            columns.append(f"images_{i}")

        data = []
        subdirs = [d for d in Path(exp_dir).iterdir() if d.is_dir() and "step" in d.name]
        natsorted_subdirs = natsorted(subdirs, key=lambda x: x.name)
        for folder in natsorted_subdirs:
            _row = (
                [exp_id, str(exp_config.model), str(exp_config.callback), folder.name]
                + [wandb.Image(str(img)) for img in folder.iterdir()]
            )
            data.append(_row)

        # Append the final images at the end to ensure it shows up as the last row
        _row = (
            [exp_id, str(exp_config.model), str(exp_config.callback), "final"]
            + [wandb.Image(str(img)) for img in img_output_dir.iterdir()]
        )
        data.append(_row)

        # Needed to pad the rows to ensure all rows have the same length
        def _pad_row(row, max_length):
            return row + [None] * (max_length - len(row))

        # Do padding to ensure all rows have the same length
        data = [_pad_row(row, len(columns)) for row in data]

        results_table = wandb.Table(data=data, columns=columns)
        wandb.log({"results": results_table})

        wandb.finish()

if __name__ == "__main__":
    main()
