import abc
import datetime
import os
import timeit
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, List, Type

import numpy as np
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
    return simple_parsing.field(default_factory=lambda_target, cmd=False, to_dict=False)


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


@dataclass(kw_only=True)
class DiffusionModelConfig(BaseConfig):
    num_inference_steps: int
    guidance_scale: float
    prompt: str
    num_images_per_prompt: int
    save_inter_images: bool = False
    """Whether to save intermediate images or not (before modifying latents)."""
    save_inter_images_modified: bool = False
    """Whether to save intermediate images after modifying the latents or not."""
    device: str = simple_parsing.choice("cuda", "cpu", default="cuda")


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
            output_dir=output_dir,
        )

    @staticmethod
    def _callback_wrapper(
        pipeline: DiffusionPipeline,
        i: int,
        t: int,
        callback_kwargs: dict,
        org_callback: callable,
        save_images_func: callable,
        save_inter_images: bool,
        save_inter_images_modified: bool,
        output_dir: Path,
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
    precision: str = simple_parsing.choice("fp16", "fp32", default="fp16")

@dataclass
class LCM_SDXL_DiffusionModel(DiffusionModel):
    config: LCM_SDXL_DiffusionModelConfig

    def _run_inference(self, callback_func) -> List[PIL.Image.Image]:
        from diffusers import LCMScheduler, UNet2DConditionModel
        from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineFast

        if self.config.precision == "fp32":
            unet = UNet2DConditionModel.from_pretrained(
                "latent-consistency/lcm-sdxl",
                torch_dtype=torch.float32,
                # variant="fp16",
            )
            pipe = StableDiffusionXLPipelineFast.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float32
            )
        elif self.config.precision == "fp16":
            unet = UNet2DConditionModel.from_pretrained(
                "latent-consistency/lcm-sdxl",
                torch_dtype=torch.float16,
                variant="fp16",
            )
            pipe = StableDiffusionXLPipelineFast.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16
            )
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        if self.config.device == "cuda":
            pipe.to("cuda")
        elif self.config.device == "cpu":
            pipe.to("cpu")


        pipe.enable_model_cpu_offload() # This does pipe.to("cuda") inside

        # Always enable slicing (one-by-one latents decoding) no matter the device to ensure fairness in benchmarking
        pipe.enable_vae_slicing()

        images = pipe(
            prompt=self.config.prompt,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            num_images_per_prompt=self.config.num_images_per_prompt,
            callback_on_step_end=callback_func,
            callback_on_step_end_tensor_inputs=pipe._callback_tensor_inputs,
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
            latents_std = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
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
    guidance_scale: float = 0.5
    prompt: str = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
    # prompt: str = "(masterpiece), (extremely intricate:1.3),, (realistic), portrait of a girl, the most beautiful in the world, (medieval armor), metal reflections, upper body, outdoors, intense sunlight, far away castle, professional photograph of a stunning woman detailed, sharp focus, dramatic, award winning, cinematic lighting, octane render, unreal engine, volumetrics dtx, (film grain, bokeh, blurry foreground, blurry background), crest on chest"
    # prompt: str = "close up Portrait photo of muscular bearded guy in a worn mech suit, ((light bokeh)), intricate, (steel metal [rust]), elegant, sharp focus, photo by greg rutkowski, soft lighting, vibrant colors, masterpiece, ((streets)), detailed face"
    num_images_per_prompt: int = 1
    precision: str = simple_parsing.choice("fp16", "fp32", default="fp16")

    def __post_init__(self):
        # LCM_Dreamshaper_DiffusionModel does not support saving
        # intermediate images after modification
        # as it uses outputs["denoised"] to generate images (which we do not modify)
        # (i.e. it's pointless as it would be same image as the original before modification)
        if self.save_inter_images_modified:
            print(
                "NOTE: LCM_Dreamshaper_DiffusionModel does not support saving intermediate images after modification."
            )
            self.save_inter_images_modified = False


@dataclass
class LCM_Dreamshaper_DiffusionModel(DiffusionModel):
    config: LCM_Dreamshaper_DiffusionModelConfig

    def _run_inference(self, callback_func) -> List[PIL.Image.Image]:


        from diffusers import AutoPipelineForText2Image, LCMScheduler

        if self.config.precision == "fp16":
            pipe = AutoPipelineForText2Image.from_pretrained('lykon/dreamshaper-7', torch_dtype=torch.float16, variant="fp16")
        elif self.config.precision == "fp32":
            pipe = AutoPipelineForText2Image.from_pretrained('lykon/dreamshaper-7', torch_dtype=torch.float32)

        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

        pipe = pipe.to(self.config.device)

        pipe.enable_vae_slicing()

        # Can be set to 1~50 steps. LCM support fast inference even <= 4 steps. Recommend: 1~8 steps.
        images = pipe(
            prompt=self.config.prompt,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            num_images_per_prompt=self.config.num_images_per_prompt,
            callback_on_step_end=callback_func,
            callback_on_step_end_tensor_inputs=pipe._callback_tensor_inputs,
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


# @dataclass
# class SDXL_ControlNetConfig(DiffusionModelConfig):
#     _target: Type = sp_target(lambda: SDXL_ControlNet)
#     prompt: str = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
#     negative_prompt: str = "low quality, bad quality, sketches"
#     controlnet_image: str = "https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
#     num_inference_steps: int = 50
#     guidance_scale: float = 5.0
#     num_images_per_prompt: int = 1

# @dataclass
# class SDXL_ControlNet(DiffusionModel):
#     config: SDXL_ControlNetConfig

#     def _run_inference(self, callback_func) -> List[PIL.Image.Image]:
#         # !pip install opencv-python transformers accelerate
#         import cv2
#         import numpy as np
#         from PIL import Image

#         from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetPipeline
#         from diffusers.utils import load_image

#         # download an image
#         image = load_image(
#             self.config.controlnet_image
#         )

#         # initialize the models and pipeline
#         controlnet_conditioning_scale = 0.5  # recommended for good generalization
#         controlnet = ControlNetModel.from_pretrained(
#             "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
#         )
#         vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
#         pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
#             "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16
#         )
#         pipe.enable_model_cpu_offload()

#         # get canny image
#         image = np.array(image)
#         image = cv2.Canny(image, 100, 200)
#         image = image[:, :, None]
#         image = np.concatenate([image, image, image], axis=2)
#         canny_image = Image.fromarray(image)

#         # generate image
#         images = pipe(
#             self.config.prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=canny_image,
#             num_images_per_prompt=self.config.num_images_per_prompt,
#             guidance_scale=self.config.guidance_scale,
#             negative_prompt=self.config.negative_prompt,
#             num_inference_steps=self.config.num_inference_steps,
#             callback_on_step_end=callback_func,
#             callback_on_step_end_tensor_inputs=pipe._callback_tensor_inputs,
#         ).images

#         return images

@dataclass
class SD_1_5_LCM_ControlNetConfig(DiffusionModelConfig):
    _target: Type = sp_target(lambda: SD_1_5_LCM_ControlNet)
    prompt: str = "the mona lisa"
    controlnet_image: str = "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
    num_inference_steps: int = 4
    guidance_scale: float = 0.5
    controlnet_conditioning_scale: float = 0.8
    cross_attention_scale: float = 1.0
    num_images_per_prompt: int = 1
    precision: str = simple_parsing.choice("fp16", "fp32", default="fp16")

@dataclass
class SD_1_5_LCM_ControlNet(DiffusionModel):
    config: SD_1_5_LCM_ControlNetConfig

    def _run_inference(self, callback_func) -> List[PIL.Image.Image]:
        import cv2
        import numpy as np
        from PIL import Image

        from diffusers import ControlNetModel, LCMScheduler, StableDiffusionControlNetPipeline
        from diffusers.utils import load_image

        image = load_image(
            self.config.controlnet_image
        ).resize((512, 512))

        image = np.array(image)

        low_threshold = 100
        high_threshold = 200

        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)

        if self.config.precision == "fp16":
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                # "runwayml/stable-diffusion-v1-5",
                "lykon/dreamshaper-7",
                controlnet=controlnet,
                torch_dtype=torch.float16,
                variant="fp16"
            )
        elif self.config.precision == "fp32":
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float32)
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                # "runwayml/stable-diffusion-v1-5",
                "lykon/dreamshaper-7",
                controlnet=controlnet,
                torch_dtype=torch.float32,
            )

        if self.config.device == "cuda":
            pipe.to("cuda")
        elif self.config.device == "cpu":
            pipe.to("cpu")

        # set scheduler
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        # load LCM-LoRA
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

        images = pipe(
            self.config.prompt,
            image=canny_image,
            num_images_per_prompt=self.config.num_images_per_prompt,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            controlnet_conditioning_scale=self.config.controlnet_conditioning_scale,
            cross_attention_kwargs={"scale": self.config.cross_attention_scale},
            callback_on_step_end=callback_func,
            callback_on_step_end_tensor_inputs=pipe._callback_tensor_inputs,
        ).images

        return images

# @dataclass
# class SDXL_Config(DiffusionModelConfig):
#     _target: Type = sp_target(lambda: SDXL)
#     prompt: str = "a photo of an astronaut riding a horse on mars"
#     num_inference_steps: int = 50
#     guidance_scale: float = 5.0
#     num_images_per_prompt: int = 1

# @dataclass
# class SDXL(DiffusionModel):
#     config: SDXL_Config

#     def _run_inference(self, callback_func) -> List[PIL.Image.Image]:
#         import torch

#         from diffusers import StableDiffusionXLPipeline

#         pipe = StableDiffusionXLPipeline.from_pretrained(
#             "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
#         )
#         pipe.enable_vae_slicing()
#         pipe = pipe.to("cuda")

#         images = pipe(
#             self.config.prompt,
#             num_inference_steps=self.config.num_inference_steps,
#             num_images_per_prompt=self.config.num_images_per_prompt,
#             guidance_scale=self.config.guidance_scale,
#             callback_on_step_end=callback_func,
#             callback_on_step_end_tensor_inputs=pipe._callback_tensor_inputs,
#         ).images

#         return images

# @dataclass
# class SD2_Config(DiffusionModelConfig):
#     _target: Type = sp_target(lambda: SD2)
#     prompt: str = "High quality photo of an astronaut riding a horse in space"
#     num_inference_steps: int = 25
#     guidance_scale: float = 5.0
#     num_images_per_prompt: int = 1

# @dataclass
# class SD2(DiffusionModel):
#     config: SD2_Config

#     def _run_inference(self, callback_func) -> List[PIL.Image.Image]:
#         from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline

#         model_id = "stabilityai/stable-diffusion-2"

#         # Use the Euler scheduler here instead
#         # scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
#         pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
#         pipe = pipe.to("cuda")

#         images = pipe(
#             self.config.prompt,
#             num_inference_steps=self.config.num_inference_steps,
#             guidance_scale=self.config.guidance_scale,
#             num_images_per_prompt=self.config.num_images_per_prompt,
#             callback_on_step_end=callback_func,
#             callback_on_step_end_tensor_inputs=pipe._callback_tensor_inputs
#         ).images

#         return images

@dataclass
class LCMPixArtAlphaConfig(DiffusionModelConfig):
    _target: Type = sp_target(lambda: LCMPixArtAlpha)
    prompt: str = "A small cactus with a happy face in the Sahara desert."
    num_inference_steps: int = 4
    guidance_scale: float = 0
    num_images_per_prompt: int = 1
    precision: str = simple_parsing.choice("fp16", "fp32", default="fp16")

@dataclass
class LCMPixArtAlpha(DiffusionModel):
    config: LCMPixArtAlphaConfig

    def _run_inference(self, callback_func) -> List[PIL.Image.Image]:
        # Implementation of the inference logic for PixArtAlpha model
        from diffusers import AutoencoderKL, ConsistencyDecoderVAE, PixArtAlphaPipeline

        if self.config.precision == "fp16":
            torch_dtype = torch.float16
        elif self.config.precision == "fp32":
            torch_dtype = torch.float32
        pipe = PixArtAlphaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-LCM-XL-2-1024-MS",
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )

        # speed-up T5
        pipe.text_encoder.to_bettertransformer()

        # If use DALL-E 3 Consistency Decoder
        # pipe.vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder", torch_dtype=torch.float16)

        # If use SA-Solver sampler
        # from diffusion.sa_solver_diffusers import SASolverScheduler
        # pipe.scheduler = SASolverScheduler.from_config(pipe.scheduler.config, algorithm_type='data_prediction')

        # Enable memory optimizations.
        if self.config.device == "cuda" and self.config.precision == "fp16":
            pipe.enable_model_cpu_offload() # This does pipe.to("cuda") inside
        elif self.config.device == "cuda" and self.config.precision == "fp32":
            pipe.enable_model_cpu_offload() # This does pipe.to("cuda") inside
        elif self.config.device == "cpu":
            pipe.to("cpu")

        pipe.vae.enable_slicing() # I think for fairness (in benchmarking), we have to always decode latents one-by-one 

        images = pipe(
            self.config.prompt,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            num_images_per_prompt=self.config.num_images_per_prompt,
            callback_on_step_end=callback_func,
            callback_on_step_end_tensor_inputs=pipe._callback_tensor_inputs,
        ).images

        return images

# ===========================================================
# ======================== CALLBACKS ========================
# ===========================================================

def _doubling_cloning_mode(callback_kwargs, batch_size, i, t, do_print=False, do_mask=False):

    if do_print:
        print(
            f"End of iteration {i}, timestep {t}: Current batch size is {batch_size}, doubling the batch size to {batch_size * 2}"
        )
    # outputs = {key: torch.cat([value] * 2) for key, value in callback_kwargs.items() if value is not None}
    # Maybe torch.cat is not as efficient as torch.repeat?
    outputs = {key: value.repeat(2, *([1] * (value.ndim - 1))) for key, value in callback_kwargs.items() if value is not None}

    if do_mask:
        mask = torch.Tensor([False] * batch_size + [True] * batch_size).bool()
    else:
        mask = None

    return outputs, mask

def _linear_cloning_mode(callback_kwargs, batch_size, i, t, do_print=False, do_mask=False):
    if do_print:
        print(
            f"End of iteration {i}, timestep {t}: Current batch size is {batch_size}, cloning the batch size to {batch_size + 1}"
        )

    outputs = {}
    for key, value in callback_kwargs.items():
        if value is not None:
            org_batch_dim_shape = value.shape[0] // batch_size
            outputs[key] = torch.cat([ 
                    torch.cat([value[:org_batch_dim_shape]] * 2),
                    value[org_batch_dim_shape:]
            ])

    if do_mask:
        mask = torch.Tensor([False] + [True] + [False] * (batch_size - 1)).bool()
    else:
        mask = None

    return outputs, mask

@dataclass(kw_only=True)
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
        """Callback functions may have additional arguments/keyword arguments (that are specified
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

@dataclass(kw_only=True)
class DummyCallbackConfig(CallbackConfig):
    _target: Type = sp_target(lambda: DummyCallback)

@dataclass
class DummyCallback(Callback):
    config: DummyCallbackConfig

    def __call__(self):
        return partial(self.callback_func)

    @staticmethod
    def callback_func(pipeline: DiffusionPipeline, i: int, t: int, callback_kwargs: dict):
        return callback_kwargs

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
class CloningOnlyCallbackConfig(CallbackConfig):
    _target: Type = sp_target(lambda: CloningOnlyCallback)
    max_batch_size: int
    """Maximum batch size to reach with cloning."""
    cloning_mode: str = simple_parsing.choice("doubling", "linear", default="doubling")
    """Rate of which to clone the batch elements."""
    steps: List[int] = None
    """List of steps to apply the callback on. Defaults to all steps until max_batch_size is reached."""

@dataclass
class CloningOnlyCallback(Callback):
    config: CloningOnlyCallbackConfig

    def __call__(self):
        return partial(
            self.callback_func,
            max_batch_size=self.config.max_batch_size,
            cloning_mode=self.config.cloning_mode,
            steps=self.config.steps,
        )
    
    @staticmethod
    def callback_func(
        pipeline: DiffusionPipeline,
        i: int,
        t: int,
        callback_kwargs: dict,
        max_batch_size: int,
        cloning_mode: str,
        steps: List[int] = None
    ):
        latents = callback_kwargs["latents"]

        batch_size = latents.shape[0]

        # Do nothing if the current iteration is not in the steps list
        if steps is not None and i not in steps:
            return callback_kwargs

        outputs = {}
        if batch_size < max_batch_size:
            if cloning_mode == "doubling":
                outputs, _ = _doubling_cloning_mode(callback_kwargs, batch_size, i, t, do_print=False, do_mask=False)
            elif cloning_mode == "linear":
                outputs, _ = _linear_cloning_mode(callback_kwargs, batch_size, i, t, do_print=False, do_mask=False)

        return outputs


@dataclass(kw_only=True)
class RandomNoiseCallbackConfig(CallbackConfig):
    _target: Type = sp_target(lambda: RandomNoiseCallback)
    max_batch_size: int
    """Maximum batch size to reach with cloning."""
    # cloning_mode: str = simple_parsing.choice("doubling", "linear", default="doubling")
    cloning_mode: str = "doubling"
    """Rate of which to clone the batch elements."""
    scalar_range: float = 0.1
    """Range of the scalar to multiply the noise with. Sampled with Uniform(-scalar_range, scalar_range)."""
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
            cloning_mode=self.config.cloning_mode,
            scalar_range=self.config.scalar_range,
            steps=self.config.steps,
            to_modify=self.config.to_modify,
        )

    @staticmethod
    def callback_func(
        pipeline: DiffusionPipeline,
        i: int,
        t: int,
        callback_kwargs: dict,
        max_batch_size: int,
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
                outputs, mask = _doubling_cloning_mode(callback_kwargs, batch_size, i, t, do_print=False, do_mask=True)
            # elif cloning_mode == "linear":
            #     outputs, mask = _linear_cloning_mode(callback_kwargs, batch_size, i, t, do_print=False, do_mask=True)

            for key in to_modify:
                tensor = outputs[key]
                mask_batch_size = tensor[mask].shape[0]

                scalar = (
                    Uniform(-scalar_range, scalar_range)
                    .sample([mask_batch_size])
                    .to(tensor.device, dtype=tensor.dtype)
                )
                # print(
                #     f"End of iteration {i}, timestep {t}: Peturbing the {key} in batch dimension with {mask=} using {scalar=}"
                # )
                tensor[mask] = tensor[mask] + torch.randn_like(tensor[mask]) * scalar.reshape(
                    [mask_batch_size] + [1] * (tensor.ndim - 1)
                )

                outputs[key] = tensor

        return outputs

@dataclass(kw_only=True)
class AwayFromAverageCallbackConfig(CallbackConfig):
    _target: Type = sp_target(lambda: AwayFromAverageCallback)
    max_batch_size: int
    """Maximum batch size to reach with cloning."""
    noise_const: float = 0.2
    mode: str = simple_parsing.choice("amp_adv", "amp", default="amp_adv")
    cooldown_steps: int = 2
    max_steps: int = None
    """Will be set automatically to the number of inference steps in the model config."""

@dataclass
class AwayFromAverageCallback(Callback):
    config: AwayFromAverageCallbackConfig

    def __call__(self):
        return partial(
            self.callback_func,
            max_batch_size=self.config.max_batch_size,
            noise_const=self.config.noise_const,
            mode=self.config.mode,
            max_steps=self.config.max_steps,
            cooldown_steps=self.config.cooldown_steps
        )
    
    @staticmethod
    def callback_func(
        pipeline: DiffusionPipeline,
        i: int,
        t: int,
        callback_kwargs: dict,
        max_batch_size: int,
        noise_const: float,
        mode: str,
        max_steps: int,
        cooldown_steps: int
    ) -> dict:
        latents = callback_kwargs["latents"]

        batch_size = latents.shape[0]
        outputs = {}

        # TODO: Requires self.noise_pred to be set to the noise prediction tensor
        # in the pipeline. Currently, only our Fast pipeline file has it
        # (maybe forget about creating new files and just modify all the existing ones we test directly)

        if mode == "amp_adv":
            if batch_size > 1 and i < max_steps - cooldown_steps:
                # print(f"End of iteration {i}, timestep {t}: Applying new noise to latents under amp_adv mode.")
                avg_noise = pipeline.noise_pred.mean(dim=(1, 2, 3), keepdim=True)
                scheduler = pipeline.scheduler
                alph = scheduler.alphas_cumprod[t]
                beta = 1 - alph
                new_noise = (pipeline.noise_pred - avg_noise )*(noise_const*beta.sqrt())
                latents += new_noise
        elif mode == "amp":
            if batch_size > 1:
                # print(f"End of iteration {i}, timestep {t}: Applying noise diff to latents under amp mode.")
                noise_diff = pipeline.noise_pred[:batch_size//2] -  pipeline.noise_pred[batch_size//2:]
                # print( "Here: ",batch_size)
                latents[:batch_size//2] += noise_diff
                latents[batch_size//2:] -= noise_diff

        if batch_size < max_batch_size:
            outputs, _ = _doubling_cloning_mode(callback_kwargs, batch_size, i, t, do_print=False, do_mask=False)
            latents = outputs["latents"]

        if i < max_steps - cooldown_steps:
            # print(f"End of iteration {i}, timestep {t}: Applying additional noise to latents with torch.randn_like() scaled by {noise_const=}.")
            latents = latents + torch.randn_like(latents) * (noise_const)

        return outputs

@dataclass(kw_only=True)
class ModifyColorsCallbackConfig(CallbackConfig):
    _target: Type = sp_target(lambda: ModifyColorsCallback)
    max_batch_size: int
    """Maximum batch size to reach with cloning."""
    # cloning_mode: str = simple_parsing.choice("doubling", "linear", default="doubling")
    cloning_mode: str = "doubling"
    """Rate of which to clone the batch elements."""

@dataclass
class ModifyColorsCallback(Callback):
    config: ModifyColorsCallbackConfig

    def __call__(self):
        return partial(
            self.callback_func,
            max_batch_size=self.config.max_batch_size,
            cloning_mode=self.config.cloning_mode,
        )

    @staticmethod
    def callback_func(
        pipeline: DiffusionPipeline,
        i: int,
        t: int,
        callback_kwargs: dict,
        max_batch_size: int,
        cloning_mode: str,
    ) -> dict:

        # Shrinking towards the mean (will also remove outliers)
        def soft_clamp_tensor(input_tensor, threshold=3.5, boundary=4):
            if max(abs(input_tensor.max()), abs(input_tensor.min())) < 4:
                return input_tensor
            channel_dim = 1

            max_vals = input_tensor.max(channel_dim, keepdim=True)[0]
            max_replace = ((input_tensor - threshold) / (max_vals - threshold)) * (boundary - threshold) + threshold
            over_mask = (input_tensor > threshold)

            min_vals = input_tensor.min(channel_dim, keepdim=True)[0]
            min_replace = ((input_tensor + threshold) / (min_vals + threshold)) * (-boundary + threshold) - threshold
            under_mask = (input_tensor < -threshold)

            return torch.where(over_mask, max_replace, torch.where(under_mask, min_replace, input_tensor))

        # Center tensor (balance colors)
        def center_tensor(input_tensor, channel_shift=1, full_shift=1, channels=[0, 1, 2, 3]):
            for channel in channels:
                input_tensor[0, channel] -= input_tensor[0, channel].mean() * channel_shift
            return input_tensor - input_tensor.mean() * full_shift


        # Maximize/normalize tensor
        def maximize_tensor(input_tensor, boundary=4, channels=[0, 1, 2]):
            min_val = input_tensor.min()
            max_val = input_tensor.max()

            normalization_factor = boundary / max(abs(min_val), abs(max_val))
            input_tensor[0, channels] *= normalization_factor

            return input_tensor


        latents = callback_kwargs["latents"]
        batch_size = int(latents.shape[0])

        outputs = {}
        if batch_size < max_batch_size:
            if cloning_mode == "doubling":
                outputs, _ = _doubling_cloning_mode(callback_kwargs, batch_size, i, t, do_print=False, do_mask=False)

            # latents = outputs["latents"][mask]
            # latents = latents + torch.randn_like(latents) * 0.01

        # Not going to bother optimizing this lol. We will only test results for this one, not speed. (Original code is slow too)
        for batch in range(batch_size):
            org_latents = outputs["latents"] if "latents" in outputs else callback_kwargs["latents"]
            latents = org_latents[batch].unsqueeze(0)
            if t > 950:
                threshold = max(latents.max(), abs(latents.min())) * 0.998
                print(
                    f"End of iteration {i}, timestep {t}: Apply soft_clamp_tensor to latents with threshold {threshold}"
                )
                latents = soft_clamp_tensor(latents, threshold*0.998, threshold)
            if t > 700:
                print(
                    f"End of iteration {i}, timestep {t}: Apply center_tensor to latents with channel_shift 0.8 and full_shift 0.8"
                )
                latents = center_tensor(latents, 0.8, 0.8)
            if 1 < t < 100:  # Simplify chained comparison
                print(
                    f"End of iteration {i}, timestep {t}: Apply center_tensor to latents with channel_shift 0.6 and full_shift 1.0, and maximize_tensor"
                )
                latents = center_tensor(latents, 0.6, 1.0)
                latents = maximize_tensor(latents)

            org_latents[batch] = latents.squeeze(0)

        if "latents" in outputs:
            return outputs
        else:
            return {"latents": callback_kwargs["latents"]}


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
            "lcm-sd-controlnet": SD_1_5_LCM_ControlNetConfig,
            "lcm-dreamshaper": LCM_Dreamshaper_DiffusionModelConfig,
            "lcm-pixart-alpha": LCMPixArtAlphaConfig,
            # "sdxl-controlnet": SDXL_ControlNetConfig,
            # "sdxl": SDXL_Config,
            # "sd2": SD2_Config
        },
        default="lcm-sdxl",
    )
    """Model configuration."""
    callback: CallbackConfig = simple_parsing.subgroups(
        {
            "dummy_callback": DummyCallbackConfig,
            "identify_inputs": IdentifyInputsCallbackConfig,
            "cloning_only": CloningOnlyCallbackConfig,
            "random_noise": RandomNoiseCallbackConfig,
            "away_from_average": AwayFromAverageCallbackConfig,
            "modify_colors": ModifyColorsCallbackConfig,
        },
        default="dummy_callback",
    )
    """Callback configuration."""
    comment: str = ""
    """Additional comment to add to the experiment ID (used mainly to distinguish wandb runs)."""
    benchmark_mode: bool = False
    """Whether to run the experiment in benchmark mode or not. Will record timing statistics and other metrics, while disabling saving of images and other outputs."""
    benchmark_runs: int = 5
    """Number of runs to perform in benchmark mode before taking average. (There will always be one warmup run.)"""

    def __post_init__(self):
        # If callback is MoveAwayFromAverageCallback, set the max_steps to the number of inference steps
        if isinstance(self.callback, AwayFromAverageCallbackConfig):
            self.callback.max_steps = self.model.num_inference_steps
            print("Setting max_steps of callback to", self.model.num_inference_steps)


def parse_args():
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(ExperimentConfig, "exp_config")

    args = parser.parse_args()
    return args


def main():
    def _should_skip_saving(exp_config):
        if isinstance(exp_config.callback, IdentifyInputsCallbackConfig):
            return True
        elif exp_config.benchmark_mode:
            return True
        # Add more conditions later
        return False

    # TODO: WARNING, do not modify anything other the latents as the mask does not work currently 
    # with other tensors. Probably will not be fixed as there's no real point of modifying the other elements.

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

    if exp_config.use_wandb and not exp_config.benchmark_mode:
        # Start wandb for regular projects
        wandb.init(
            project="diffusion",
            entity="kwand_jalnyn_csc2231",
            name=exp_id,
            config=exp_config.to_dict(save_dc_types=True),
            mode="offline" if exp_config.wandb_offline else "online",
        )

    # Construct all the objects from the configs
    callback = exp_config.callback.setup()
    callback_func = callback()
    model = exp_config.model.setup()

    if exp_config.use_full_determinism:
        enable_full_determinism()
    else:
        torch.manual_seed(exp_config.seed)

    # This will also serve as the warm-up run before benchmarking. Otherwise, is the regular inference run.
    with torch.inference_mode():
        start = timeit.default_timer()
        images = model.run_inference(callback_func, exp_dir)
        end = timeit.default_timer()
        print(f"Total runtime: {end - start:.2f} s")

    if exp_config.benchmark_mode:
        benchmark_times = []
        for _ in range(exp_config.benchmark_runs - 1):
            start = timeit.default_timer()
            with torch.inference_mode():
                images = model.run_inference(callback_func, exp_dir)
            end = timeit.default_timer()
            benchmark_times.append(end - start)
        benchmark_times = np.array(benchmark_times)
        avg_runtime = np.mean(benchmark_times)
        print(f"Average runtime over {exp_config.benchmark_runs} runs: {avg_runtime:.2f} s")

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
    if exp_config.use_wandb and not skip_saving:
        columns = ["exp_id", "model_config", "callback_config", "stage"]
        for i in range(exp_config.callback.max_batch_size):
            columns.append(f"images_{i}")

        data = []
        subdirs = [d for d in Path(exp_dir).iterdir() if d.is_dir() and "step" in d.name]
        natsorted_subdirs = natsorted(subdirs, key=lambda x: x.name)
        for folder in natsorted_subdirs:
            _row = [exp_id, str(exp_config.model), str(exp_config.callback), folder.name] + [
                wandb.Image(str(img)) for img in folder.iterdir()
            ]
            data.append(_row)

        # Append the final images at the end to ensure it shows up as the last row
        _row = [exp_id, str(exp_config.model), str(exp_config.callback), "final"] + [
            wandb.Image(str(img)) for img in img_output_dir.iterdir()
        ]
        data.append(_row)

        # Needed to pad the rows to ensure all rows have the same length
        def _pad_row(row, max_length):
            return row + [None] * (max_length - len(row))

        # Do padding to ensure all rows have the same length
        data = [_pad_row(row, len(columns)) for row in data]

        results_table = wandb.Table(data=data, columns=columns)
        wandb.log({"results": results_table})

        wandb.finish()

    # Create a seperate wandb project just to record benchmark timings
    if exp_config.benchmark_mode:
        wandb.init(
            project="diffusion_benchmark2",
            entity="kwand_jalnyn_csc2231",
            name=exp_id,
            config=exp_config.to_dict(save_dc_types=True),
            mode="online",
        )
        columns = ["exp_id", "model_config", "callback_config", "device",
                   "precision", "num_images_per_prompt",
                   "callback_max_batch_size",
                   "num_inference_steps",
                   "benchmark_time", "all_runtimes"]
        data = [
            [
                exp_id, str(exp_config.model), str(exp_config.callback), exp_config.model.device,
                exp_config.model.precision, exp_config.model.num_images_per_prompt,
                getattr(exp_config.callback, "max_batch_size", -1),
                exp_config.model.num_inference_steps,
                avg_runtime, benchmark_times,
            ]
        ]
        results_table = wandb.Table(data=data, columns=columns)
        wandb.log({"benchmark_results": results_table})

        wandb.finish()


if __name__ == "__main__":
    main()
