# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import math
import os
import shutil
from typing import Callable, List, Optional, Union
import subprocess

import numpy as np
import torch
import torchvision
from torchvision import transforms

from packaging import version

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging

from einops import rearrange
import cv2

from ..models.unet import UNet3DConditionModel
from ..utils.util import read_video, read_audio, write_video, check_ffmpeg_installed, get_device, compute_short_time_energy_from_wav, energy_to_difficulty
from ..utils.image_processor import ImageProcessor, load_fixed_mask
from ..whisper.audio2feature import Audio2Feature
import tqdm
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor

# helping hand for optimizing convolutional algorithms on CUDA
torch.backends.cudnn.benchmark = True

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []
    #device_util = get_device()

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.set_progress_bar_config(desc="Steps")

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        import torch

        # Normalize latents
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")

        device_type = latents.device.type
        autocast_dtype = torch.float16 if device_type in ["cuda", "mps"] else torch.float32

        try:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                    decoded = self.vae.decode(latents).sample

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "MPS backend" in str(e):
                print("MPS OOM detected during VAE decode — falling back to CPU...")

                if torch.backends.mps.is_available():
                    import torch.mps
                    torch.mps.empty_cache()

                vae_device = next(self.vae.parameters()).device
                self.vae = self.vae.to("cpu")

                with torch.no_grad():
                    decoded = self.vae.decode(latents.to("cpu", dtype=torch.float32)).sample

                self.vae = self.vae.to(vae_device)
                decoded = decoded.to(vae_device)

            else:
                raise

        if latents.device.type == "mps":
            import torch.mps
            torch.mps.empty_cache()

        return decoded



    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            1,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )  # (b, c, f, h, w)
        rand_device = "cpu" if str(device) == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance):
        import torch

        # Resize mask to match VAE latent shape (downsample by VAE scale factor)
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)

        device_type = device.type
        autocast_dtype = torch.float16 if device_type in ["cuda", "mps"] else torch.float32

        # encode with FP16 and CPU fallback
        try:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                    masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)

        except RuntimeError as e:
            if "mps" in str(e).lower() and "out of memory" in str(e).lower():
                print("MPS OOM during VAE encode (mask). Falling back to CPU...")
                import torch.mps
                torch.mps.empty_cache()

                vae_device = next(self.vae.parameters()).device
                self.vae = self.vae.to("cpu")

                with torch.no_grad():
                    masked_image_cpu = masked_image.to("cpu", dtype=torch.float32)
                    masked_image_latents = self.vae.encode(masked_image_cpu).latent_dist.sample(generator=generator)

                self.vae = self.vae.to(vae_device)
                masked_image_latents = masked_image_latents.to(vae_device)

            elif "cuda" in str(e).lower() and "out of memory" in str(e).lower():
                print("CUDA OOM during VAE encode (mask). Retrying with CPU fallback...")
                torch.cuda.empty_cache()

                vae_device = next(self.vae.parameters()).device
                self.vae = self.vae.to("cpu")

                with torch.no_grad():
                    masked_image_cpu = masked_image.to("cpu", dtype=torch.float32)
                    masked_image_latents = self.vae.encode(masked_image_cpu).latent_dist.sample(generator=generator)

                self.vae = self.vae.to(vae_device)
                masked_image_latents = masked_image_latents.to(vae_device)

            else:
                raise

        # Normalize encoded latents
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # Align devices and types
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # Assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        # Duplicate for classifier-free guidance
        if do_classifier_free_guidance:
            mask = torch.cat([mask] * 2)
            masked_image_latents = torch.cat([masked_image_latents] * 2)

        return mask, masked_image_latents



    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        import torch

        images = images.to(device=device, dtype=dtype)
        device_type = device.type
        autocast_dtype = torch.float16 if device_type in ["cuda", "mps"] else torch.float32

        try:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                    image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        except RuntimeError as e:
            if "mps" in str(e).lower() and "out of memory" in str(e).lower():
                print("[WARN] MPS OOM during VAE encode (image). Falling back to CPU...")
                import torch.mps
                torch.mps.empty_cache()

                vae_device = next(self.vae.parameters()).device
                self.vae = self.vae.to("cpu")

                with torch.no_grad():
                    images_cpu = images.to("cpu", dtype=torch.float32)
                    image_latents = self.vae.encode(images_cpu).latent_dist.sample(generator=generator)

                self.vae = self.vae.to(vae_device)
                image_latents = image_latents.to(vae_device)
            else:
                raise

        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")

        if do_classifier_free_guidance:
            image_latents = torch.cat([image_latents] * 2)

        return image_latents


    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    def affine_transform_video(self, video_frames: np.ndarray):
        faces = []
        boxes = []
        affine_matrices = []
        print(f"Affine transforming {len(video_frames)} faces...")
        for frame in tqdm.tqdm(video_frames):
            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)

        faces = torch.stack(faces)
        return faces, boxes, affine_matrices

    def restore_video(self, faces: torch.Tensor, video_frames: np.ndarray, boxes: list, affine_matrices: list):
        video_frames = video_frames[: len(faces)]
        out_frames = []
        print(f"Restoring {len(faces)} faces...")
        for index, face in enumerate(tqdm.tqdm(faces)):
            x1, y1, x2, y2 = boxes[index]
            height = int(y2 - y1)
            width = int(x2 - x1)
            face = torchvision.transforms.functional.resize(
                face, size=(height, width), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
            )
            out_frame = self.image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
            out_frames.append(out_frame)
        return np.stack(out_frames, axis=0)

    def loop_video(self, whisper_chunks: list, video_frames: np.ndarray):
        # If the audio is longer than the video, we need to loop the video
        if len(whisper_chunks) > len(video_frames):
            faces, boxes, affine_matrices = self.affine_transform_video(video_frames)
            num_loops = math.ceil(len(whisper_chunks) / len(video_frames))
            loop_video_frames = []
            loop_faces = []
            loop_boxes = []
            loop_affine_matrices = []
            for i in range(num_loops):
                if i % 2 == 0:
                    loop_video_frames.append(video_frames)
                    loop_faces.append(faces)
                    loop_boxes += boxes
                    loop_affine_matrices += affine_matrices
                else:
                    loop_video_frames.append(video_frames[::-1])
                    loop_faces.append(faces.flip(0))
                    loop_boxes += boxes[::-1]
                    loop_affine_matrices += affine_matrices[::-1]

            video_frames = np.concatenate(loop_video_frames, axis=0)[: len(whisper_chunks)]
            faces = torch.cat(loop_faces, dim=0)[: len(whisper_chunks)]
            boxes = loop_boxes[: len(whisper_chunks)]
            affine_matrices = loop_affine_matrices[: len(whisper_chunks)]
        else:
            video_frames = video_frames[: len(whisper_chunks)]
            faces, boxes, affine_matrices = self.affine_transform_video(video_frames)

        return video_frames, faces, boxes, affine_matrices

    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask_image_path: str = "latentsync/utils/mask.png",
        temp_dir: str = "temp",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        is_train = self.unet.training
        self.unet.eval()

        check_ffmpeg_installed()

        # 0. Define call parameters
        device = get_device()
        print(f"LipsyncPipeline initialized on device: {self.device}")
        # device = self._execution_device
        mask_image = load_fixed_mask(height, mask_image_path)
        self.image_processor = ImageProcessor(height, device=device, mask_image=mask_image)
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        # 1. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Check inputs
        self.check_inputs(height, width, callback_steps)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        whisper_feature = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)

        audio_samples = read_audio(audio_path)
        video_frames = read_video(video_path, use_decord=False)

        # difficulty curve at 50Hz, then we will sample per video frame
        energy_curve = compute_short_time_energy_from_wav(audio_path, frame_hz=50)
        difficulty_curve = energy_to_difficulty(energy_curve)  # (T50,)

        video_frames, faces, boxes, affine_matrices = self.loop_video(whisper_chunks, video_frames)

        synced_video_frames = []

        num_channels_latents = self.vae.config.latent_channels

        # Prepare latent variables
        all_latents = self.prepare_latents(
            len(whisper_chunks),
            num_channels_latents,
            height,
            width,
            weight_dtype,
            device,
            generator,
        )

        num_inferences = math.ceil(len(whisper_chunks) / num_frames)

        # Threaded data preparation
        def prepare_next(i):
            # difficulty slice for these frames (aligned 1:1 with video fps)
            start_f = i * num_frames
            end_f = (i + 1) * num_frames
            # difficulty is at 50Hz, but video is at 25fps -> just sample every 2nd
            local_difficulty = None
            if difficulty_curve is not None and len(difficulty_curve) > 0:
                # take every 2nd energy frame to match 25fps
                local_slice = difficulty_curve[0 : len(difficulty_curve) : 2]  # 50Hz -> 25Hz
                local_difficulty = local_slice[start_f:end_f]  # (<= num_frames,)

            # Prepare all CPU-heavy inputs for inference i
            audio_embeds = None
            if self.unet.add_audio_layer:
                audio_embeds = torch.stack(whisper_chunks[i * num_frames : (i + 1) * num_frames])
                audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                if do_classifier_free_guidance:
                    null_audio_embeds = torch.zeros_like(audio_embeds)
                    audio_embeds = torch.cat([null_audio_embeds, audio_embeds])

            inference_faces = faces[i * num_frames : (i + 1) * num_frames]
            latents = all_latents[:, :, i * num_frames : (i + 1) * num_frames]

            ref_pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                inference_faces, affine_transform=False
            )

            mask_latents, masked_image_latents = self.prepare_mask_latents(
                masks,
                masked_pixel_values,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )

            ref_latents = self.prepare_image_latents(
                ref_pixel_values,
                device,
                weight_dtype,
                generator,
                do_classifier_free_guidance,
            )

            return (
                audio_embeds,
                latents,
                mask_latents,
                masked_image_latents,
                masks,
                ref_pixel_values,
                ref_latents,
                local_difficulty,
            )


        # Use 2 threads (main GPU + 1 CPU worker)
        executor = ThreadPoolExecutor(max_workers=2)

        # Prepare the first batch
        future = executor.submit(prepare_next, 0)

        for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):
            # Wait for current batch to be ready
            audio_embeds, latents, mask_latents, masked_image_latents, masks, ref_pixel_values, ref_latents, local_difficulty = future.result()
            # from latentsync.utils.util import compute_audio_difficulty
            # decide guidance_scale based on audio difficulty
            # per-chunk guidance
            chunk_guidance_scale = guidance_scale
            if local_difficulty is not None and len(local_difficulty) > 0:
                # e.g. average difficulty in this chunk
                diff_val = float(local_difficulty.mean().item())
                # 0.0 -> 1.0x, 1.0 -> 1.3x (tunable)
                chunk_guidance_scale = guidance_scale * (1.0 + 0.3 * diff_val)
            if i % 5 == 0:
                print(f"[Adaptive] Chunk {i}: difficulty={diff_val:.3f}, guidance_scale={chunk_guidance_scale:.2f}")


            # Prefetch next batch in background
            if i + 1 < num_inferences:
                future = executor.submit(prepare_next, i + 1)

            # ---- Denoising Loop (GPU work) ----
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for j, t in enumerate(timesteps):
                    unet_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    unet_input = self.scheduler.scale_model_input(unet_input, t)
                    unet_input = torch.cat([unet_input, mask_latents, masked_image_latents, ref_latents], dim=1)

                    device_type = device.type
                    autocast_dtype = torch.float16 if device_type in ["cuda", "mps"] else torch.float32
                    with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                        noise_pred = self.unet(unet_input, t, encoder_hidden_states=audio_embeds).sample

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + chunk_guidance_scale * (noise_pred_audio - noise_pred_uncond)

                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and j % callback_steps == 0:
                            callback(j, t, latents)


            # ---- Postprocess & decode ----
            decoded_latents = self.decode_latents(latents)
            decoded_latents = self.paste_surrounding_pixels_back(
                decoded_latents, ref_pixel_values, 1 - masks, device, weight_dtype
            )
            synced_video_frames.append(decoded_latents)

        executor.shutdown()

        synced_video_frames = self.restore_video(torch.cat(synced_video_frames), video_frames, boxes, affine_matrices)

        audio_samples_remain_length = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        if is_train:
            self.unet.train()

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        write_video(os.path.join(temp_dir, "video.mp4"), synced_video_frames, fps=video_fps)

        sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)

        command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v libx264 -crf 18 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        subprocess.run(command, shell=True)
