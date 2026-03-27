from __future__ import annotations

import os
import sys
import warnings
import traceback

# Import from paths module for config access
try:
    from .paths import load_config
except ImportError:
    from paths import load_config

# Silence the noisy CUDA autocast warning on Mac
warnings.filterwarnings(
    "ignore",
    message="User provided device_type of 'cuda', but CUDA is not available",
    category=UserWarning,
)

import torch
import numpy as np
from PIL import Image
from diffusers import ZImagePipeline

try:
    from .sdnq_policy import apply_sdnq_compile_policy
except ImportError:
    from sdnq_policy import apply_sdnq_compile_policy

# Basic Triton detection is a fast precheck only; SDNQ runtime probe remains
# the final authority when Triton appears available.
apply_sdnq_compile_policy()

from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model
from safetensors.torch import load_file
from diffusers.loaders.peft import _SET_ADAPTER_SCALE_FN_MAPPING

# Import from hardware module
try:
    from .hardware import (
        PrecisionId,
        MODEL_ID_MAP,
        get_available_models,
        should_enable_attention_slicing,
        get_ram_gb,
        detect_device,
        is_xpu_available,
    )
    from .logger import get_logger
except ImportError:
    from hardware import (
        PrecisionId,
        MODEL_ID_MAP,
        get_available_models,
        should_enable_attention_slicing,
        get_ram_gb,
        detect_device,
        is_xpu_available,
    )
    from logger import get_logger

logger = get_logger("zimage.engine")

def log_info(message: str):
    logger.info(message)

def log_warn(message: str):
    logger.warning(message)


def _empty_xpu_cache() -> None:
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and hasattr(xpu, "empty_cache"):
        xpu.empty_cache()


def _is_xpu_bf16_supported() -> bool:
    xpu = getattr(torch, "xpu", None)
    if xpu is None or not hasattr(xpu, "is_bf16_supported"):
        return False
    try:
        return bool(xpu.is_bf16_supported())
    except Exception as e:
        log_warn(f"Failed to probe XPU bfloat16 support: {e}")
        return False

# Environment variable to force-enable torch.compile (use at your own risk)
_TORCH_COMPILE_ENV_VAR = "ZIMAGE_ENABLE_TORCH_COMPILE"

def is_torch_compile_safe() -> bool:
    """
    Determine if torch.compile is safe to use for the current environment.

    Returns True if torch.compile is known to be stable, False otherwise.
    This can be overridden by setting ZIMAGE_ENABLE_TORCH_COMPILE=1 via
    environment variable or config file (~/.z-image-studio/config.json).

    The safety check considers:
    - Python version (3.12+ has known torch.compile issues with Z-Image models)
    - ROCm/AMD GPUs (experimentally supported, disabled by default)
    - Future: PyTorch version (when 2.6+ potentially stabilizes 3.12 support)

    Returns:
        bool: True if torch.compile is considered safe, False otherwise
    """
    # User override - opt-in to experimental/unsafe behavior
    # Priority: environment variable > config file
    if os.getenv(_TORCH_COMPILE_ENV_VAR, "") == "1":
        log_warn(f"{_TORCH_COMPILE_ENV_VAR}=1: User has forced torch.compile enabled (via env)")
        return True

    # Check config file
    cfg = load_config()
    cfg_value = cfg.get(_TORCH_COMPILE_ENV_VAR) if isinstance(cfg, dict) else None
    if cfg_value:
        if str(cfg_value) == "1" or cfg_value is True:
            log_warn(f"{_TORCH_COMPILE_ENV_VAR}=1: User has forced torch.compile enabled (via config)")
            return True

    # Python 3.12+ has known compatibility issues with torch.compile on Z-Image models
    # See: https://github.com/anthropics/z-image-studio/issues/49
    if sys.version_info >= (3, 12):
        return False
        
    # ROCm: Disable torch.compile by default as it is experimental on AMD
    if detect_device() == "rocm":
        return False

    # TODO: Add PyTorch version check when 2.6+ is released
    # Future: if torch.__version__ >= (2, 6) and sys.version_info >= (3, 12):
    #             return True

    # Default: safe for Python < 3.12
    return True

warnings.filterwarnings(
    "ignore",
    message="`torch_dtype` is deprecated! Use `dtype` instead!",
    category=FutureWarning,
)

_cached_pipe = None
_cached_precision = None
_cached_original_transformer = None  # Store uncompiled transformer for fallback
_is_using_compiled_transformer = False  # Track if transformer is compiled

def load_pipeline(device: str = None, precision: PrecisionId = "q8") -> ZImagePipeline:
    global _cached_pipe, _cached_precision, _cached_original_transformer, _is_using_compiled_transformer
    
    # Cache key uses precision directly now
    cache_key = precision

    if _cached_pipe is not None and _cached_precision == cache_key:
        return _cached_pipe

    if device is None:
        device = detect_device()
    log_info(f"using device: {device}")
    
    # If we are switching models, unload the old one first to free memory
    if _cached_pipe is not None:
        log_info(f"Switching model. Unloading old model...")
        del _cached_pipe
        import gc
        gc.collect()
        if (device == "cuda" or device == "rocm") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if device == "mps" and torch.backends.mps.is_available():
             torch.mps.empty_cache()
        if device == "xpu" and is_xpu_available():
            _empty_xpu_cache()
        _cached_pipe = None

    # Directly use MODEL_ID_MAP
    model_id = MODEL_ID_MAP[precision] # This will raise KeyError if precision is not valid, let it
    log_info(f"using model: {model_id} (precision={precision})")

    # Select optimal dtype based on device capabilities
    if device == "cpu":
        torch_dtype = torch.float32
    elif device == "mps":
        torch_dtype = torch.bfloat16
    elif device == "cuda":
        if torch.cuda.is_bf16_supported():
            log_info("CUDA device supports bfloat16 -> using bfloat16")
            torch_dtype = torch.bfloat16
        else:
            log_warn("CUDA device does NOT support bfloat16 -> falling back to float16")
            torch_dtype = torch.float16
    elif device == "rocm":
        # ROCm often supports float16. bfloat16 depends on newer cards (MI200+).
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
             log_info("ROCm device supports bfloat16 -> using bfloat16")
             torch_dtype = torch.bfloat16
        else:
             log_info("ROCm device -> using float16")
             torch_dtype = torch.float16
    elif device == "xpu":
        if _is_xpu_bf16_supported():
            log_info("XPU device supports bfloat16 -> using bfloat16")
            torch_dtype = torch.bfloat16
        else:
            log_info("XPU device -> using float16")
            torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Replaced subprocess call with get_ram_gb() from hardware module
    total_ram_gb = get_ram_gb()
    if total_ram_gb is None:
        # Fallback logic if detection fails, assume high ram? Or low?
        # Original code would crash if sysctl failed on Mac, or maybe returned 0.
        # Let's assume 0 or handle None.
        total_ram_gb = 0

    if model_id == "Tongyi-MAI/Z-Image-Turbo" and total_ram_gb >= 32:
        low_cpu_mem_usage=False
    else:
        low_cpu_mem_usage=True

    log_info(f"try to load model with torch_dtype={torch_dtype} ...")

    pipe = ZImagePipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    
    # PyTorch ROCm builds use "cuda" as the device type
    torch_device = "cuda" if device == "rocm" else device
    pipe = pipe.to(torch_device)
    
    # Compatibility shim for SD3LoraLoaderMixin which expects text_encoder_2 and 3
    if not hasattr(pipe, "text_encoder_2"):
        pipe.text_encoder_2 = None
    if not hasattr(pipe, "text_encoder_3"):
        pipe.text_encoder_3 = None

    # Monkey-patch peft scale mapping for ZImageTransformer2DModel
    if "ZImageTransformer2DModel" not in _SET_ADAPTER_SCALE_FN_MAPPING:
        log_info("Monkey-patching PEFT mapping for ZImageTransformer2DModel")
        _SET_ADAPTER_SCALE_FN_MAPPING["ZImageTransformer2DModel"] = lambda model_cls, weights: weights

    # Enable INT8 MatMul for AMD, Intel ARC and Nvidia GPUs:
    # Note: torch.compile is only applied when deemed safe for the current environment
    _is_using_compiled_transformer = False
    # Explicitly exclude ROCm for now as it may be unstable with SDNQ/Triton kernels
    if triton_is_available and (device == "cuda" or device == "xpu"):
        pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
        pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)

        # Store original uncompiled transformer for potential fallback
        _cached_original_transformer = pipe.transformer

        # Apply torch.compile only if safe for the current environment
        if is_torch_compile_safe():
            try:
                pipe.transformer = torch.compile(pipe.transformer)
                _is_using_compiled_transformer = True
                log_info("torch.compile enabled for transformer")
            except Exception as e:
                log_warn(f"torch.compile failed during setup: {e}")
                _is_using_compiled_transformer = False
        else:
            log_info(
                f"torch.compile disabled for this environment. "
                f"Set {_TORCH_COMPILE_ENV_VAR}=1 to force enable (experimental)."
            ) 

    if device == "cuda":
        pipe.enable_model_cpu_offload()

    if should_enable_attention_slicing(device):
        pipe.enable_attention_slicing()
    else:
        pipe.disable_attention_slicing()

    if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
        log_info("disable safety_checker")
        pipe.safety_checker = None

    _cached_pipe = pipe
    _cached_precision = cache_key
    return pipe

def generate_image(
    prompt: str,
    steps: int,
    width: int,
    height: int,
    seed: int = None,
    precision: str = "q4",
    loras: list[tuple[str, float]] = None,
):
    global _is_using_compiled_transformer
    pipe = load_pipeline(precision=precision)
    
    log_info(f"generating image for prompt: {prompt!r}")
    
    if loras:
        log_info(f"using LoRAs: {loras}")

    # Removed: DEBUG print statement
    # print(
    #     f"DEBUG: steps={steps}, width={width}, "
    #     f"height={height}, guidance_scale=0.0, seed={seed}, precision={precision}, "
    #     f"loras={loras}"
    # )

    active_adapters = []
    adapter_weights = []
    # Store LoRA data for potential torch.compile fallback
    lora_data = []  # List of (adapter_name, remapped_state_dict, strength) tuples

    if loras:
        try:
            for i, (path, strength) in enumerate(loras):
                adapter_name = f"lora_{i}"

                # Load raw state dict
                state_dict = load_file(path)

                # Remap keys: diffusion_model -> transformer
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("diffusion_model."):
                        new_key = key.replace("diffusion_model.", "transformer.")
                    else:
                        new_key = key
                    new_state_dict[new_key] = value

                # Store for potential fallback
                lora_data.append((adapter_name, new_state_dict, strength))

                pipe.transformer.load_lora_adapter(
                    new_state_dict,
                    adapter_name=adapter_name,
                    prefix="transformer",
                )
                active_adapters.append(adapter_name)
                adapter_weights.append(strength)

            if active_adapters:
                pipe.transformer.set_adapters(active_adapters, weights=adapter_weights)

        except Exception as e:
            log_warn(f"Failed to load LoRA weights: {e}")
            traceback.print_exc()
            # Clean up any loaded adapters if possible, though finally block should handle it
            raise e
    log_info(
        f"DEBUG: steps={steps}, width={width}, "
        f"height={height}, guidance_scale=0.0, seed={seed}, precision={precision}"
    )

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    # Prepare kwargs for generation
    gen_kwargs = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "height": height,
        "width": width,
        "guidance_scale": 0.0,
        "generator": generator,
    }

    had_error = False
    try:
        with torch.inference_mode():
            image = pipe(**gen_kwargs).images[0]
    except RuntimeError as e:
        had_error = True
        # Check if this is a torch.compile-related error that we can recover from
        error_msg = str(e)
        is_compile_error = (
            "shape of the mask" in error_msg.lower() or
            "pow_by_natural" in error_msg.lower() or
            "sympy" in error_msg.lower() or
            any(x in error_msg for x in ["indexed tensor", "does not match"])
        )

        if is_compile_error and _is_using_compiled_transformer and _cached_original_transformer is not None:
            # Fall back to uncompiled transformer
            log_warn("torch.compile failed during inference, falling back to uncompiled transformer")
            log_warn(f"Error: {error_msg}")
            pipe.transformer = _cached_original_transformer
            _is_using_compiled_transformer = False

            # Reapply LoRAs to the fallback transformer if they were loaded
            if lora_data:
                log_warn("Reapplying LoRA adapters to fallback transformer")
                fallback_adapters = []
                fallback_weights = []
                for adapter_name, state_dict, strength in lora_data:
                    pipe.transformer.load_lora_adapter(
                        state_dict,
                        adapter_name=adapter_name,
                        prefix="transformer",
                    )
                    fallback_adapters.append(adapter_name)
                    fallback_weights.append(strength)

                if fallback_adapters:
                    pipe.transformer.set_adapters(fallback_adapters, weights=fallback_weights)

            # Retry generation with uncompiled model
            with torch.inference_mode():
                image = pipe(**gen_kwargs).images[0]
        else:
            # Re-raise if not a recoverable compile error
            raise
    finally:
        if loras:
            try:
                log_info("unloading LoRA weights")
                pipe.transformer.unload_lora()
            except Exception as e:
                log_warn(f"Failed to unload LoRA weights: {e}")

        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if is_xpu_available():
            _empty_xpu_cache()
        # Clear CUDA cache on error to free GPU memory for next request
        if had_error and torch.cuda.is_available():
            log_warn("Clearing CUDA cache after error")
            torch.cuda.empty_cache()

    return image

def cleanup_memory():
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if is_xpu_available():
        _empty_xpu_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def edit_image(
    prompt: str,
    init_image: Image.Image,
    mask_image: Image.Image | None,
    strength: float,
    steps: int,
    width: int,
    height: int,
    seed: int = None,
    precision: str = "q4",
    loras: list[tuple[str, float]] = None,
) -> Image.Image:
    """
    Generate an image using img2img or inpainting via the existing ZImagePipeline.

    Uses manual VAE encoding + noise injection to achieve img2img/inpainting
    without needing FluxImg2ImgPipeline (which has component compatibility issues
    with ZImage models).

    Args:
        prompt: Text prompt for generation.
        init_image: Source image (PIL Image, RGB).
        mask_image: Mask image for inpainting (PIL Image, L mode).
                    White = areas to regenerate, black = areas to keep.
                    None for img2img mode.
        strength: Denoising strength (0.0 = no change, 1.0 = fully regenerated).
        steps: Number of inference steps.
        width: Output width (multiple of 16).
        height: Output height (multiple of 16).
        seed: Random seed for reproducibility.
        precision: Model precision ('full', 'q8', 'q4').
        loras: List of (path, strength) tuples for LoRA adapters.

    Returns:
        PIL Image with the edited result.
    """
    global _is_using_compiled_transformer

    mode = "inpaint" if mask_image is not None else "img2img"
    pipe = load_pipeline(precision=precision)

    log_info(f"editing image ({mode}) for prompt: {prompt!r}, strength={strength}")

    # Resize init_image (and mask) to target dimensions
    init_image = init_image.convert("RGB").resize((width, height), Image.LANCZOS)
    if mask_image is not None:
        mask_image = mask_image.convert("L").resize((width, height), Image.LANCZOS)

    # Load LoRAs (same logic as generate_image)
    active_adapters = []
    adapter_weights = []
    lora_data = []
    if loras:
        log_info(f"using LoRAs: {loras}")
        try:
            for i, (path, lora_strength) in enumerate(loras):
                adapter_name = f"lora_{i}"
                state_dict = load_file(path)
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("diffusion_model."):
                        new_key = key.replace("diffusion_model.", "transformer.")
                    else:
                        new_key = key
                    new_state_dict[new_key] = value

                lora_data.append((adapter_name, new_state_dict, lora_strength))
                pipe.transformer.load_lora_adapter(
                    new_state_dict,
                    adapter_name=adapter_name,
                    prefix="transformer",
                )
                active_adapters.append(adapter_name)
                adapter_weights.append(lora_strength)

            if active_adapters:
                pipe.transformer.set_adapters(active_adapters, weights=adapter_weights)
        except Exception as e:
            log_warn(f"Failed to load LoRA weights: {e}")
            traceback.print_exc()
            raise

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    had_error = False
    try:
        with torch.inference_mode():
            # --- Encode init_image to latents via VAE ---
            img_tensor = torch.from_numpy(
                np.array(init_image).astype(np.float32) / 127.5 - 1.0
            )
            # [H, W, C] -> [1, C, H, W]
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(device=pipe.device, dtype=pipe.dtype)

            latent_dist = pipe.vae.encode(img_tensor).latent_dist
            init_latents = latent_dist.sample(generator=generator)

            # Apply VAE scaling
            vae_config = pipe.vae.config
            if hasattr(vae_config, "shift_factor") and vae_config.shift_factor is not None:
                init_latents = (init_latents - vae_config.shift_factor) * vae_config.scaling_factor
            else:
                init_latents = init_latents * vae_config.scaling_factor

            # --- Compute the number of denoising steps based on strength ---
            # strength=1.0 -> full denoising (like txt2img)
            # strength=0.0 -> no denoising (return original)
            actual_steps = max(1, int(steps * strength))
            log_info(f"img2img: {actual_steps}/{steps} denoising steps (strength={strength})")

            # --- Add noise to unpacked latents (pipeline packs them internally) ---
            batch_size, channels, lat_h, lat_w = init_latents.shape

            # randn_like with generator not supported on MPS; generate on CPU and move
            noise = torch.randn(
                init_latents.shape,
                generator=torch.Generator("cpu").manual_seed(seed) if seed is not None else None,
                dtype=init_latents.dtype,
                device="cpu",
            ).to(init_latents.device)

            # For flow matching (FLUX), interpolate between noise and latents.
            # FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=...) applies its
            # internal shift factor to ANY sigmas passed in. To avoid double-shifting,
            # we compute raw (unshifted) sigmas and let the scheduler shift them.
            raw_sigmas = np.linspace(1.0, 1.0 / steps, steps).tolist()

            # Take partial schedule based on strength (without terminal 0.0 —
            # the scheduler appends it automatically)
            start_idx = len(raw_sigmas) - actual_steps
            partial_raw_sigmas = raw_sigmas[start_idx:]

            # Let the scheduler compute shifted timesteps from raw sigmas
            pipe.scheduler.set_timesteps(sigmas=partial_raw_sigmas, device=pipe.device)
            # The scheduler now holds properly shifted sigmas (with terminal 0.0 appended)
            shifted_sigma_start = pipe.scheduler.sigmas[0].item()

            if actual_steps > 0:
                latents = (1.0 - shifted_sigma_start) * init_latents + shifted_sigma_start * noise
            else:
                latents = init_latents

            # --- Run the standard pipeline with pre-computed latents ---
            gen_kwargs = {
                "prompt": prompt,
                "num_inference_steps": steps,
                "height": height,
                "width": width,
                "guidance_scale": 0.0,
                "generator": generator,
                "latents": latents,
                # Pass raw sigmas — the pipeline/scheduler will shift them correctly
                "sigmas": partial_raw_sigmas,
            }

            result = pipe(**gen_kwargs).images[0]

            # For inpainting: composite original (unmasked) with generated (masked)
            if mask_image is not None:
                result_np = np.array(result).astype(np.float32)
                init_np = np.array(init_image.resize(result.size, Image.LANCZOS)).astype(np.float32)
                # Resize mask to output dimensions
                mask_resized = np.array(
                    mask_image.resize(result.size, Image.LANCZOS)
                ).astype(np.float32) / 255.0
                mask_3ch = np.stack([mask_resized] * 3, axis=-1)
                # Blend: mask=1 (white) -> use generated, mask=0 (black) -> use original
                composited = mask_3ch * result_np + (1.0 - mask_3ch) * init_np
                image = Image.fromarray(composited.clip(0, 255).astype(np.uint8))
            else:
                image = result

    except RuntimeError as e:
        had_error = True
        error_msg = str(e)
        is_compile_error = (
            "shape of the mask" in error_msg.lower() or
            "pow_by_natural" in error_msg.lower() or
            "sympy" in error_msg.lower() or
            any(x in error_msg for x in ["indexed tensor", "does not match"])
        )

        if is_compile_error and _is_using_compiled_transformer and _cached_original_transformer is not None:
            log_warn("torch.compile failed during edit, falling back to uncompiled transformer")
            pipe.transformer = _cached_original_transformer
            _is_using_compiled_transformer = False
            # Retry is complex for edit_image; re-raise for now
            raise
        else:
            raise
    finally:
        if loras:
            try:
                log_info("unloading LoRA weights")
                pipe.transformer.unload_lora()
            except Exception as e:
                log_warn(f"Failed to unload LoRA weights: {e}")

        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if is_xpu_available():
            _empty_xpu_cache()
        if had_error and torch.cuda.is_available():
            log_warn("Clearing CUDA cache after error")
            torch.cuda.empty_cache()

    return image


# ---------------------------------------------------------------------------
# Upscale stage configuration
# ---------------------------------------------------------------------------
_UPSCALE_STAGES = [
    # Stage 1: tiny generation — low shift for creativity
    {"shift": 0.6, "cfg": 2.0, "denoise": 1.0, "steps": 9},
    # Stage 2: 2x latent upscale — high shift for prompt adherence
    {"shift": 10.0, "cfg": 4.0, "denoise": 0.4, "steps": 9},
    # Stage 3: 2x latent upscale — low shift, add details
    {"shift": 1.0, "cfg": 4.0, "denoise": 0.5, "steps": 9},
    # Stage 4: 2x latent upscale — high shift for final composition
    {"shift": 7.0, "cfg": 1.0, "denoise": 0.5, "steps": 9},
]


def upscale_generate(
    prompt: str,
    steps: int,
    width: int,
    height: int,
    seed: int = None,
    precision: str = "q4",
    loras: list[tuple[str, float]] = None,
    init_image: Image.Image = None,
) -> Image.Image:
    """
    Progressive 4-stage upscale generation workflow.

    Generates at a tiny resolution and progressively upscales through
    latent-space interpolation with varying scheduler shift values.
    This produces significantly higher quality and more detailed images
    than single-pass generation at the target resolution.

    If init_image is provided, stage 1 is skipped and the existing image
    is used as starting point for the upscale stages (2-4).

    Based on the community workflow:
    Stage 1: Generate at ~1/8 target res (shift=0.6, creative outline)
    Stage 2: 2x latent upscale + denoise (shift=10, follow prompt)
    Stage 3: 2x latent upscale + denoise (shift=1, add details)
    Stage 4: 2x latent upscale + denoise (shift=7, final refinement)

    Args:
        prompt: Text prompt for generation.
        steps: Base number of inference steps (used per stage).
        width: Final output width (multiple of 16).
        height: Final output height (multiple of 16).
        seed: Random seed for reproducibility.
        precision: Model precision ('full', 'q8', 'q4').
        loras: List of (path, strength) tuples for LoRA adapters.
        init_image: Optional source image to upscale (PIL Image, RGB).

    Returns:
        PIL Image at the requested resolution.
    """
    global _is_using_compiled_transformer

    pipe = load_pipeline(precision=precision)
    original_shift = pipe.scheduler.shift

    has_init = init_image is not None
    log_info(f"upscale generate: prompt={prompt!r}, target={width}x{height}, "
             f"seed={seed}, precision={precision}, has_init_image={has_init}")

    # When an init_image is provided, skip stage 1 and derive base from the
    # source image size. Otherwise compute a tiny base from the target.
    if has_init:
        init_image = init_image.convert("RGB")
        src_w, src_h = init_image.size
        # Round source dimensions to multiples of 16
        base_w = max(16, (src_w // 16) * 16)
        base_h = max(16, (src_h // 16) * 16)
        # Target is always 2x source (ignore user-provided width/height)
        width = max(16, (base_w * 2) // 16 * 16)
        height = max(16, (base_h * 2) // 16 * 16)
        # Progressive upscale stages, capped at target resolution
        stage_resolutions = [
            (base_w, base_h),                                      # init image (not generated)
            (min(base_w * 2, width), min(base_h * 2, height)),     # Stage 2: 2x (capped)
            (min((base_w * 4) // 16 * 16, width), min((base_h * 4) // 16 * 16, height)),  # Stage 3: 4x (capped)
            (width, height),                                       # Stage 4: target
        ]
        start_stage = 1  # skip stage 1 (generation)
    else:
        base_w = max(16, (width // 8) // 16 * 16)
        base_h = max(16, (height // 8) // 16 * 16)
        stage_resolutions = [
            (base_w, base_h),           # Stage 1: tiny base
            (base_w * 2, base_h * 2),   # Stage 2: 2x
            (base_w * 4, base_h * 4),   # Stage 3: 4x
            (width, height),            # Stage 4: target resolution
        ]
        start_stage = 0

    for i, (w, h) in enumerate(stage_resolutions):
        skip = " (init image)" if has_init and i == 0 else ""
        log_info(f"  stage {i+1}: {w}x{h}{skip}")

    # Load LoRAs (same logic as generate_image)
    active_adapters = []
    adapter_weights = []
    lora_data = []
    if loras:
        log_info(f"using LoRAs: {loras}")
        try:
            for i, (path, lora_strength) in enumerate(loras):
                adapter_name = f"lora_{i}"
                state_dict = load_file(path)
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("diffusion_model."):
                        new_key = key.replace("diffusion_model.", "transformer.")
                    else:
                        new_key = key
                    new_state_dict[new_key] = value
                lora_data.append((adapter_name, new_state_dict, lora_strength))
                pipe.transformer.load_lora_adapter(
                    new_state_dict,
                    adapter_name=adapter_name,
                    prefix="transformer",
                )
                active_adapters.append(adapter_name)
                adapter_weights.append(lora_strength)
            if active_adapters:
                pipe.transformer.set_adapters(active_adapters, weights=adapter_weights)
        except Exception as e:
            log_warn(f"Failed to load LoRA weights: {e}")
            traceback.print_exc()
            raise

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    had_error = False
    image = None
    latents = None

    try:
        with torch.inference_mode():
            # If we have an init_image, encode it to latents before the loop
            if has_init:
                init_resized = init_image.resize(
                    (stage_resolutions[0][0], stage_resolutions[0][1]), Image.LANCZOS
                )
                img_tensor = torch.from_numpy(
                    np.array(init_resized).astype(np.float32) / 127.5 - 1.0
                ).permute(2, 0, 1).unsqueeze(0).to(
                    device=pipe.device, dtype=pipe.dtype
                )
                latents = pipe.vae.encode(img_tensor).latent_dist.sample(
                    generator=generator
                )
                vae_config = pipe.vae.config
                if hasattr(vae_config, "shift_factor") and vae_config.shift_factor is not None:
                    latents = (latents - vae_config.shift_factor) * vae_config.scaling_factor
                else:
                    latents = latents * vae_config.scaling_factor
                image = init_resized
                log_info(f"upscale: encoded init image {base_w}x{base_h} to latents")

            for stage_idx, stage_cfg in enumerate(_UPSCALE_STAGES):
                if stage_idx < start_stage:
                    continue

                stage_num = stage_idx + 1
                s_w, s_h = stage_resolutions[stage_idx]
                s_shift = stage_cfg["shift"]
                s_cfg = stage_cfg["cfg"]
                s_denoise = stage_cfg["denoise"]
                s_steps = stage_cfg["steps"]

                log_info(f"upscale stage {stage_num}/4: {s_w}x{s_h}, "
                         f"shift={s_shift}, cfg={s_cfg}, denoise={s_denoise}")

                # Temporarily set scheduler shift for this stage
                pipe.scheduler._shift = s_shift

                if stage_idx == 0:
                    # --- Stage 1: Full txt2img at tiny resolution ---
                    gen_kwargs = {
                        "prompt": prompt,
                        "num_inference_steps": s_steps,
                        "height": s_h,
                        "width": s_w,
                        "guidance_scale": s_cfg,
                        "generator": generator,
                    }
                    result = pipe(**gen_kwargs)
                    image = result.images[0]

                    # Encode to latent for next stage's upscale
                    img_tensor = torch.from_numpy(
                        np.array(image).astype(np.float32) / 127.5 - 1.0
                    ).permute(2, 0, 1).unsqueeze(0).to(
                        device=pipe.device, dtype=pipe.dtype
                    )
                    latents = pipe.vae.encode(img_tensor).latent_dist.sample(
                        generator=generator
                    )
                    vae_config = pipe.vae.config
                    if hasattr(vae_config, "shift_factor") and vae_config.shift_factor is not None:
                        latents = (latents - vae_config.shift_factor) * vae_config.scaling_factor
                    else:
                        latents = latents * vae_config.scaling_factor

                else:
                    # --- Stages 2-4: Latent upscale + denoise ---
                    # Upscale latents to match this stage's target resolution.
                    # VAE downscales by factor 8, so target latent size = pixel_size / 8.
                    target_lat_h = s_h // 8
                    target_lat_w = s_w // 8
                    latents = torch.nn.functional.interpolate(
                        latents, size=(target_lat_h, target_lat_w), mode="nearest"
                    )

                    actual_steps = max(1, int(s_steps * s_denoise))

                    # Generate noise on CPU for MPS compatibility
                    noise = torch.randn(
                        latents.shape,
                        generator=torch.Generator("cpu").manual_seed(
                            seed + stage_idx if seed is not None else torch.randint(0, 2**32, (1,)).item()
                        ),
                        dtype=latents.dtype,
                        device="cpu",
                    ).to(latents.device)

                    # Compute raw sigmas and let scheduler shift them
                    raw_sigmas = np.linspace(1.0, 1.0 / s_steps, s_steps).tolist()
                    start_idx = len(raw_sigmas) - actual_steps
                    partial_raw_sigmas = raw_sigmas[start_idx:]

                    pipe.scheduler.set_timesteps(
                        sigmas=partial_raw_sigmas, device=pipe.device
                    )
                    shifted_sigma_start = pipe.scheduler.sigmas[0].item()

                    # Mix latents with noise at the starting sigma
                    noisy_latents = (
                        (1.0 - shifted_sigma_start) * latents
                        + shifted_sigma_start * noise
                    )

                    gen_kwargs = {
                        "prompt": prompt,
                        "num_inference_steps": s_steps,
                        "height": s_h,
                        "width": s_w,
                        "guidance_scale": s_cfg,
                        "generator": generator,
                        "latents": noisy_latents,
                        "sigmas": partial_raw_sigmas,
                    }
                    result = pipe(**gen_kwargs)
                    image = result.images[0]

                    # Re-encode to latent for next stage (if not last)
                    if stage_idx < 3:
                        img_tensor = torch.from_numpy(
                            np.array(image).astype(np.float32) / 127.5 - 1.0
                        ).permute(2, 0, 1).unsqueeze(0).to(
                            device=pipe.device, dtype=pipe.dtype
                        )
                        latents = pipe.vae.encode(img_tensor).latent_dist.sample(
                            generator=generator
                        )
                        vae_config = pipe.vae.config
                        if hasattr(vae_config, "shift_factor") and vae_config.shift_factor is not None:
                            latents = (latents - vae_config.shift_factor) * vae_config.scaling_factor
                        else:
                            latents = latents * vae_config.scaling_factor

                log_info(f"upscale stage {stage_num}/4 complete")

    except RuntimeError as e:
        had_error = True
        raise
    finally:
        # Restore original scheduler shift
        pipe.scheduler._shift = original_shift

        if loras:
            try:
                log_info("unloading LoRA weights")
                pipe.transformer.unload_lora()
            except Exception as e:
                log_warn(f"Failed to unload LoRA weights: {e}")

        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if is_xpu_available():
            _empty_xpu_cache()
        if had_error and torch.cuda.is_available():
            log_warn("Clearing CUDA cache after error")
            torch.cuda.empty_cache()

    return image
