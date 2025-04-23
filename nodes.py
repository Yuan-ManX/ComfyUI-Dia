import argparse
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import soundfile as sf
import torch

from .src.model import Dia


class LoadDiaModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "nari-labs/Dia-1.6B",
                }),
                "device": ("STRING", {
                    "default": "auto",  # Options: "auto", "cuda", "cpu", "mps"
                })
            }
        }

    RETURN_TYPES = ("NARIMODEL", "STRING",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "Dia-TTS"

    def load(self, model_path, device):
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        device_torch = torch.device(device)
        print(f"Loading model from {model_path} on {device_torch}...")
        model = Dia.from_pretrained(model_path, device=device_torch)
        return (model,)


class InputText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "Enter your speech text here...",
                    "multiline": True
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "return_text"
    CATEGORY = "Dia-TTS"

    def return_text(self, text):
        return (text,)


class DiaTTS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "text_input": ("STRING",),
                "audio_prompt": ("AUDIO",),
                "max_new_tokens": ("INT", {"default": 3072, "min": 860, "max": 3072, "step": 50}),
                "cfg_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 5.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 1.5, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.8, "max": 1.0, "step": 0.01}),
                "cfg_filter_top_k": ("INT", {"default": 30, "min": 15, "max": 50, "step": 1}),
                "speed_factor": ("FLOAT", {"default": 0.94, "min": 0.1, "max": 5.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = "Dia-TTS"

    def run(self, model, text_input, audio_prompt_input, max_new_tokens, cfg_scale,
            temperature, top_p, cfg_filter_top_k, speed_factor):
        """
        Runs Nari inference using the globally loaded model and provided inputs.
        Uses temporary files for text and audio prompt compatibility with inference.generate.
        """

        if not text_input or text_input.isspace():
            print("Text input cannot be empty.")

        temp_txt_file_path = None
        temp_audio_prompt_path = None
        output_audio = (44100, np.zeros(1, dtype=np.float32))

        try:
            prompt_path_for_generate = None
            if audio_prompt_input is not None:
                sr, audio_data = audio_prompt_input
                # Check if audio_data is valid
                if (
                    audio_data is None or audio_data.size == 0 or audio_data.max() == 0
                ):  # Check for silence/empty
                    print("Audio prompt seems empty or silent, ignoring prompt.")
                else:
                    # Save prompt audio to a temporary WAV file
                    with tempfile.NamedTemporaryFile(
                        mode="wb", suffix=".wav", delete=False
                    ) as f_audio:
                        temp_audio_prompt_path = f_audio.name  # Store path for cleanup

                        # Basic audio preprocessing for consistency
                        # Convert to float32 in [-1, 1] range if integer type
                        if np.issubdtype(audio_data.dtype, np.integer):
                            max_val = np.iinfo(audio_data.dtype).max
                            audio_data = audio_data.astype(np.float32) / max_val
                        elif not np.issubdtype(audio_data.dtype, np.floating):
                            print(
                                f"Unsupported audio prompt dtype {audio_data.dtype}, attempting conversion."
                            )
                            # Attempt conversion, might fail for complex types
                            try:
                                audio_data = audio_data.astype(np.float32)
                            except Exception as conv_e:
                                print(
                                    f"Failed to convert audio prompt to float32: {conv_e}"
                                )

                        # Ensure mono (average channels if stereo)
                        if audio_data.ndim > 1:
                            if audio_data.shape[0] == 2:  # Assume (2, N)
                                audio_data = np.mean(audio_data, axis=0)
                            elif audio_data.shape[1] == 2:  # Assume (N, 2)
                                audio_data = np.mean(audio_data, axis=1)
                            else:
                                print(
                                    f"Audio prompt has unexpected shape {audio_data.shape}, taking first channel/axis."
                                )
                                audio_data = (
                                    audio_data[0]
                                    if audio_data.shape[0] < audio_data.shape[1]
                                    else audio_data[:, 0]
                                )
                            audio_data = np.ascontiguousarray(
                                audio_data
                            )  # Ensure contiguous after slicing/mean

                        # Write using soundfile
                        try:
                            sf.write(
                                temp_audio_prompt_path, audio_data, sr, subtype="FLOAT"
                            )  # Explicitly use FLOAT subtype
                            prompt_path_for_generate = temp_audio_prompt_path
                            print(
                                f"Created temporary audio prompt file: {temp_audio_prompt_path} (orig sr: {sr})"
                            )
                        except Exception as write_e:
                            print(f"Error writing temporary audio file: {write_e}")
                            print(f"Failed to save audio prompt: {write_e}")

            # 3. Run Generation

            start_time = time.time()

            # Use torch.inference_mode() context manager for the generation call
            with torch.inference_mode():
                output_audio_np = model.generate(
                    text_input,
                    max_tokens=max_new_tokens,
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    top_p=top_p,
                    use_cfg_filter=True,
                    cfg_filter_top_k=cfg_filter_top_k,  # Pass the value here
                    use_torch_compile=False,  # Keep False for Gradio stability
                    audio_prompt_path=prompt_path_for_generate,
                )

            end_time = time.time()
            print(f"Generation finished in {end_time - start_time:.2f} seconds.")

            # 4. Convert Codes to Audio
            if output_audio_np is not None:
                # Get sample rate from the loaded DAC model
                output_sr = 44100

                # --- Slow down audio ---
                original_len = len(output_audio_np)
                # Ensure speed_factor is positive and not excessively small/large to avoid issues
                speed_factor = max(0.1, min(speed_factor, 5.0))
                target_len = int(
                    original_len / speed_factor
                )  # Target length based on speed_factor
                if (
                    target_len != original_len and target_len > 0
                ):  # Only interpolate if length changes and is valid
                    x_original = np.arange(original_len)
                    x_resampled = np.linspace(0, original_len - 1, target_len)
                    resampled_audio_np = np.interp(x_resampled, x_original, output_audio_np)
                    output_audio = (
                        output_sr,
                        resampled_audio_np.astype(np.float32),
                    )  # Use resampled audio
                    print(
                        f"Resampled audio from {original_len} to {target_len} samples for {speed_factor:.2f}x speed."
                    )
                else:
                    output_audio = (
                        output_sr,
                        output_audio_np,
                    )  # Keep original if calculation fails or no change
                    print(f"Skipping audio speed adjustment (factor: {speed_factor:.2f}).")
                # --- End slowdown ---

                print(
                    f"Audio conversion successful. Final shape: {output_audio[1].shape}, Sample Rate: {output_sr}"
                )

            else:
                print("\nGeneration finished, but no valid tokens were produced.")
                # Return default silence
                print("Generation produced no output.")

        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback

            traceback.print_exc()
            # Re-raise as Gradio error to display nicely in the UI
            print(f"Inference failed: {e}")

        finally:
            # 5. Cleanup Temporary Files defensively
            if temp_txt_file_path and Path(temp_txt_file_path).exists():
                try:
                    Path(temp_txt_file_path).unlink()
                    print(f"Deleted temporary text file: {temp_txt_file_path}")
                except OSError as e:
                    print(
                        f"Warning: Error deleting temporary text file {temp_txt_file_path}: {e}"
                    )
            if temp_audio_prompt_path and Path(temp_audio_prompt_path).exists():
                try:
                    Path(temp_audio_prompt_path).unlink()
                    print(f"Deleted temporary audio prompt file: {temp_audio_prompt_path}")
                except OSError as e:
                    print(
                        f"Warning: Error deleting temporary audio prompt file {temp_audio_prompt_path}: {e}"
                    )

        return (output_audio,)


