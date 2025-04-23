import argparse
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import soundfile as sf
import torch

from dia.model import Dia


# --- Global Setup ---
parser = argparse.ArgumentParser(description="Gradio interface for Nari TTS")
parser.add_argument(
    "--device", type=str, default=None, help="Force device (e.g., 'cuda', 'mps', 'cpu')"
)
parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")

args = parser.parse_args()


# Determine device
if args.device:
    device = torch.device(args.device)
elif torch.cuda.is_available():
    device = torch.device("cuda")
# Simplified MPS check for broader compatibility
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    # Basic check is usually sufficient, detailed check can be problematic
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Load Nari model and config
print("Loading Nari model...")
try:
    # Use the function from inference.py
    model = Dia.from_pretrained("nari-labs/Dia-1.6B", device=device)
except Exception as e:
    print(f"Error loading Nari model: {e}")
    raise


def run_inference(
    text_input: str,
    audio_prompt_input: Optional[Tuple[int, np.ndarray]],
    max_new_tokens: int,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    cfg_filter_top_k: int,
    speed_factor: float,
):
    """
    Runs Nari inference using the globally loaded model and provided inputs.
    Uses temporary files for text and audio prompt compatibility with inference.generate.
    """
    global model, device  # Access global model, config, device

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
                        # Attempt conversion, might fail for complex types
                        try:
                            audio_data = audio_data.astype(np.float32)
                        except Exception as conv_e:
                            raise gr.Error(
                                f"Failed to convert audio prompt to float32: {conv_e}"
                            )

                    # Ensure mono (average channels if stereo)
                    if audio_data.ndim > 1:
                        if audio_data.shape[0] == 2:  # Assume (2, N)
                            audio_data = np.mean(audio_data, axis=0)
                        elif audio_data.shape[1] == 2:  # Assume (N, 2)
                            audio_data = np.mean(audio_data, axis=1)
                        else:
                            gr.Warning(
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

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

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

    return output_audio


