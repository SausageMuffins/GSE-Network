import os
# Set this before importing torch so that it takes effect immediately.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import time
import argparse
import torch
import torchaudio
from model import GSENet  # Ensure model.py is in the same folder or PYTHONPATH

def load_audio(filepath, device):
    """
    Loads an audio file and returns a [1, samples] tensor.
    """
    waveform, sample_rate = torchaudio.load(filepath)
    # If multichannel, take the first channel
    waveform = waveform[0].unsqueeze(0)
    return waveform.to(device), sample_rate

def main():
    # Set up command-line arguments.
    parser = argparse.ArgumentParser(
        description="Test GSENet with options for device selection (cpu, cuda, or mps)."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to use: 'cpu', 'cuda', or 'mps' (for Apple Silicon/Metal). "
             "If not specified, the script selects automatically."
    )
    args = parser.parse_args()

    # Determine the device to use.
    if args.device is not None:
        if args.device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        elif args.device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print("MPS requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"Using device: {device}")

    # Instantiate the model and move it to the selected device.
    model = GSENet().to(device)
    checkpoint_path = "checkpoints/gsenet_model.ckpt"

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print("Loaded checkpoint with keys:", list(checkpoint.keys()))

    # Extract the state_dict from the checkpoint if needed.
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # Remove the "model." prefix if present.
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint

    # Load the state dict into the model.
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print("Missing keys:", missing_keys)
    if unexpected_keys:
        print("Unexpected keys:", unexpected_keys)

    model.eval()  # Set the model to evaluation mode

    # Path to the validation data folder.
    val_dir = "data/val"
    subdirs = [
        os.path.join(val_dir, d)
        for d in os.listdir(val_dir)
        if os.path.isdir(os.path.join(val_dir, d))
    ]

    for subdir in subdirs:
        beam_path = os.path.join(subdir, "channel_1.wav")  # beamformed output
        ref_path = os.path.join(subdir, "channel_2.wav")   # reference mic

        # Check if both files exist.
        if not os.path.exists(beam_path) or not os.path.exists(ref_path):
            print(f"Skipping {subdir} (missing one of the required files)")
            continue

        # Load audio files.
        beam_waveform, sr_beam = load_audio(beam_path, device)
        ref_waveform, sr_ref = load_audio(ref_path, device)

        if sr_beam != sr_ref:
            print(f"Sample rate mismatch in {subdir} (beam: {sr_beam}, ref: {sr_ref}). Skipping.")
            continue

        # Ensure both waveforms have the same length.
        min_len = min(ref_waveform.shape[1], beam_waveform.shape[1])
        ref_waveform = ref_waveform[:, :min_len]
        beam_waveform = beam_waveform[:, :min_len]

        # Run inference and record the time.
        with torch.no_grad():
            if device.type == "cuda":
                # For CUDA, use CUDA events for accurate timing.
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                enhanced_waveform = model(ref_waveform, beam_waveform)
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event)  # in milliseconds
            else:
                # For CPU or MPS, use time.time(). For MPS, fallback operations will run on CPU.
                if device.type == "mps":
                    # Optional: synchronize MPS if needed.
                    torch.mps.synchronize()
                start_time = time.time()
                enhanced_waveform = model(ref_waveform, beam_waveform)
                if device.type == "mps":
                    torch.mps.synchronize()
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000  # convert seconds to milliseconds

            print(f"Inference time for {subdir}: {elapsed_time:.2f} ms")

        # Save the enhanced audio in the same subdirectory.
        output_path = os.path.join(subdir, "enhanced.wav")
        torchaudio.save(output_path, enhanced_waveform.cpu(), sample_rate=sr_ref)
        print(f"Enhanced audio saved to: {output_path}")

if __name__ == "__main__":
    main()
