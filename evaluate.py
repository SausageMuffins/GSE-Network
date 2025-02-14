import os
import argparse
import csv
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np

# Import the GSENet model from your model.py file
from model import GSENet

# Optionally import STOI and PESQ measures
try:
    from pystoi import stoi
except ImportError:
    stoi = None

try:
    from pesq import pesq
except ImportError:
    pesq = None


def compute_snr(reference, estimate):
    """
    Compute the Signal-to-Noise Ratio (SNR) in dB.
    
    Args:
        reference (np.array): The ground truth waveform.
        estimate (np.array): The enhanced (or estimated) waveform.
        
    Returns:
        float: SNR value in dB.
    """
    # Align the lengths by taking the minimum
    min_len = min(reference.shape[0], estimate.shape[0])
    reference = reference[:min_len]
    estimate = estimate[:min_len]
    
    error = reference - estimate
    energy_ref = np.sum(reference ** 2)
    energy_err = np.sum(error ** 2)
    if energy_err == 0:
        return float("inf")
    return 10 * np.log10(energy_ref / energy_err)


def evaluate_recording(recording_path, model, device, sample_rate=16000):
    """
    Evaluate a single recording directory.
    
    Args:
        recording_path (str): Path to the directory containing:
                              ground_truth.wav, channel_1.wav, channel_2.wav.
        model (torch.nn.Module): Loaded GSENet model.
        device (torch.device): Device to run the model on.
        sample_rate (int): Expected sample rate for the recordings.
    
    Returns:
        dict: A dictionary containing SNR metrics (and optionally STOI and PESQ).
    """
    # Define file paths
    gt_path = os.path.join(recording_path, "ground_truth.wav")
    beam_path = os.path.join(recording_path, "channel_1.wav")  # beamformed output
    ref_path = os.path.join(recording_path, "channel_2.wav")   # reference mic
    
    # Load the audio files using torchaudio (expected shape: [channels, samples])
    gt_wave, sr_gt = torchaudio.load(gt_path)
    beam_wave, sr_beam = torchaudio.load(beam_path)
    ref_wave, sr_ref = torchaudio.load(ref_path)
    
    # Check sample rates
    if sr_gt != sample_rate or sr_beam != sample_rate or sr_ref != sample_rate:
        print(f"Warning: one or more files in {recording_path} do not match the expected sample rate {sample_rate}.")
    
    # If multi-channel, take the first channel; then squeeze to get [samples]
    gt_wave = gt_wave[0] if gt_wave.shape[0] > 1 else gt_wave.squeeze(0)
    beam_wave = beam_wave[0] if beam_wave.shape[0] > 1 else beam_wave.squeeze(0)
    ref_wave = ref_wave[0] if ref_wave.shape[0] > 1 else ref_wave.squeeze(0)
    
    # Ensure all signals have the same length by truncating to the shortest length
    min_length = min(gt_wave.shape[-1], beam_wave.shape[-1], ref_wave.shape[-1])
    gt_wave = gt_wave[:min_length]
    beam_wave = beam_wave[:min_length]
    ref_wave = ref_wave[:min_length]
    
    # Convert to [batch, samples] and send to device
    gt_tensor = gt_wave.unsqueeze(0).to(device)
    beam_tensor = beam_wave.unsqueeze(0).to(device)
    ref_tensor = ref_wave.unsqueeze(0).to(device)
    
    # Forward pass through the model.
    # NOTE: The model.forward(ref_mic, beamformed) expects the reference as first argument
    with torch.no_grad():
        enhanced_tensor = model(ref_tensor, beam_tensor)
    
    # Save the enhanced waveform to the same folder
    enhanced_save_path = os.path.join(recording_path, "enhanced.wav")
    # enhanced_tensor is [batch, samples]; ensure it's on CPU
    torchaudio.save(enhanced_save_path, enhanced_tensor.cpu(), sample_rate)
    
    # Convert results to numpy arrays for evaluation
    enhanced = enhanced_tensor.squeeze(0).cpu().numpy()
    gt = gt_tensor.squeeze(0).cpu().numpy()
    beam = beam_tensor.squeeze(0).cpu().numpy()
    ref = ref_tensor.squeeze(0).cpu().numpy()
    
    # Compute SNR metrics
    snr_enhanced = compute_snr(gt, enhanced)
    snr_beam = compute_snr(gt, beam)
    snr_ref = compute_snr(gt, ref)
    
    metrics = {
        "snr_enhanced": snr_enhanced,
        "snr_beam": snr_beam,
        "snr_ref": snr_ref,
        "snr_improvement": snr_enhanced - snr_beam
    }
    
    # Optionally compute STOI
    if stoi is not None:
        try:
            stoi_score = stoi(gt, enhanced, sample_rate, extended=False)
            metrics["stoi"] = stoi_score
        except Exception as e:
            print(f"Error computing STOI for {recording_path}: {e}")
    
    # Optionally compute PESQ (wideband mode)
    if pesq is not None:
        try:
            pesq_score = pesq(sample_rate, gt, enhanced, 'wb')
            metrics["pesq"] = pesq_score
        except Exception as e:
            print(f"Error computing PESQ for {recording_path}: {e}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate the GSENet speech enhancement model.")
    parser.add_argument("--test_folder", type=str, default="data/test",
                        help="Path to the folder containing test recordings directories.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/gsenet_model_v2.ckpt",
                        help="Path to the model checkpoint file.")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Sample rate of the audio recordings.")
    args = parser.parse_args()
    
    # Device selection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    
    # Load the GSENet model and checkpoint
    model = GSENet().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # If the checkpoint is a dict with key "state_dict", extract it
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Remove the "model." prefix from the keys, if it exists
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("model."):
            new_key = key[len("model."):]
        new_state_dict[new_key] = value

    # Load state dict into the model; strict=False will ignore any unexpected keys (e.g., "loss_fn.window")
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    # Get list of directories in the test folder
    recording_dirs = [d for d in os.listdir(args.test_folder)
                      if os.path.isdir(os.path.join(args.test_folder, d))]
    
    if not recording_dirs:
        print("No recordings found in", args.test_folder)
        return
    
    all_metrics = []
    for rec in recording_dirs:
        rec_path = os.path.join(args.test_folder, rec)
        print(f"Evaluating recording: {rec_path}")
        metrics = evaluate_recording(rec_path, model, device, sample_rate=args.sample_rate)
        # Add the recording name to the metrics dictionary
        metrics["recording_name"] = rec
        print("Metrics:", metrics)
        all_metrics.append(metrics)
    
    # Compute and print average metrics over the test set
    avg_metrics = {}
    # Using the keys from the first recording (assumes all recordings yield the same keys)
    for key in all_metrics[0].keys():
        # Only average numeric metrics; skip the recording_name
        if key != "recording_name":
            avg_metrics[key] = np.mean([m[key] for m in all_metrics if key in m])
    
    print("\n--- Average Metrics ---")
    for key, value in avg_metrics.items():
        print(f"{key}: {value:.2f}")
    
    # Save the metrics for each sample to a CSV file in the test folder
    csv_file_path = os.path.join(args.test_folder, "evaluation_results.csv")
    # Ensure consistent ordering of columns; recording_name is first.
    fieldnames = ["recording_name"] + [key for key in all_metrics[0].keys() if key != "recording_name"]
    with open(csv_file_path, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_metrics:
            writer.writerow(row)
    
    print(f"CSV results saved to: {csv_file_path}")


if __name__ == "__main__":
    main()
