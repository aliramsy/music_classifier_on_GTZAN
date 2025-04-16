import torch
import torchaudio
import os

output_dir = "../classifier/dataset/genres_original/noise"
num_samples = 30 * 22050 
num_files = 1000
sample_rate = 22050

os.makedirs(output_dir, exist_ok=True)

for i in range(num_files):
    noise = torch.rand(num_samples) * 2 - 1
    filename = os.path.join(output_dir, f"noise_{i:03d}.wav")
    torchaudio.save(filename, noise.unsqueeze(0), sample_rate)
    print(f"Saved: {filename}")

print("Noise dataset generation complete!")