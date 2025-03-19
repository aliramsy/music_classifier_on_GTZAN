from torch.utils.data import DataLoader


def compute_mean_std(dataset, batch_size, shuffle):
    loader = DataLoader(dataset, batch_size, shuffle)
    mean = 0.0
    std = 0.0
    num_samples = 0

    for mel_spectrograms, _ in loader:
        batch_samples = mel_spectrograms.size(0)
        mean += mel_spectrograms.mean() * batch_samples
        std += mel_spectrograms.std() * batch_samples
        num_samples += batch_samples

    mean /= num_samples
    std /= num_samples
    return mean, std