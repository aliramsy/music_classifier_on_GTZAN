import json
from torch.utils.data import Dataset
import torchaudio
import os
import torch
import glob

class AudioDataset(Dataset):
    def __init__(self, audio_dir, transformation, sample_rate, num_samples, device, augment = None):
        self.audio_dir = audio_dir
        self.num_samples = num_samples
        self.transformation = transformation.to(device)
        self.sample_rate = sample_rate
        self.device = device
        self.audio_files, self.labels = self.load_files()
        self.augment = augment

    def load_files(self):
        """Loads all audio file paths and their corresponding labels."""
        audio_files = []
        labels = []
        genres = sorted(os.listdir(self.audio_dir)) 
        genres.remove('noise')
        #print(genres)

        genre_to_label = {genre: idx + 1 for idx, genre in enumerate(genres) if genre != "noise"}  
        genre_to_label["noise"] = 0  
        #print(genre_to_label)   
        genres.append('noise')
        #print(genres)

        for genre in genres:
            genre_path = os.path.join(self.audio_dir, genre)
            if os.path.isdir(genre_path):  
                files = glob.glob(os.path.join(genre_path, "*.wav")) 
                audio_files.extend(files)
                if genre == "noise":
                    labels.extend([0] * len(files))  
                else:
                    labels.extend([genre_to_label[genre]] * len(files)) 
        return audio_files, labels

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        audio_file = self.audio_files[index]
        label = self.labels[index]

        if os.path.getsize(audio_file) == 0:  
            print(f"Skipping empty file: {audio_file}")
            return self.__getitem__((index + 1) % len(self))  

        try:
            signal, sr = torchaudio.load(audio_file)
        except Exception as e:
            #print(f"Error loading {audio_file}: {e}")
            return self.__getitem__((index + 1) % len(self))
        
        signal = signal.to(self.device)

        signal = self.resample_signal(signal, sr, self.sample_rate)
        signal = self.down_sample_signal(signal)

        if signal.shape[1] > self.num_samples:
            signal = self.cut_samples(signal)
        elif signal.shape[1] < self.num_samples:
            signal = self.right_pad(signal)

        if self.augment:
            signal = self.augment(signal)

        signal = self.transformation(signal)

        label_tensor = torch.zeros(11, dtype=torch.float32)  
        label_tensor[label] = 1.0  

        return signal, label_tensor

    def resample_signal(self, signal, pre_sample_rate, sample_rate):
        if pre_sample_rate != sample_rate:
            resampler = torchaudio.transforms.Resample(pre_sample_rate, sample_rate).to(self.device)
            signal = resampler(signal)
        return signal

    def down_sample_signal(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def cut_samples(self, signal):
        return signal[:, :self.num_samples]

    def right_pad(self, signal):
        num_missing_samples = self.num_samples - signal.shape[1]
        last_dim_padding = (0, num_missing_samples)
        return torch.nn.functional.pad(signal, last_dim_padding)

if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)

    audio_dir = config["audio_dir"]
    sample_rate = config["sample_rate"]
    seconds = config["seconds"]
    num_samples = int(seconds * sample_rate)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    n_fft = config["n_fft"]
    hop_length = config["hop_length"]
    win_length = config["win_length"]
    n_mels = config["n_mels"]  

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        f_min=0,
        f_max=10000,
        norm=None,  
        mel_scale="htk"
    )

    ad = AudioDataset(audio_dir, mel_spectrogram, sample_rate, num_samples, device)
    print(f"Loaded {len(ad)} audio files")
    print(ad[0][0].shape)
    print(ad[0][1])
