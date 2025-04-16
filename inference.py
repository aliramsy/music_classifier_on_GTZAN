import json
import torch
import torchaudio
from torch import nn
import torchaudio.transforms as T
from model import ResNet18MusicGenre  
from torch.utils.data import DataLoader, random_split
from compute_mean_and_std import compute_mean_std
from torchvision import transforms
from AudioDatasetN import AudioDataset

with open('config.json', 'r') as f:
    config = json.load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
audio_dir = config["inference_audio_dir"]
checkpoint_path = config["checkpoint_path"]
num_classes = config["num_classes"]
batch_size = config["batch_size"]
sample_rate = config["sample_rate"]
seconds = config["seconds"]
num_samples = int(seconds * sample_rate)
n_fft = config["n_fft"]
hop_length = config["hop_length"]
win_length = config["win_length"]
n_mels = config["n_mels"]  
f_min = config["f_min"]
f_max = config["f_max"]
test_size = config["test_size"]
input_channels = config["input_channels"]

#mel_spectrogram = torchaudio.transforms.MelSpectrogram(
#    sample_rate=sample_rate,
#    n_mels=n_mels,
#    n_fft=n_fft,
#    hop_length=hop_length,
#    win_length=win_length,
#    f_min=0,
#    f_max=10000,
#    norm=None,  
#    mel_scale="htk"
#)
#
#ad = AudioDataset(audio_dir, mel_spectrogram, sample_rate, num_samples, device)

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        f_min=f_min,
        f_max=f_max,
    )
amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power').to(device)
    
transformation = torch.nn.Sequential(
        mel_spectrogram,
        amplitude_to_db
    )

ad = AudioDataset(audio_dir, transformation, sample_rate, num_samples, device, normalize= True)

total_size = len(ad)
training_size = total_size - test_size
train_dataset, test_dataset = random_split(ad, [training_size, test_size])

#mean, std = compute_mean_std(train_dataset, batch_size=batch_size, shuffle=False)
#
#test_transforms = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[mean.item()], std=[std.item()])
#])

#test_dataset.dataset.transform = test_transforms
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

loss_fn = nn.BCEWithLogitsLoss()

def load_model():
    """Loads the trained model from checkpoint."""
    model = ResNet18MusicGenre(num_classes=num_classes, input_channels=input_channels).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def predict(test_loader, model):
    """Runs inference on the test dataset and evaluates accuracy."""
    incorrect_predictions = []
    test_loss = 0
    test_correct = 0
    test_incorrect = 0
    total = 0

    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.to(device), labels.to(device).float()

            outputs = model(signals)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
            #preds = (torch.sigmoid(outputs) > 0.5).float()
            preds = torch.zeros_like(outputs)
            preds[torch.arange(outputs.size(0)), torch.argmax(outputs, dim=1)] = 1

            for i in range(len(labels)):
                if not torch.equal(preds[i], labels[i]):
                    test_incorrect += 1
                    incorrect_predictions.append((labels[i].cpu().numpy(), preds[i].cpu().numpy()))

            total += len(labels)       
    test_acc = 1 -  test_incorrect/total
    print(test_incorrect)
    test_loss /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    print("\nIncorrect Predictions (Actual → Predicted):")
    for actual, predicted in incorrect_predictions[:5]: 
        print(f"{actual} → {predicted}")

if __name__ == "__main__":
    model = load_model()
    predict(test_loader, model)
