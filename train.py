import json
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from model import ResNet18MusicGenre
from AudioDataset import AudioDataset
import torchaudio
import matplotlib.pyplot as plt
import os
from compute_mean_and_std import compute_mean_std
from torchvision import transforms
import torchaudio.transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau
from evaluate import evaluate_model

with open('config.json', 'r') as f:
    config = json.load(f)

torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
audio_dir = config["audio_dir"]
num_classes = config["num_classes"]
input_channels = config["input_channels"]
log_file = config["log_file"]
batch_size = config["batch_size"]
drop_out_rate = config["drop_out_rate"]
epochs = config["epochs"]
lr = config["learning_rate"]
validation_ratio = config["validation_ratio"]
test_ratio = config["test_ratio"]
sample_rate = config["sample_rate"]
seconds = config["seconds"]
num_samples = int(seconds * sample_rate)
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

total_size = len(ad)
validation_size = int(total_size * validation_ratio)
test_size = int(total_size * test_ratio)
training_size = total_size - validation_size - test_size
train_dataset, val_dataset, test_dataset = random_split(
    ad, [training_size, validation_size, test_size])

mean, std = compute_mean_std(train_dataset, batch_size= batch_size, shuffle= False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

audioClassifier = ResNet18MusicGenre(num_classes=num_classes, input_channels=input_channels, drop_out_rate= drop_out_rate).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(audioClassifier.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

def accuracy(outputs, labels):
    preds = (torch.sigmoid(outputs) > 0.5).float()
    correct = (preds == labels).float().mean()
    return correct.item()


for epoch in range(epochs):
    print(f'in epoch number {epoch + 1}:/n')
    audioClassifier.train()
    train_loss = 0
    train_acc = 0

    for signals, labels in train_loader:
        signals = signals.to(device)
        labels = labels.to(device).float()
        outputs = audioClassifier(signals)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += accuracy(outputs, labels)

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    audioClassifier.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for signals, labels in val_loader:
            signals = signals.to(device)
            labels = labels.to(device).float()
            outputs = audioClassifier(signals)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            val_acc += accuracy(outputs, labels)

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    scheduler.step(val_loss) 
    current_lr = optimizer.param_groups[0]['lr']
    log_message = (f"Epoch [{epoch+1}/{epochs}], "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                   f"LR: {current_lr:.6f}\n")
    print(log_message)
    with open(log_file, "a") as f:
        f.write(log_message)

    if (epoch + 1) % 10 == 0:
        model_path = f"model_epoch_{epoch+1}.pth"
        torch.save(audioClassifier.state_dict(), model_path)
        print(f"Saved model checkpoint: {model_path}")

audioClassifier.eval()
test_loss = 0
test_acc = 0
with torch.no_grad():
    for signals, labels in test_loader:
        signals, labels = signals.to(device), labels.to(device).float()
        outputs = audioClassifier(signals)
        loss = loss_fn(outputs, labels)
        test_loss += loss.item()
        test_acc += accuracy(outputs, labels)

test_loss /= len(test_loader)
test_acc /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
