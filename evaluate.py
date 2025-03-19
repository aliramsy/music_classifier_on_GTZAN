import torch

def evaluate_model(model, data_loader, criterion, device):
    model.eval()  
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  
        for signals, labels in data_loader:
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total

    return avg_loss, accuracy
