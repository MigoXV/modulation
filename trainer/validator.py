import torch
import wandb

from sklearn.metrics import precision_score, recall_score, f1_score


class Validator:
    
    def __init__(self,val_dataloader, loss_fn, device):
        self.dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.device = device
        
    def callback(self, model, steps):
        metrics = validate(model, self.dataloader, self.loss_fn, self.device)
        wandb.log(metrics)
        
def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }