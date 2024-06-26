import torch
import wandb

from sklearn.metrics import precision_score, recall_score, f1_score

from textbrewer.distiller_utils import move_to_device

class Validator:
    
    def __init__(self, config, val_dataloader, loss_fn, device):
        self.config = config
        self.dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.device = device
        self.epoch = 0
        
    def callback(self, model, step):
        metrics = validate(model, self.dataloader, self.loss_fn, self.device)
        log_data = metrics.copy()
        self.epoch += 1
        log_data["step"] = step
        log_data["epoch"] = self.epoch
        print(f"Epoch: {self.epoch}, Step: {step}, Metrics: {metrics}")
        if self.config.report_to == "wandb":
            wandb.log(log_data)
        
def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_to_device(batch, device)
            inputs, labels = batch
            # inputs, labels = inputs.to(device), labels.to(device)
            outputs = validate_on_batch(device,model,batch, {})
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
    
def validate_on_batch(device,model,batch, args) -> torch.Tensor:
    batch = move_to_device(batch, device)
    if type(batch) is dict:
        results = model(**batch,**args)
    else:
        results = model(*batch, **args)

    return results