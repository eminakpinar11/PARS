from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader


class CNN(nn.Module):

    def __init__(self, input_dim, hidden_dims, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNMnist(nn.Module):

    def __init__(self, input_dim, hidden_dims, num_classes) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_fednova(
    net: nn.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
) -> Tuple[float, List[torch.Tensor], Dict[str, Any]]:

    import copy
    import time
    
    validation_enabled = len(valloader.dataset) > 0 if valloader is not None else False
    
    training_start_time = time.time()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
    )
    
    prev_net = [param.detach().clone() for param in net.parameters()]
    
    best_model_state = copy.deepcopy(net.state_dict())
    best_val_loss = float('inf')
    
    history = {
        "train_loss": [],
        "validation_enabled": validation_enabled
    }
    
    if validation_enabled:
        history["val_loss"] = []
        history["val_accuracy"] = []
    
    net.train()
    local_steps = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            local_steps += 1
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        history["train_loss"].append(avg_epoch_loss)
        
        if validation_enabled:
            net.eval()
            val_loss, val_acc = test(net, valloader, device)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(net.state_dict())
            
            net.train()
    
    if validation_enabled and best_val_loss < float('inf'):
        net.load_state_dict(best_model_state)
    
    a_i = (
        local_steps - (momentum * (1 - momentum**local_steps) / (1 - momentum))
    ) / (1 - momentum)
    
    g_i = [
        torch.div(prev_param - param.detach(), a_i)
        for prev_param, param in zip(prev_net, net.parameters())
    ]
    
    training_time = time.time() - training_start_time
    
    training_metrics = {
        "training_time": training_time,
        "history": history,
        "best_val_loss": best_val_loss if validation_enabled and best_val_loss < float('inf') else None,
        "validation_enabled": validation_enabled,
    }

    return a_i, g_i, training_metrics


def test(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> Tuple[float, float]:

    criterion = nn.CrossEntropyLoss(reduction="sum")
    net.eval()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    loss = loss / total
    acc = correct / total
    return loss, acc

def test_with_metrics(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> Tuple[float, float, Dict[str, float]]:

    criterion = nn.CrossEntropyLoss(reduction="sum")
    net.eval()
    correct, total, loss = 0, 0, 0.0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    loss = loss / total
    accuracy = correct / total
    
    metrics = {"accuracy": float(accuracy)}
    
    try:
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        f1 = f1_score(all_targets, all_preds, average='weighted')
        precision = precision_score(all_targets, all_preds, average='weighted')
        recall = recall_score(all_targets, all_preds, average='weighted')
        
        metrics.update({
            "f1_score": float(f1),
            "precision": float(precision),
            "recall": float(recall)
        })
    except ImportError:
        print("sklearn library not found! To calculate F1, precision and recall metrics, install it with 'pip install scikit-learn'.")
        pass
    except Exception as e:
        print(f"Error occurred while calculating metrics: {e}")
        pass
    
    return loss, accuracy, metrics