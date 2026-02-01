from typing import List, Tuple, Dict, Union
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def train_fedavg(
    net: nn.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
) -> Tuple[nn.Module, Dict[str, Union[float, List[float]]]]:
    import copy
    import time
    
    validation_enabled = len(valloader.dataset) > 0
    
    start_time = time.time()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
    )
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    
    best_model_state = copy.deepcopy(net.state_dict())
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        net.train()
        epoch_train_loss = 0.0
        train_correct, train_total = 0, 0
        
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            
            output = net(data)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * target.size(0)
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        avg_train_loss = epoch_train_loss / train_total if train_total > 0 else 0
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        
        if validation_enabled:
            net.eval()
            val_loss, val_acc = test(net, valloader, device)
            
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(net.state_dict())
        else:
            val_loss, val_acc = 0.0, 0.0
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            best_model_state = copy.deepcopy(net.state_dict())
            best_val_loss = 0.0
    
    if validation_enabled and best_val_loss < float('inf'):
        net.load_state_dict(best_model_state)
    elif not validation_enabled:
        net.load_state_dict(best_model_state)
    
    training_time = time.time() - start_time
    
    metrics = {
        "training_time": training_time,
        "history": history,
        "best_val_loss": best_val_loss if validation_enabled and best_val_loss < float('inf') else None,
        "validation_enabled": validation_enabled,
    }
    
    return net, metrics


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
        print("sklearn library not found! Install with 'pip install scikit-learn' to calculate F1, precision and recall metrics.")
        pass
    except Exception as e:
        print(f"Error occurred while calculating metrics: {e}")
        pass
    
    return loss, accuracy, metrics