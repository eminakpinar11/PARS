from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
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


class ScaffoldOptimizer(SGD):

    def __init__(self, grads, step_size, momentum, weight_decay):
        super().__init__(
            grads, lr=step_size, momentum=momentum, weight_decay=weight_decay
        )

    def step_custom(self, server_cv, client_cv):
        self.step()
        for group in self.param_groups:
            for i, par in enumerate(group["params"]):
                device = par.device
                s_cv_device = server_cv[i].to(device)
                c_cv_device = client_cv[i].to(device)

                par.data.add_(s_cv_device - c_cv_device, alpha=-group["lr"])


def train_scaffold(
        net: nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        epochs: int,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
        server_cv: List[torch.Tensor],
        client_cv: List[torch.Tensor],
) -> Tuple[nn.Module, Dict[str, Any]]:
    import time
    import copy

    validation_enabled = len(valloader.dataset) > 0 if valloader else False

    training_start_time = time.time()

    criterion = nn.CrossEntropyLoss()
    optimizer = ScaffoldOptimizer(
        net.parameters(), learning_rate, momentum, weight_decay
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_model_state = copy.deepcopy(net.state_dict())
    best_val_loss = float('inf')

    net.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()

            optimizer.step_custom(server_cv, client_cv)

            epoch_loss += loss.item() * target.size(0)
            _, predicted = torch.max(output.data, 1)
            epoch_total += target.size(0)
            epoch_correct += (predicted == target).sum().item()

        avg_epoch_loss = epoch_loss / epoch_total if epoch_total > 0 else 0
        epoch_acc = epoch_correct / epoch_total if epoch_total > 0 else 0

        history["train_loss"].append(avg_epoch_loss)
        history["train_acc"].append(epoch_acc)

        if validation_enabled:
            net.eval()
            val_loss, val_acc = test(net, valloader, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(net.state_dict())

            net.train()
        else:
            history["val_loss"].append(0.0)
            history["val_acc"].append(0.0)

            best_model_state = copy.deepcopy(net.state_dict())
            best_val_loss = 0.0

    if validation_enabled and best_val_loss < float('inf'):
        net.load_state_dict(best_model_state)
    elif not validation_enabled:
        net.load_state_dict(best_model_state)

    training_time = time.time() - training_start_time

    training_metrics = {
        "training_time": training_time,
        "history": history,
        "best_val_loss": best_val_loss if validation_enabled and best_val_loss < float('inf') else None,
        "validation_enabled": validation_enabled,
    }

    return net, training_metrics


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
        print(
            "sklearn library not found! Install with 'pip install scikit-learn' to calculate F1, precision and recall metrics.")
        pass
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        pass

    return loss, accuracy, metrics