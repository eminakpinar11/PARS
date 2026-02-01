from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
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


class ParsOptimizer(Adam):
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.accumulated_grads = []
        self.num_steps = 0
        
    def zero_grad_and_reset_accumulation(self):
        self.zero_grad()
        self.accumulated_grads = []
        self.num_steps = 0
        
    def step_pars(self, server_cv, client_cv):
        
        raw_grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    raw_grads.append(p.grad.clone())
        
        if not self.accumulated_grads:
            self.accumulated_grads = [g.clone() for g in raw_grads]
        else:
            for i, g in enumerate(raw_grads):
                self.accumulated_grads[i].add_(g)
        self.num_steps += 1
        
        param_idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    device = p.device
                    s_cv_device = server_cv[param_idx].to(device)
                    c_cv_device = client_cv[param_idx].to(device)
                    
                    p.grad.add_(s_cv_device - c_cv_device)
                    param_idx += 1
        
        super().step()
        
    def get_average_gradients(self):
        """Get average of accumulated raw gradients for CV update."""
        if self.num_steps == 0:
            return []
        return [g / self.num_steps for g in self.accumulated_grads]


def train_pars(
        net: nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        epochs: int,
        learning_rate: float,
        betas: Tuple[float, float],
        eps: float,
        weight_decay: float,
        server_cv: List[torch.Tensor],
        client_cv: List[torch.Tensor],
        adam_state: Dict[str, Any] = None,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:

    import time
    import copy

    validation_enabled = len(valloader.dataset) > 0 if valloader else False

    training_start_time = time.time()

    criterion = nn.CrossEntropyLoss()
    optimizer = ParsOptimizer(
        net.parameters(), 
        lr=learning_rate, 
        betas=betas,
        eps=eps,
        weight_decay=weight_decay
    )
    
    if adam_state is not None:
        optimizer.load_state_dict(adam_state)
    
    optimizer.zero_grad_and_reset_accumulation()

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

            optimizer.step_pars(server_cv, client_cv)

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
    
    avg_gradients = optimizer.get_average_gradients()

    training_metrics = {
        "training_time": training_time,
        "history": history,
        "best_val_loss": best_val_loss if validation_enabled and best_val_loss < float('inf') else None,
        "validation_enabled": validation_enabled,
        "avg_gradients": avg_gradients,
    }
    
    new_adam_state = optimizer.state_dict()
    for state in new_adam_state['state'].values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cpu()

    return net, training_metrics, new_adam_state


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