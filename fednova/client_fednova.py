import time
import platform
import psutil
import numpy as np
from typing import Callable, Dict, List, OrderedDict

import flwr as fl
import torch
from flwr.common import Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, ConcatDataset

from models import test, test_with_metrics, train_fednova


class FlowerClientFedNova(fl.client.NumPyClient):

    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
    ) -> None:
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.cid = None
        self.round = 0
        
        self.validation_enabled = len(valloader.dataset) > 0 if valloader is not None else False

    def get_parameters(self, config: Dict[str, Scalar]):
        parameters = [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        return parameters

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config: Dict[str, Scalar]):
        self.round += 1
        
        self.set_parameters(parameters)
        
        a_i, g_i, training_metrics = train_fednova(
            self.net,
            self.trainloader,
            self.valloader,
            self.device,
            self.num_epochs,
            self.learning_rate,
            self.momentum,
            self.weight_decay,
        )
        
        g_i_np = [param.cpu().numpy() for param in g_i]
        
        if self.validation_enabled:
            val_loss, val_acc = test(self.net, self.valloader, self.device)
            _, _, extended_metrics = test_with_metrics(self.net, self.valloader, self.device)
            
            print(f"Client {self.cid} Round {self.round} Validation Results:")
            print(f"  Validation Loss: {val_loss:.4f}")
            print(f"  Validation Accuracy: {val_acc:.4f}")
            if 'f1_score' in extended_metrics:
                print(f"  F1 Score: {extended_metrics['f1_score']:.4f}")
        else:
            val_loss, val_acc = 0.0, 0.0
            extended_metrics = {
                "accuracy": 0.0,
                "validation_enabled": False
            }
            print(f"Client {self.cid} Round {self.round}: Validation disabled")
        
        num_samples = len(self.trainloader.dataset)
        
        metrics = {
            "client_id": str(self.cid) if self.cid is not None else "unknown",
            "round": int(self.round),
            "a_i": float(a_i),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "validation_enabled": bool(self.validation_enabled),
            "num_samples": float(num_samples),
        }
        
        if "history" in training_metrics:
            history = training_metrics["history"]
            
            for key, values in history.items():
                if isinstance(values, list) and values:
                    metrics[f"final_{key}"] = float(values[-1])
                elif isinstance(values, (int, float, bool)):
                    metrics[f"final_{key}"] = values if isinstance(values, bool) else float(values)
                    
            if self.validation_enabled and "val_loss" in history and len(history["val_loss"]) > 1:
                val_losses = history["val_loss"]
                improvement_rate = (val_losses[0] - val_losses[-1]) / val_losses[0] if val_losses[0] != 0 else 0
                metrics["val_loss_improvement_rate"] = float(improvement_rate)
        
        for key, value in extended_metrics.items():
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
            elif isinstance(value, (str, bool)):
                metrics[key] = value
        
        metrics_clean = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, str, bytes, bool)) or (
                isinstance(value, list) and all(isinstance(item, (int, float, str, bytes, bool)) for item in value)
            ):
                metrics_clean[key] = value
        
        return g_i_np, len(self.trainloader.dataset), metrics_clean

    def evaluate(self, parameters, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        
        train_dataset = self.trainloader.dataset
        val_dataset = self.valloader.dataset
        
        if len(val_dataset) > 0:
            combined_dataset = ConcatDataset([train_dataset, val_dataset])
            eval_data_type = "combined_train_val"
            print(f"Client {self.cid} Server Evaluation: {len(train_dataset)} train + {len(val_dataset)} val = {len(combined_dataset)} total samples")
        else:
            combined_dataset = train_dataset
            eval_data_type = "training_only"
            print(f"Client {self.cid} Server Evaluation: {len(train_dataset)} training samples (no validation data)")
        
        evaluation_loader = DataLoader(
            combined_dataset, 
            batch_size=self.trainloader.batch_size,
            shuffle=False
        )
        
        try:
            loss, acc = test(self.net, evaluation_loader, self.device)
            _, _, extended_metrics = test_with_metrics(self.net, evaluation_loader, self.device)
            
            print(f"Client {self.cid} Server-side Evaluation:")
            print(f"  Data Type: {eval_data_type} ({len(combined_dataset)} samples)")
            print(f"  Loss: {loss:.4f}, Accuracy: {acc:.4f}")
            
        except Exception as e:
            print(f"Error during server-side evaluation for client {self.cid}: {e}")
            return 0.0, 1, {"accuracy": 0.0, "error": str(e)}
        
        metrics = {
            "accuracy": float(acc),
            "client_id": str(self.cid) if self.cid is not None else "unknown",
            "eval_data_type": eval_data_type,
            "dataset_size": float(len(combined_dataset)),
            "server_evaluation": True,
        }
        
        for key, value in extended_metrics.items():
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
        
        metrics_clean = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, str, bytes, bool)):
                metrics_clean[key] = value
        
        return float(loss), len(combined_dataset), metrics_clean


def gen_client_fn(
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    num_epochs: int,
    learning_rate: float,
    model: DictConfig,
    momentum: float = 0.9,
    weight_decay: float = 1e-5,
) -> Callable[[str], FlowerClientFedNova]:

    def client_fn(cid: str) -> FlowerClientFedNova:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = instantiate(model).to(device)

        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]
        
        validation_enabled = len(valloader.dataset) > 0 if valloader is not None else False
        
        if not validation_enabled:
            print(f"Client {cid}: Validation disabled (empty validation dataset)")
        else:
            print(f"Client {cid}: Validation enabled ({len(valloader.dataset)} validation samples)")

        client = FlowerClientFedNova(
            net,
            trainloader,
            valloader,
            device,
            num_epochs,
            learning_rate,
            momentum,
            weight_decay,
        )
        client.cid = cid
        
        return client

    return client_fn