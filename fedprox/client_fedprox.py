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

from models import test, test_with_metrics, train_fedprox


class FlowerClientFedProx(fl.client.NumPyClient):

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
        proximal_mu: float = 0.01,
        validation_enabled: bool = True,
    ) -> None:
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.proximal_mu = proximal_mu
        self.validation_enabled = validation_enabled
        self.cid = None
        self.round = 0

    def get_parameters(self, config: Dict[str, Scalar]):
        parameters = [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        return parameters

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict()
        
        for k, v in params_dict:
            if isinstance(v, np.ndarray):
                if v.dtype.kind in 'fc':
                    state_dict[k] = torch.Tensor(v)
                elif v.dtype.kind in 'iu':
                    state_dict[k] = torch.Tensor(v.astype(np.float32))
                else:
                    print(f"Warning: Skipping parameter {k} with non-numeric dtype {v.dtype}")
                    continue
            else:
                try:
                    state_dict[k] = torch.Tensor(v)
                except Exception as e:
                    print(f"Warning: Could not convert parameter {k}: {e}")
                    continue
        
        self.net.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config: Dict[str, Scalar]):
        self.round += 1
        
        self.set_parameters(parameters)
        
        proximal_mu = config.get("proximal_mu", self.proximal_mu)
        
        if self.validation_enabled and len(self.valloader.dataset) > 0:
            validation_loader = self.valloader
        else:
            validation_loader = self.valloader
        
        self.net, training_metrics = train_fedprox(
            self.net,
            self.trainloader,
            validation_loader,
            self.device,
            self.num_epochs,
            self.learning_rate,
            self.momentum,
            self.weight_decay,
            proximal_mu,
        )
        
        final_p_np = self.get_parameters({})
        
        if self.validation_enabled and len(self.valloader.dataset) > 0:
            val_loss, val_acc = test(self.net, self.valloader, self.device)
            
            _, _, extended_metrics = test_with_metrics(self.net, self.valloader, self.device)
            
            print(f"Client {self.cid} Round {self.round} Validation Results (FedProx μ={proximal_mu}):")
            print(f"  Validation Loss: {val_loss:.4f}")
            print(f"  Validation Accuracy: {val_acc:.4f}")
            if 'f1_score' in extended_metrics:
                print(f"  F1 Score: {extended_metrics['f1_score']:.4f}")
        else:
            val_loss, val_acc = 0.0, 0.0
            extended_metrics = {"accuracy": 0.0, "f1_score": 0.0, "precision": 0.0, "recall": 0.0}
            print(f"Client {self.cid} Round {self.round}: Validation disabled (FedProx μ={proximal_mu})")
        
        num_samples = len(self.trainloader.dataset)
        
        metrics = {
            "client_id": str(self.cid) if self.cid is not None else "unknown",
            "round": int(self.round),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "validation_enabled": self.validation_enabled,
            "proximal_mu": float(proximal_mu),
            "num_samples": float(num_samples),
        }
        
        if "history" in training_metrics:
            history = training_metrics["history"]
            
            for key, values in history.items():
                if values:
                    metrics[f"final_{key}"] = float(values[-1])
                    
                    if key == "train_loss" and len(values) > 1:
                        improvement_rate = (values[0] - values[-1]) / values[0] if values[0] != 0 else 0
                        metrics["train_loss_improvement_rate"] = float(improvement_rate)
                    
                    if key in ["val_loss", "val_acc"] and not self.validation_enabled:
                        metrics[f"final_{key}"] = 0.0
        
        if "proximal_loss" in training_metrics:
            metrics["proximal_loss"] = float(training_metrics["proximal_loss"])
        
        for key, value in extended_metrics.items():
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
            elif isinstance(value, str):
                metrics[key] = value
        
        metrics_clean = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, str, bytes, bool)) or (
                isinstance(value, list) and all(isinstance(item, (int, float, str, bytes, bool)) for item in value)
            ):
                metrics_clean[key] = value
        
        return final_p_np, len(self.trainloader.dataset), metrics_clean

    def evaluate(self, parameters, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        
        from torch.utils.data import ConcatDataset
        
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
    proximal_mu: float = 0.01,
) -> Callable[[str], FlowerClientFedProx]:

    def client_fn(cid: str) -> FlowerClientFedProx:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = instantiate(model).to(device)

        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]
        
        validation_enabled = len(valloader.dataset) > 0
        
        if not validation_enabled:
            print(f"Client {cid}: Validation disabled (empty validation dataset)")
        else:
            print(f"Client {cid}: Validation enabled ({len(valloader.dataset)} validation samples)")

        client = FlowerClientFedProx(
            net,
            trainloader,
            valloader,
            device,
            num_epochs,
            learning_rate,
            momentum,
            weight_decay,
            proximal_mu,
            validation_enabled=validation_enabled,
        )
        client.cid = cid
        
        return client

    return client_fn