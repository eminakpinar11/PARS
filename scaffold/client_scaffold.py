import os
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

from models import test, test_with_metrics, train_scaffold


class FlowerClientScaffold(fl.client.NumPyClient):

    def __init__(
            self,
            cid: int,
            net: torch.nn.Module,
            trainloader: DataLoader,
            valloader: DataLoader,
            device: torch.device,
            num_epochs: int,
            learning_rate: float,
            momentum: float,
            weight_decay: float,
            save_dir: str = "",
            validation_enabled: bool = True,
    ) -> None:
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.round = 0

        self.validation_enabled = validation_enabled and len(valloader.dataset) > 0

        if not self.validation_enabled:
            print(f"Client {self.cid}: Validation disabled (empty validation dataset or disabled in config)")
        else:
            print(f"Client {self.cid}: Validation enabled ({len(valloader.dataset)} validation samples)")

        self.client_cv = []
        for param in self.net.parameters():
            self.client_cv.append(torch.zeros(param.shape))

        if save_dir == "":
            save_dir = "client_cvs"
        self.dir = save_dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def get_parameters(self, config: Dict[str, Scalar]):
        parameters = [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        return parameters

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config: Dict[str, Scalar]):
        try:
            self.round += 1

            server_cv = parameters[len(parameters) // 2:]
            parameters = parameters[: len(parameters) // 2]

            self.set_parameters(parameters)

            self.client_cv = []
            for param in self.net.parameters():
                self.client_cv.append(param.clone().detach().to(self.device))
    
            if os.path.exists(f"{self.dir}/client_cv_{self.cid}.pt"):
                loaded_cv = torch.load(f"{self.dir}/client_cv_{self.cid}.pt")
                self.client_cv = [cv.to(self.device) for cv in loaded_cv]
                print(f"Client {self.cid}: Loaded CV from file")
            else:
                print(f"Client {self.cid}: Using parameter-cloned CV (first round - no asymmetry)")
    
            server_cv = [torch.Tensor(cv).to(self.device) for cv in server_cv]

            print(f"Client {self.cid}: Training with SCAFFOLD-SGD, LR={self.learning_rate}")

            training_start_time = time.time()

            self.net, training_metrics = train_scaffold(
                self.net,
                self.trainloader,
                self.valloader,
                self.device,
                self.num_epochs,
                self.learning_rate,
                self.momentum,
                self.weight_decay,
                server_cv,
                self.client_cv,
            )

            training_time = training_metrics["training_time"]

            x = parameters
            y_i = self.get_parameters(config={})
            c_i_n = []
            server_update_x = []
            server_update_c = []

            print(f"[Client {self.cid}] CV UPDATE - Round {self.round}")

            for idx, (c_i_j, c_j, x_j, y_i_j) in enumerate(zip(self.client_cv, server_cv, x, y_i)):
                c_i_j_cpu = c_i_j.cpu() if c_i_j.is_cuda else c_i_j
                c_j_cpu = c_j.cpu() if c_j.is_cuda else c_j

                param_diff = torch.tensor(x_j, dtype=torch.float32) - torch.tensor(y_i_j, dtype=torch.float32)
                scale_factor = 1.0 / (self.learning_rate * self.num_epochs * len(self.trainloader))

                new_cv = (
                        c_i_j_cpu
                        - c_j_cpu
                        + scale_factor * param_diff
                )

                c_i_n.append(new_cv)

                server_update_x.append((y_i_j - x_j))
                server_update_c.append((new_cv - c_i_j_cpu).numpy())

            self.client_cv = [cv.to(self.device) for cv in c_i_n]

            try:
                cv_cpu = [cv.cpu() for cv in self.client_cv]
                torch.save(cv_cpu, f"{self.dir}/client_cv_{self.cid}.pt")
            except Exception as e:
                print(f"Client {self.cid} ERROR saving CV: {str(e)}")

            combined_updates = server_update_x + server_update_c

            if self.validation_enabled and len(self.valloader.dataset) > 0:
                val_loss, val_acc = test(self.net, self.valloader, self.device)

                _, _, extended_metrics = test_with_metrics(self.net, self.valloader, self.device)

                print(f"Client {self.cid} Round {self.round} Validation Results:")
                print(f"  Validation Loss: {val_loss:.4f}")
                print(f"  Validation Accuracy: {val_acc:.4f}")
                if 'f1_score' in extended_metrics:
                    print(f"  F1 Score: {extended_metrics['f1_score']:.4f}")
            else:
                val_loss, val_acc = 0.0, 0.0
                extended_metrics = {"accuracy": 0.0, "f1_score": 0.0, "precision": 0.0, "recall": 0.0}
                print(f"Client {self.cid} Round {self.round}: Validation disabled")

            num_samples = len(self.trainloader.dataset)

            metrics = {
                "client_id": str(self.cid),
                "round": self.round,
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc),
                "training_time": float(training_time),
                "validation_enabled": self.validation_enabled,
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

            for key, value in extended_metrics.items():
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
                elif isinstance(value, str):
                    metrics[key] = value

            metrics_clean = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float, str, bytes, bool)) or (
                        isinstance(value, list) and all(
                    isinstance(item, (int, float, str, bytes, bool)) for item in value)
                ):
                    metrics_clean[key] = value

            return (
                combined_updates,
                len(self.trainloader.dataset),
                metrics_clean,
            )
        except Exception as e:
            print(f"Client {self.cid} error in fit: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def evaluate(self, parameters, config: Dict[str, Scalar]):
        try:
            expected_model_params = len(list(self.net.parameters()))
            total_params = len(parameters)
            
            print(f"Client {self.cid} Evaluate: Expected {expected_model_params} model params, received {total_params} total params")
            
            if total_params == expected_model_params * 2:
                server_cv = parameters[expected_model_params:]
                parameters = parameters[:expected_model_params]
                print(f"Client {self.cid}: SCAFFOLD format detected (model + CV)")
            elif total_params == expected_model_params:
                server_cv = []
                print(f"Client {self.cid}: Model-only format detected")
            else:
                print(f"Warning: Client {self.cid} - Unexpected parameter count: {total_params}")
                server_cv = []
                parameters = parameters[:expected_model_params]
            
            try:
                self.set_parameters(parameters)
            except Exception as e:
                print(f"Client {self.cid} Parameter loading error: {e}")
                print(f"Expected model params: {len(list(self.net.parameters()))}")
                print(f"Received params: {len(parameters)}")
                return 0.0, 1, {"accuracy": 0.0, "error": "parameter_mismatch"}
            
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
            
            loss, acc = test(self.net, evaluation_loader, self.device)
            _, _, extended_metrics = test_with_metrics(self.net, evaluation_loader, self.device)
            
            print(f"Client {self.cid} Server-side Evaluation:")
            print(f"  Data Type: {eval_data_type} ({len(combined_dataset)} samples)")
            print(f"  Loss: {loss:.4f}, Accuracy: {acc:.4f}")
            
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
            
        except Exception as e:
            print(f"Client {self.cid} Server evaluation error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, 1, {"accuracy": 0.0, "error": str(e)}

def gen_client_fn(
        trainloaders: List[DataLoader],
        valloaders: List[DataLoader],
        client_cv_dir: str,
        num_epochs: int,
        learning_rate: float,
        model: DictConfig,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        validation_enabled: bool = True,
) -> Callable[[str], FlowerClientScaffold]:
    def client_fn(cid: str) -> FlowerClientScaffold:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = instantiate(model).to(device)

        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)] if len(valloaders) > int(cid) else DataLoader([])

        client_validation_enabled = validation_enabled and len(valloader.dataset) > 0

        if not client_validation_enabled:
            print(f"Client {cid}: Validation disabled (empty validation dataset or disabled in config)")
        else:
            print(f"Client {cid}: Validation enabled ({len(valloader.dataset)} validation samples)")

        return FlowerClientScaffold(
            int(cid),
            net,
            trainloader,
            valloader,
            device,
            num_epochs,
            learning_rate,
            momentum,
            weight_decay,
            save_dir=client_cv_dir,
            validation_enabled=client_validation_enabled,
        )

    return client_fn