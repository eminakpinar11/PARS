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


class CVConfidence:


    def __init__(self, alpha=0.5, save_dir="", client_id=0):

        self.alpha = alpha
        self.save_dir = save_dir
        self.client_id = client_id

        self.prev_cv_gradients = {}
        self.confidence_history = []

        self.state_file = os.path.join(save_dir, f"cv_confidence_{client_id}.pt")
        self.load_state()

        print(f"CVConfidence initialized: λ = 1 + {alpha:.3f} * cos_sim, "
              f"range=[{1 - alpha}, {1 + alpha}], client_id={client_id}")

    def load_state(self):
        try:
            if os.path.exists(self.state_file):
                state = torch.load(self.state_file)
                self.prev_cv_gradients = state.get('prev_gradients', {})
                self.confidence_history = state.get('confidence_history', [])
                print(f"[CV Confidence] Loaded state for client {self.client_id}: "
                      f"{len(self.prev_cv_gradients)} gradients")
            else:
                print(f"[CV Confidence] No previous state for client {self.client_id}")
        except Exception as e:
            print(f"[CV Confidence] Error loading state: {e}")
            self.prev_cv_gradients = {}
            self.confidence_history = []

    def save_state(self):
        try:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

            state = {
                'prev_gradients': {k: v.cpu() if v is not None else None
                                   for k, v in self.prev_cv_gradients.items()},
                'confidence_history': self.confidence_history[-50:]
            }

            torch.save(state, self.state_file)
            print(f"[CV Confidence] Saved state for client {self.client_id}")
        except Exception as e:
            print(f"[CV Confidence] Error saving state: {e}")

    def update_cv_with_confidence(self, c_i_old, c_server, param_diff,
                                  scale_factor, param_idx=0):

        try:
            cv_gradient = -c_server + scale_factor * param_diff

            prev_gradient = self.prev_cv_gradients.get(param_idx, None)

            if (prev_gradient is None or
                    prev_gradient.shape != cv_gradient.shape or
                    prev_gradient.dtype != cv_gradient.dtype):
                confidence_coeff = 1.0
                cos_sim_value = 0.0
            else:
                cv_flat = cv_gradient.flatten()
                prev_flat = prev_gradient.flatten()

                norm_curr = torch.norm(cv_flat)
                norm_prev = torch.norm(prev_flat)

                if norm_curr < 1e-8 or norm_prev < 1e-8:
                    confidence_coeff = 1.0
                    cos_sim_value = 0.0
                else:
                    cos_sim = torch.dot(cv_flat, prev_flat) / (norm_curr * norm_prev)
                    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
                    cos_sim_value = float(cos_sim)

                    confidence_coeff = 1.0 + self.alpha * cos_sim_value

            confidence_coeff = max(0.1, confidence_coeff)

            new_cv = c_i_old + confidence_coeff * cv_gradient

            self.prev_cv_gradients[param_idx] = cv_gradient.detach().clone()

            if param_idx < 3:
                print(f"[CV Confidence Client {self.client_id}] Param {param_idx}: "
                      f"cos_sim={cos_sim_value:.3f}, λ={confidence_coeff:.3f}")

            return new_cv, float(confidence_coeff)

        except Exception as e:
            print(f"[CV Confidence] Error in update for param {param_idx}: {e}")
            new_cv = c_i_old - c_server + scale_factor * param_diff
            return new_cv, 1.0

    def track_round_confidence(self, round_confidences):
        if round_confidences:
            round_stats = {
                'avg': float(np.mean(round_confidences)),
                'min': float(np.min(round_confidences)),
                'max': float(np.max(round_confidences)),
                'std': float(np.std(round_confidences))
            }
            self.confidence_history.append(round_stats)

    def get_confidence_stats(self):
        if not self.confidence_history:
            return {
                'avg_confidence': 1.0,
                'min_confidence': 1.0 - self.alpha,
                'max_confidence': 1.0 + self.alpha,
                'std_confidence': 0.0,
                'num_rounds': 0
            }

        latest_round = self.confidence_history[-1]
        return {
            'avg_confidence': latest_round['avg'],
            'min_confidence': latest_round['min'],
            'max_confidence': latest_round['max'],
            'std_confidence': latest_round['std'],
            'num_rounds': len(self.confidence_history)
        }

    def reset(self):
        """Reset all state"""
        self.prev_cv_gradients = {}
        self.confidence_history = []
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
        print(f"[CV Confidence] Reset state for client {self.client_id}")


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
            cv_confidence_enabled: bool = False,
            cv_alpha: float = 0.5,
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
        self.cv_confidence_enabled = cv_confidence_enabled
        self.cv_alpha = cv_alpha
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

        if self.cv_confidence_enabled:
            self.cv_confidence_tracker = CVConfidence(
                alpha=cv_alpha,
                save_dir=self.dir,
                client_id=self.cid
            )
            print(f"[Client {self.cid}] *** CV CONFIDENCE ENABLED *** "
                  f"λ = 1 + {cv_alpha:.3f} * cos_sim, "
                  f"range=[{1 - cv_alpha}, {1 + cv_alpha}]")
        else:
            self.cv_confidence_tracker = None
            print(f"[Client {self.cid}] *** CV CONFIDENCE DISABLED *** Using standard SCAFFOLD")

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
            confidence_factors_used = []

            if self.cv_confidence_enabled:
                print(f"[Client {self.cid}] *** CV CONFIDENCE UPDATE *** Round {self.round}, "
                      f"λ = 1 + {self.cv_alpha:.3f} * cos_sim, range=[{1 - self.cv_alpha}, {1 + self.cv_alpha}]")
            else:
                print(f"[Client {self.cid}] *** STANDARD CV UPDATE *** Round {self.round}")

            for idx, (c_i_j, c_j, x_j, y_i_j) in enumerate(zip(self.client_cv, server_cv, x, y_i)):
                c_i_j_cpu = c_i_j.cpu() if c_i_j.is_cuda else c_i_j
                c_j_cpu = c_j.cpu() if c_j.is_cuda else c_j

                param_diff = torch.tensor(x_j, dtype=torch.float32) - torch.tensor(y_i_j, dtype=torch.float32)
                scale_factor = 1.0 / (self.learning_rate * self.num_epochs * len(self.trainloader))

                if self.cv_confidence_enabled and self.cv_confidence_tracker is not None:
                    new_cv, confidence_used = self.cv_confidence_tracker.update_cv_with_confidence(
                        c_i_j_cpu, c_j_cpu, param_diff, scale_factor, param_idx=idx
                    )
                    confidence_factors_used.append(confidence_used)
                else:
                    new_cv = (
                            c_i_j_cpu
                            - c_j_cpu
                            + scale_factor * param_diff
                    )
                    confidence_factors_used.append(1.0)

                c_i_n.append(new_cv)
                server_update_x.append((y_i_j - x_j))
                server_update_c.append((new_cv - c_i_j_cpu).numpy())

            self.client_cv = [cv.to(self.device) for cv in c_i_n]

            if self.cv_confidence_enabled and confidence_factors_used:
                avg_conf = sum(confidence_factors_used) / len(confidence_factors_used)
                min_conf = min(confidence_factors_used)
                max_conf = max(confidence_factors_used)

                print(f"[Client {self.cid}] *** CV CONFIDENCE STATS *** "
                      f"Avg: {avg_conf:.3f}, Range: [{min_conf:.3f}, {max_conf:.3f}]")

                if self.cv_confidence_tracker is not None:
                    self.cv_confidence_tracker.track_round_confidence(confidence_factors_used)
                    self.cv_confidence_tracker.save_state()

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

            if self.cv_confidence_enabled and self.cv_confidence_tracker is not None:
                confidence_stats = self.cv_confidence_tracker.get_confidence_stats()
                metrics.update({
                    "cv_confidence_enabled": True,
                    "cv_alpha": float(self.cv_alpha),
                    "cv_avg_confidence": float(confidence_stats['avg_confidence']),
                    "cv_min_confidence": float(confidence_stats['min_confidence']),
                    "cv_max_confidence": float(confidence_stats['max_confidence']),
                    "cv_std_confidence": float(confidence_stats['std_confidence']),
                })
            else:
                metrics.update({
                    "cv_confidence_enabled": False,
                    "cv_avg_confidence": 1.0,
                    "cv_min_confidence": 1.0,
                    "cv_max_confidence": 1.0,
                })

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

            print(
                f"Client {self.cid} Evaluate: Expected {expected_model_params} model params, received {total_params} total params")

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
                print(
                    f"Client {self.cid} Server Evaluation: {len(train_dataset)} train + {len(val_dataset)} val = {len(combined_dataset)} total samples")
            else:
                combined_dataset = train_dataset
                eval_data_type = "training_only"
                print(
                    f"Client {self.cid} Server Evaluation: {len(train_dataset)} training samples (no validation data)")

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
        cv_confidence_enabled: bool = False,
        cv_alpha: float = 0.5,
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
            cv_confidence_enabled=cv_confidence_enabled,
            cv_alpha=cv_alpha,
        )

    return client_fn