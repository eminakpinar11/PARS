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
from torch.utils.data import DataLoader

from models import test, test_with_metrics, train_pars


class CVConfidenceAdam:


    def __init__(self, alpha=0.2, save_dir="", client_id=0):

        self.alpha = alpha
        self.save_dir = save_dir
        self.client_id = client_id

        self.prev_avg_gradients = {}
        self.confidence_history = []

        self.state_file = os.path.join(save_dir, f"cv_confidence_adam_{client_id}.pt")
        self.load_state()

        print(f"[CVConfidenceAdam] Client {client_id}: λ = 1 + {alpha:.3f} * cos_sim, "
              f"range=[{1 - alpha:.3f}, {1 + alpha:.3f}]")

    def load_state(self):
        try:
            if os.path.exists(self.state_file):
                state = torch.load(self.state_file, map_location='cpu')
                loaded_gradients = state.get('prev_gradients', {})
                
                self.prev_avg_gradients = {}
                for k, v in loaded_gradients.items():
                    if v is not None:
                        if torch.is_tensor(v) and v.is_cuda:
                            v = v.cpu()
                            print(f"[CVConfidenceAdam] Client {self.client_id}: Moved loaded gradient {k} from CUDA to CPU")
                        self.prev_avg_gradients[k] = v
                
                self.confidence_history = state.get('confidence_history', [])
                print(f"[CVConfidenceAdam] Client {self.client_id}: "
                      f"Loaded state with {len(self.prev_avg_gradients)} gradients")
            else:
                print(f"[CVConfidenceAdam] Client {self.client_id}: No previous state")
        except Exception as e:
            print(f"[CVConfidenceAdam] Client {self.client_id}: Error loading state: {e}")
            self.prev_avg_gradients = {}
            self.confidence_history = []

    def save_state(self):
        """Save current state with proper device management"""
        try:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

            state = {
                'prev_gradients': {k: v.cpu() if torch.is_tensor(v) and v.is_cuda else v
                                   for k, v in self.prev_avg_gradients.items()},
                'confidence_history': self.confidence_history[-50:]
            }

            torch.save(state, self.state_file)
            print(f"[CVConfidenceAdam] Client {self.client_id}: Saved state")
        except Exception as e:
            print(f"[CVConfidenceAdam] Client {self.client_id}: Error saving state: {e}")

    def update_cv_with_confidence(self, avg_gradient_current, param_idx=0):

        try:
            original_device = avg_gradient_current.device
            if avg_gradient_current.is_cuda:
                avg_gradient_current = avg_gradient_current.cpu()

            prev_gradient = self.prev_avg_gradients.get(param_idx, None)

            if (prev_gradient is None or
                    prev_gradient.shape != avg_gradient_current.shape or
                    prev_gradient.dtype != avg_gradient_current.dtype):
                new_cv = avg_gradient_current.clone()
                confidence_coeff = 1.0
                cos_sim_value = 0.0

                if param_idx < 3:
                    print(f"[CVConfidenceAdam] Client {self.client_id} Param {param_idx}: "
                          f"First round, λ={confidence_coeff:.3f}")
            else:
                if torch.is_tensor(prev_gradient) and prev_gradient.is_cuda:
                    prev_gradient = prev_gradient.cpu()
                    print(f"[CVConfidenceAdam] Client {self.client_id}: Moved prev_gradient {param_idx} from CUDA to CPU")

                curr_flat = avg_gradient_current.flatten()
                prev_flat = prev_gradient.flatten()

                norm_curr = torch.norm(curr_flat)
                norm_prev = torch.norm(prev_flat)

                if norm_curr < 1e-8 or norm_prev < 1e-8:
                    cos_sim_value = 0.0
                    if param_idx < 3:
                        print(f"[CVConfidenceAdam] Client {self.client_id} Param {param_idx}: "
                              f"Zero gradient detected, cos_sim=0.0")
                else:
                    cos_sim = torch.dot(curr_flat, prev_flat) / (norm_curr * norm_prev)
                    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
                    cos_sim_value = float(cos_sim)

                confidence_coeff = 1.0 + self.alpha * cos_sim_value
                confidence_coeff = max(0.1, confidence_coeff)

                gradient_change = avg_gradient_current - prev_gradient

                cv_update = confidence_coeff * gradient_change

                new_cv = prev_gradient + cv_update

                if param_idx < 3:
                    print(f"[CVConfidenceAdam] Client {self.client_id} Param {param_idx}: "
                          f"cos_sim={cos_sim_value:.3f}, λ={confidence_coeff:.3f}, "
                          f"expected_range=[{1-self.alpha:.3f}, {1+self.alpha:.3f}]")

            self.prev_avg_gradients[param_idx] = avg_gradient_current.detach().clone().cpu()

            if original_device.type == 'cuda':
                new_cv = new_cv.to(original_device)

            return new_cv, float(confidence_coeff)

        except Exception as e:
            print(f"[CVConfidenceAdam] Client {self.client_id}: Error in param {param_idx}: {e}")
            import traceback
            traceback.print_exc()
            
            if avg_gradient_current.is_cuda:
                avg_gradient_current = avg_gradient_current.cpu()
            self.prev_avg_gradients[param_idx] = avg_gradient_current.detach().clone().cpu()
            return avg_gradient_current.clone(), 1.0

    def track_round_confidence(self, round_confidences):
        if round_confidences:
            round_stats = {
                'avg': float(np.mean(round_confidences)),
                'min': float(np.min(round_confidences)),
                'max': float(np.max(round_confidences)),
                'std': float(np.std(round_confidences)),
                'in_range_count': sum(1 for c in round_confidences 
                                    if (1 - self.alpha) <= c <= (1 + self.alpha)),
                'total_params': len(round_confidences)
            }
            self.confidence_history.append(round_stats)
            
            in_range_pct = (round_stats['in_range_count'] / round_stats['total_params']) * 100
            print(f"[CVConfidenceAdam] Client {self.client_id}: "
                  f"{round_stats['in_range_count']}/{round_stats['total_params']} "
                  f"({in_range_pct:.1f}%) parameters in expected range "
                  f"[{1-self.alpha:.3f}, {1+self.alpha:.3f}]")

    def get_confidence_stats(self):
        """Get statistics about current confidence factors"""
        if not self.confidence_history:
            return {
                'avg_confidence': 1.0,
                'min_confidence': 1.0 - self.alpha,
                'max_confidence': 1.0 + self.alpha,
                'std_confidence': 0.0,
                'num_rounds': 0,
                'in_range_percentage': 0.0
            }

        latest_round = self.confidence_history[-1]
        return {
            'avg_confidence': latest_round['avg'],
            'min_confidence': latest_round['min'],
            'max_confidence': latest_round['max'],
            'std_confidence': latest_round['std'],
            'num_rounds': len(self.confidence_history),
            'in_range_percentage': (latest_round['in_range_count'] / latest_round['total_params']) * 100
        }

    def reset(self):
        """Reset all state"""
        self.prev_avg_gradients = {}
        self.confidence_history = []
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
        print(f"[CVConfidenceAdam] Client {self.client_id}: Reset state")


class FlowerClientPars(fl.client.NumPyClient):

    def __init__(
            self,
            cid: int,
            net: torch.nn.Module,
            trainloader: DataLoader,
            valloader: DataLoader,
            device: torch.device,
            num_epochs: int,
            learning_rate: float,
            betas: tuple = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0,
            save_dir: str = "",
            validation_enabled: bool = True,
            cv_confidence_enabled: bool = False,
            cv_alpha: float = 0.2,
    ) -> None:
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
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

        self.adam_state = None

        if save_dir == "":
            save_dir = "client_cvs"
        self.dir = save_dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        if self.cv_confidence_enabled:
            self.cv_confidence_tracker = CVConfidenceAdam(
                alpha=cv_alpha,
                save_dir=self.dir,
                client_id=self.cid
            )
            print(f"[Client {self.cid}] *** CV CONFIDENCE ADAM ENABLED *** "
                  f"λ = 1 + {cv_alpha:.3f} * cos_sim, "
                  f"range=[{1 - cv_alpha:.3f}, {1 + cv_alpha:.3f}]")
        else:
            self.cv_confidence_tracker = None
            print(f"[Client {self.cid}] *** CV CONFIDENCE DISABLED *** Using standard PARS")

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

            cv_path = os.path.join(self.dir, f"client_cv_{self.cid}.pt")
            if os.path.exists(cv_path):
                try:
                    loaded_cv = torch.load(cv_path, map_location='cpu')
                    self.client_cv = [cv.to(self.device) for cv in loaded_cv]
                    print(f"Client {self.cid}: Loaded CV from file")
                except Exception as e:
                    print(f"Client {self.cid} WARN loading CV: {e}")
            else:
                print(f"Client {self.cid}: Using parameter-cloned CV (first round - no asymmetry)")

            adam_state_path = f"{self.dir}/adam_state_{self.cid}.pt"
            if os.path.exists(adam_state_path):
                self.adam_state = torch.load(adam_state_path, map_location='cpu')
                print(f"Client {self.cid}: Loaded Adam state from previous round")

            server_cv = [torch.Tensor(cv).to(self.device) for cv in server_cv]

            if self.cv_confidence_enabled:
                print(f"Client {self.cid}: Training with PARS + CV Confidence, "
                      f"LR={self.learning_rate}, Betas={self.betas}, α={self.cv_alpha}")
            else:
                print(f"Client {self.cid}: Training with standard PARS, "
                      f"LR={self.learning_rate}, Betas={self.betas}")

            training_start_time = time.time()

            self.net, training_metrics, new_adam_state = train_pars(
                self.net,
                self.trainloader,
                self.valloader,
                self.device,
                self.num_epochs,
                self.learning_rate,
                self.betas,
                self.eps,
                self.weight_decay,
                server_cv,
                self.client_cv,
                self.adam_state,
            )

            self.adam_state = new_adam_state
            adam_state_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in self.adam_state.items()}
            torch.save(adam_state_cpu, adam_state_path)

            training_time = training_metrics["training_time"]
            avg_gradients = training_metrics["avg_gradients"]

            x = parameters
            y_i = self.get_parameters(config={})

            c_i_n = []
            server_update_x = []
            server_update_c = []
            confidence_factors_used = []

            if self.cv_confidence_enabled:
                print(f"[Client {self.cid}] *** CV CONFIDENCE ADAM UPDATE *** Round {self.round}")
            else:
                print(f"[Client {self.cid}] *** STANDARD CV ADAM UPDATE *** Round {self.round}")

            if avg_gradients and self.cv_confidence_enabled and self.cv_confidence_tracker is not None:
                for idx, avg_grad in enumerate(avg_gradients):
                    if avg_grad.is_cuda:
                        print(f"[Client {self.cid}] WARNING: avg_grad {idx} on CUDA, moving to CPU for CV confidence")
                        avg_grad = avg_grad.cpu()
                    
                    new_cv, confidence_used = self.cv_confidence_tracker.update_cv_with_confidence(
                        avg_grad, param_idx=idx
                    )
                    confidence_factors_used.append(confidence_used)

                    c_i_n.append(new_cv)

                    server_update_x.append((y_i[idx] - x[idx]))

                    c_cv_device = self.client_cv[idx]
                    if c_cv_device.is_cuda:
                        c_cv_device = c_cv_device.cpu()
                    
                    server_update_c.append((new_cv.cpu().numpy() - c_cv_device.numpy()))

                device = self.client_cv[0].device if self.client_cv else self.device
                self.client_cv = [cv.to(device) for cv in c_i_n]

            elif avg_gradients:
                for idx, avg_grad in enumerate(avg_gradients):
                    if avg_grad.is_cuda:
                        avg_grad = avg_grad.cpu()
                    
                    c_i_n.append(avg_grad)
                    confidence_factors_used.append(1.0)

                    server_update_x.append((y_i[idx] - x[idx]))

                    c_cv_device = self.client_cv[idx].cpu() if self.client_cv[idx].is_cuda else self.client_cv[idx]
                    server_update_c.append((avg_grad.cpu().numpy() - c_cv_device.numpy()))

                self.client_cv = [cv.to(self.device) for cv in c_i_n]

            else:
                print(f"Client {self.cid}: No average gradients available, using parameter-based CV update")
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
                    confidence_factors_used.append(1.0)
                    server_update_x.append((y_i_j - x[idx]))
                    server_update_c.append((new_cv - c_i_j_cpu).numpy())

                self.client_cv = [cv.to(self.device) for cv in c_i_n]

            if self.cv_confidence_enabled and confidence_factors_used:
                avg_conf = sum(confidence_factors_used) / len(confidence_factors_used)
                min_conf = min(confidence_factors_used)
                max_conf = max(confidence_factors_used)

                print(f"[Client {self.cid}] *** CV CONFIDENCE ADAM STATS *** "
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
                "optimizer": "Adam",
                "learning_rate": float(self.learning_rate),
                "beta1": float(self.betas[0]),
                "beta2": float(self.betas[1]),
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
                    "cv_in_range_percentage": float(confidence_stats['in_range_percentage']),
                })
            else:
                metrics.update({
                    "cv_confidence_enabled": False,
                    "cv_avg_confidence": 1.0,
                    "cv_min_confidence": 1.0,
                    "cv_max_confidence": 1.0,
                    "cv_in_range_percentage": 100.0,
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
            # Model'ın beklediği parametre sayısını al
            expected_model_params = len(list(self.net.parameters()))
            total_params = len(parameters)

            print(
                f"Client {self.cid} Evaluate: Expected {expected_model_params} model params, received {total_params} total params")

            if total_params == expected_model_params * 2:
                server_cv = parameters[expected_model_params:]
                parameters = parameters[:expected_model_params]
                print(f"Client {self.cid}: PARS format detected (model + CV)")
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
                return 0.0, 1, {"accuracy": 0.0, "error": "parameter_mismatch", "optimizer": "Adam",
                                "cv_confidence": self.cv_confidence_enabled}

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

            from torch.utils.data import DataLoader
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
                "optimizer": "Adam",
                "cv_confidence": self.cv_confidence_enabled,
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
            return 0.0, 1, {"accuracy": 0.0, "error": str(e), "optimizer": "Adam",
                            "cv_confidence": self.cv_confidence_enabled}


def gen_client_fn_adam(
        trainloaders: List[DataLoader],
        valloaders: List[DataLoader],
        client_cv_dir: str,
        num_epochs: int,
        learning_rate: float,
        model: DictConfig,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        validation_enabled: bool = True,
        cv_confidence_enabled: bool = False,
        cv_alpha: float = 0.2,
) -> Callable[[str], FlowerClientPars]:
    if isinstance(betas, list):
        betas = tuple(betas)

    def client_fn(cid: str) -> FlowerClientPars:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = instantiate(model).to(device)

        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)] if len(valloaders) > int(cid) else DataLoader([])

        client_validation_enabled = validation_enabled and len(valloader.dataset) > 0

        if not client_validation_enabled:
            print(f"Client {cid}: Validation disabled (empty validation dataset or disabled in config)")
        else:
            print(f"Client {cid}: Validation enabled ({len(valloader.dataset)} validation samples)")

        return FlowerClientPars(
            int(cid),
            net,
            trainloader,
            valloader,
            device,
            num_epochs,
            learning_rate,
            betas,
            eps,
            weight_decay,
            save_dir=client_cv_dir,
            validation_enabled=client_validation_enabled,
            cv_confidence_enabled=cv_confidence_enabled,
            cv_alpha=cv_alpha,
        )

    return client_fn