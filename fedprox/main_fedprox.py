import os
import pickle
import time
import copy
import platform
import psutil
from typing import Dict, Tuple, List, Optional, Any, Union

import flwr as fl
import hydra
import numpy as np
import torch
import pandas as pd
from flwr.common import Scalar, Parameters, NDArrays, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_manager import SimpleClientManager
from flwr.server.server import Server
from flwr.server.history import History
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset import load_datasets
from models import test_with_metrics


def load_initial_weights(cfg: DictConfig, device: torch.device) -> Optional[torch.nn.Module]:
    if not cfg.get("initial_weights", {}).get("enabled", False):
        print("Initial weights disabled in config.")
        return None
    
    weights_dir = cfg.initial_weights.get("weights_dir", "saved_parameters")
    dataset_name = cfg.dataset_name
    
    weight_file_mapping = {
        "mnist": "mnist_weights.pkl",
        "fmnist": "fmnist_weights.pkl", 
        "cifar10": "cifar10_weights.pkl"
    }
    
    if dataset_name not in weight_file_mapping:
        print(f"Warning: No initial weights available for dataset '{dataset_name}'. Starting with random weights.")
        return None
    
    weight_file = weight_file_mapping[dataset_name]
    weight_path = os.path.join(weights_dir, weight_file)
    
    if not os.path.exists(weight_path):
        print(f"Warning: Initial weight file not found at '{weight_path}'. Starting with random weights.")
        return None
    
    try:
        with open(weight_path, "rb") as f:
            saved_parameters = pickle.load(f)
        
        print(f"Loading initial weights from: {weight_path}")
        
        net = instantiate(cfg.model).to(device)
        
        if isinstance(saved_parameters, list):
            params_dict = zip(net.state_dict().keys(), saved_parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            net.load_state_dict(state_dict, strict=True)
            print(f"✓ Successfully loaded initial weights from {weight_file}")
            
        elif isinstance(saved_parameters, dict):
            if "parameters" in saved_parameters:
                actual_parameters = saved_parameters["parameters"]
                print(f"Found wrapped parameters with metadata: {list(saved_parameters.keys())}")
                
                if isinstance(actual_parameters, list):
                    params_dict = zip(net.state_dict().keys(), actual_parameters)
                    state_dict = {k: torch.tensor(v) for k, v in params_dict}
                    net.load_state_dict(state_dict, strict=True)
                    print(f"✓ Successfully loaded wrapped initial weights from {weight_file}")
                elif isinstance(actual_parameters, dict):
                    state_dict = {}
                    for k, v in actual_parameters.items():
                        if isinstance(v, np.ndarray):
                            state_dict[k] = torch.tensor(v)
                        else:
                            state_dict[k] = v
                    net.load_state_dict(state_dict, strict=True)
                    print(f"✓ Successfully loaded wrapped initial weights from {weight_file}")
                else:
                    print(f"Warning: Unsupported wrapped parameter format in {weight_file}. Starting with random weights.")
                    return None
            else:
                state_dict = {}
                for k, v in saved_parameters.items():
                    if isinstance(v, np.ndarray):
                        state_dict[k] = torch.tensor(v)
                    else:
                        state_dict[k] = v
                net.load_state_dict(state_dict, strict=True)
                print(f"✓ Successfully loaded initial weights from {weight_file}")
            
        else:
            print(f"Warning: Unsupported saved parameter format in {weight_file}. Starting with random weights.")
            return None
            
        return net
        
    except Exception as e:
        print(f"Error loading initial weights from {weight_path}: {e}")
        print("Starting with random weights instead.")
        return None


class BestModelTracker:

    def __init__(
        self,
        model_config: DictConfig,
        testloader: DataLoader,
        device: torch.device,
        output_dir: str,
        convergence_patience: int = 5,
        convergence_threshold: float = 0.001
    ):
        self.model_config = model_config
        self.testloader = testloader
        self.device = device
        self.output_dir = output_dir
        self.convergence_patience = convergence_patience
        self.convergence_threshold = convergence_threshold
        
        self.best_accuracy = 0.0
        self.best_model_state = None
        self.best_round = 0
        self.metrics_history = []
        
        self.convergence_round = None
        self.no_improvement_count = 0
        self.last_significant_improvement = 0
        self.accuracy_history = []
        self.federated_evaluation_history = []
        self.centralized_evaluation_history = []
    
    def update(self, round_num: int, parameters: List[np.ndarray], accuracy: float) -> bool:
        is_best = False
        
        self.accuracy_history.append(accuracy)
        
        if accuracy > self.best_accuracy:
            improvement = accuracy - self.best_accuracy
            
            if improvement > self.convergence_threshold:
                self.last_significant_improvement = round_num
                self.no_improvement_count = 0
            
            self.best_accuracy = accuracy
            self.best_round = round_num
            
            net = instantiate(self.model_config).to(self.device)
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            net.load_state_dict(state_dict, strict=True)
            
            self.best_model_state = copy.deepcopy(net.state_dict())
            is_best = True
            
            self._save_checkpoint({
                'round': round_num,
                'state_dict': self.best_model_state,
                'accuracy': accuracy,
            })
            
            print(f"Round {round_num}: New best model (Accuracy: {accuracy:.4f})")
        else:
            self.no_improvement_count += 1
            
            if self.convergence_round is None and self.no_improvement_count >= self.convergence_patience:
                self.convergence_round = self.last_significant_improvement
                print(f"Convergence detected! Convergence round: {self.convergence_round}")
        
        return is_best
    
    def _save_checkpoint(self, state: Dict[str, Any]) -> None:
        checkpoint_path = os.path.join(self.output_dir, 'best_model.pth')
        torch.save(state, checkpoint_path)
        print(f"Best model checkpoint saved: {checkpoint_path}")
    
    def evaluate_best_model(self) -> Dict[str, float]:
        if self.best_model_state is None:
            print("No best model found for evaluation")
            return {}
        
        net = instantiate(self.model_config).to(self.device)
        net.load_state_dict(self.best_model_state)
        
        loss, accuracy, metrics = test_with_metrics(net, self.testloader, self.device)
        
        metrics.update({
            'best_round': self.best_round,
            'convergence_round': self.convergence_round if self.convergence_round is not None else -1,
            'loss': loss,
            'accuracy': accuracy,
        })
        
        metrics_path = os.path.join(self.output_dir, 'best_model_metrics.pkl')
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)
        
        print("\n" + "="*50)
        print(f"Best Model Evaluation (Round {self.best_round}):")
        print(f"Accuracy: {accuracy:.4f}")
        
        f1_score = metrics.get('f1_score', 'N/A')
        precision = metrics.get('precision', 'N/A')
        recall = metrics.get('recall', 'N/A')
        
        print(f"F1 Score: {f1_score if isinstance(f1_score, str) else f'{f1_score:.4f}'}")
        print(f"Precision: {precision if isinstance(precision, str) else f'{precision:.4f}'}")
        print(f"Recall: {recall if isinstance(recall, str) else f'{recall:.4f}'}")
        print("="*50 + "\n")
        
        return metrics
    
    def track_round_metrics(self, round_num: int, metrics: Dict[str, float], evaluation_type: str = "centralized") -> None:
        round_data = {'round': round_num, 'evaluation_type': evaluation_type, **metrics}
        self.metrics_history.append(round_data)
        
        if evaluation_type == "federated":
            self.federated_evaluation_history.append(round_data)
        else:
            self.centralized_evaluation_history.append(round_data)
        
        history_path = os.path.join(self.output_dir, 'metrics_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.metrics_history, f)
    
    def get_convergence_info(self) -> Dict[str, Any]:
        if self.convergence_round is None:
            self.convergence_round = self.best_round
        
        return {
            'convergence_round': self.convergence_round,
            'best_round': self.best_round,
            'best_accuracy': self.best_accuracy,
            'accuracy_history': self.accuracy_history,
            'federated_evaluations': len(self.federated_evaluation_history),
            'centralized_evaluations': len(self.centralized_evaluation_history),
        }


def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    model: DictConfig,
):

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        net = instantiate(model)
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        
        state_dict = {}
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
        
        net.load_state_dict(state_dict, strict=False)
        net.to(device)

        from models import test
        loss, accuracy = test(net, testloader, device=device)
        return loss, {"accuracy": accuracy}

    return evaluate

def weighted_average_eval(metrics):
    """Federated evaluation metrics aggregation function"""
    total_examples = 0
    acc_sum = 0.0
    for num_examples, m in metrics:
        acc = float(m.get("accuracy", 0.0))
        total_examples += int(num_examples)
        acc_sum += acc * int(num_examples)
        
    final_accuracy = (acc_sum / total_examples) if total_examples > 0 else 0.0
    return {"accuracy": round(final_accuracy, 4)}

def customize_server(server: Server, model_tracker) -> Server:
    orig_evaluate_round = server.evaluate_round
    
    def enhanced_evaluate_round(server_round: int, timeout: Optional[float] = None) -> Optional[Tuple[Optional[float], Dict[str, Scalar]]]:
        result = orig_evaluate_round(server_round, timeout)
        
        if result is not None:
            if len(result) == 2:
                loss, metrics = result
            elif len(result) == 3:
                loss, metrics, _ = result
            else:
                print(f"Warning: Unexpected result format: {result}")
                return result
                
            accuracy = metrics.get("accuracy", 0.0)
            evaluation_type = metrics.get("evaluation_type", "unknown")
            
            if hasattr(server.parameters, 'tensors'):
                parameters_ndarrays = parameters_to_ndarrays(server.parameters)
            else:
                parameters_ndarrays = server.parameters
            
            proximal_mu = getattr(server.strategy, 'proximal_mu', 'N/A')
            strategy_name = server.strategy.__class__.__name__
            
            if evaluation_type == "centralized":
                print(f"Round {server_round}: Centralized evaluation completed ({strategy_name} μ={proximal_mu}, Accuracy: {accuracy:.4f})")
                model_tracker.update(server_round, parameters_ndarrays, accuracy, "centralized")
                model_tracker.track_round_metrics(server_round, metrics, "centralized")
            else:
                fed_accuracy = accuracy
                if fed_accuracy == 0.0 and isinstance(metrics, dict):
                    client_accuracies = []
                    for key, value in metrics.items():
                        if 'accuracy' in str(key).lower() and isinstance(value, (int, float)) and value > 0:
                            client_accuracies.append(value)
                    if client_accuracies:
                        fed_accuracy = sum(client_accuracies) / len(client_accuracies)
                
                print(f"Round {server_round}: Federated evaluation completed (Accuracy: {fed_accuracy:.4f})")
                model_tracker.track_round_metrics(server_round, metrics, "federated")
        
        return result
    
    server.evaluate_round = enhanced_evaluate_round
    
    return server


@hydra.main(config_path="conf", config_name="fedprox_base", version_base=None)
def main(cfg: DictConfig) -> None:
    
    if "mnist" in cfg.dataset_name:
        cfg.model.input_dim = 256
        cfg.model._target_ = "models.CNNMnist"
    print(OmegaConf.to_yaml(cfg))

    experiment_start_time = time.time()

    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = DictConfig(cfg)

    trainloaders, valloaders, testloader = load_datasets(
        config=cfg.dataset,
        num_clients=cfg.num_clients,
        val_ratio=cfg.dataset.val_split,
    )

    client_fn = call(
        cfg.client_fn,
        trainloaders,
        valloaders,
        model=cfg.model,
    )

    device = cfg.server_device
    
    initial_model = load_initial_weights(cfg, device)
    if initial_model is not None:
        initial_parameters = [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
        print(f"✓ Initial weights loaded - {len(initial_parameters)} parameter groups")
        
        evaluate_fn = gen_evaluate_fn(testloader, device=device, model=cfg.model)
        
        initial_loss, initial_metrics = evaluate_fn(0, initial_parameters, {})
        print(f"Initial model accuracy (with initial weights): {initial_metrics.get('accuracy', 0.0):.4f}")
    else:
        evaluate_fn = gen_evaluate_fn(testloader, device=device, model=cfg.model)
        initial_parameters = None

    save_path = HydraConfig.get().runtime.output_dir
    model_tracker = BestModelTracker(
        model_config=cfg.model,
        testloader=testloader,
        device=device,
        output_dir=save_path,
        convergence_patience=5,
        convergence_threshold=0.001
    )

    original_evaluate_fn = evaluate_fn
    
    def enhanced_evaluate_fn(server_round, parameters, config):
        loss, metrics = original_evaluate_fn(server_round, parameters, config)
        
        metrics["evaluation_type"] = "centralized"
        
        if "accuracy" in metrics:
            if hasattr(parameters, 'tensors'):
                parameters_ndarrays = parameters_to_ndarrays(parameters)
            else:
                parameters_ndarrays = parameters
                
            model_tracker.update(server_round, parameters_ndarrays, metrics["accuracy"])
        
        model_tracker.track_round_metrics(server_round, metrics)
        
        return loss, metrics
    
    evaluate_fn = enhanced_evaluate_fn

    strategy_kwargs = {
        "evaluate_fn": evaluate_fn,
        "evaluate_metrics_aggregation_fn": weighted_average_eval,
    }
    
    if initial_parameters is not None:
        strategy_kwargs["initial_parameters"] = ndarrays_to_parameters(initial_parameters)
        print("✓ Strategy configured with initial weights")
    
    strategy = instantiate(cfg.strategy, **strategy_kwargs)

    server = Server(strategy=strategy, client_manager=SimpleClientManager())
    server = customize_server(server, model_tracker)

    try:
        history = fl.simulation.start_simulation(
            server=server,
            client_fn=client_fn,
            num_clients=cfg.num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            client_resources={
                "num_cpus": cfg.client_resources.num_cpus,
                "num_gpus": cfg.client_resources.num_gpus,
            },
            strategy=strategy,
        )
        
        experiment_end_time = time.time()
        total_experiment_time = experiment_end_time - experiment_start_time

        print("\nFinal evaluation of the best model...")
        best_model_metrics = model_tracker.evaluate_best_model()

        convergence_info = model_tracker.get_convergence_info()

        print(f"\nSaving results to: {save_path}")
        
        with open(os.path.join(save_path, "history.pkl"), "wb") as f_ptr:
            pickle.dump(history, f_ptr)
        
        proximal_mu = cfg.get('proximal_mu', getattr(cfg.get('strategy', {}), 'proximal_mu', 'N/A'))
        
        summary = {
            "config": OmegaConf.to_container(cfg),
            "best_model_metrics": best_model_metrics,
            "convergence_info": convergence_info,
            "total_experiment_time": total_experiment_time,
            "num_clients": cfg.num_clients,
            "num_rounds": cfg.num_rounds,
            "dataset": cfg.dataset_name,
            "partitioning": cfg.partitioning,
            "algorithm": "FedProx",
            "proximal_mu": proximal_mu,
            "initial_weights_used": cfg.get("initial_weights", {}).get("enabled", False),
            "validation_enabled": cfg.dataset.get("val_split", 0.2) > 0.0,
        }
        
        with open(os.path.join(save_path, "experiment_summary.pkl"), "wb") as f_ptr:
            pickle.dump(summary, f_ptr)
        
        print("\n" + "="*60)
        print(f"EXPERIMENT SUMMARY (FedProx)")
        print("="*60)
        print(f"Dataset: {cfg.dataset_name}")
        print(f"Data partitioning: {cfg.partitioning}")
        print(f"Number of clients: {cfg.num_clients}")
        print(f"Number of rounds: {cfg.num_rounds}")
        print(f"Proximal μ (mu): {proximal_mu}")
        print(f"Total experiment time: {total_experiment_time:.2f} seconds")
        
        if cfg.get("initial_weights", {}).get("enabled", False):
            print(f"Initial weights: USED")
        else:
            print("Initial weights: NOT USED (started with random weights)")
        
        if cfg.dataset.get("val_split", 0.2) == 0.0:
            print("Validation: DISABLED (all data used for training)")
        else:
            print(f"Validation: ENABLED (val_split: {cfg.dataset.val_split})")
        
        print("\nBest Model Metrics:")
        print(f"  Round: {best_model_metrics.get('best_round', 'N/A')}")
        print(f"  Convergence Round: {best_model_metrics.get('convergence_round', 'N/A')}")
        print(f"  Accuracy: {best_model_metrics.get('accuracy', 0.0):.4f}")
        
        f1_score = best_model_metrics.get('f1_score', 'N/A')
        precision = best_model_metrics.get('precision', 'N/A')
        recall = best_model_metrics.get('recall', 'N/A')
        
        print(f"  F1 Score: {f1_score if isinstance(f1_score, str) else f'{f1_score:.4f}'}")
        print(f"  Precision: {precision if isinstance(precision, str) else f'{precision:.4f}'}")
        print(f"  Recall: {recall if isinstance(recall, str) else f'{recall:.4f}'}")
        print("="*60)
        
    except Exception as e:
        print(f"An error occurred during simulation: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nError occurred, but trying to evaluate the best model...")
        
        try:
            best_model_metrics = model_tracker.evaluate_best_model()
            
            if best_model_metrics:
                print("\n" + "="*60)
                print("EXPERIMENT SUMMARY (After Error)")
                print("="*60)
                print("\nBest Model Metrics:")
                print(f"  Round: {best_model_metrics.get('best_round', 'N/A')}")
                print(f"  Convergence Round: {best_model_metrics.get('convergence_round', 'N/A')}")
                print(f"  Accuracy: {best_model_metrics.get('accuracy', 0.0):.4f}")
                
                f1_score = best_model_metrics.get('f1_score', 'N/A')
                precision = best_model_metrics.get('precision', 'N/A')
                recall = best_model_metrics.get('recall', 'N/A')
                
                print(f"  F1 Score: {f1_score if isinstance(f1_score, str) else f'{f1_score:.4f}'}")
                print(f"  Precision: {precision if isinstance(precision, str) else f'{precision:.4f}'}")
                print(f"  Recall: {recall if isinstance(recall, str) else f'{recall:.4f}'}")
                print("="*60)
        except Exception as e2:
            print(f"Second error occurred while evaluating best model: {e2}")


if __name__ == "__main__":
    main()