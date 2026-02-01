from logging import WARNING
import numpy as np
from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from typing import Dict, List, Tuple, Optional, Union
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate


class ScaffoldStrategy(FedAvg):

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print(f"ScaffoldStrategy.aggregate_fit called with {len(results)} results and {len(failures)} failures")
        
        if not results:
            print("No results to aggregate.")
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        combined_parameters_all_updates = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        len_combined_parameter = len(combined_parameters_all_updates[0])
        num_examples_all_updates = [fit_res.num_examples for _, fit_res in results]
        weights_results = [
            (update[: len_combined_parameter // 2], num_examples)
            for update, num_examples in zip(
                combined_parameters_all_updates, num_examples_all_updates
            )
        ]
        parameters_aggregated = aggregate(weights_results)

        client_cv_updates_and_num_examples = [
            (update[len_combined_parameter // 2 :], num_examples)
            for update, num_examples in zip(
                combined_parameters_all_updates, num_examples_all_updates
            )
        ]
        aggregated_cv_update = aggregate(client_cv_updates_and_num_examples)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return (
            ndarrays_to_parameters(parameters_aggregated + aggregated_cv_update),
            metrics_aggregated,
        )
    
    def aggregate_evaluate(self, server_round: int, results, failures):
        """SCAFFOLD federated evaluation aggregator"""
        if not results:
            return None, {}
        
        total_examples = 0
        acc_sum = 0.0
        successful_evals = 0
        
        for _, eval_res in results:
            if eval_res.metrics and eval_res.metrics.get("accuracy", 0.0) > 0:
                num_examples = eval_res.num_examples
                acc = eval_res.metrics["accuracy"]
                total_examples += num_examples
                acc_sum += acc * num_examples
                successful_evals += 1
        
        weighted_accuracy = acc_sum / total_examples if total_examples > 0 else 0.0
        
        total_loss = sum(res.loss * res.num_examples for _, res in results if res.loss > 0)
        total_loss_examples = sum(res.num_examples for _, res in results if res.loss > 0)
        aggregated_loss = total_loss / total_loss_examples if total_loss_examples > 0 else 0.0
        
        return aggregated_loss, {
            "accuracy": round(weighted_accuracy, 4),
            "successful_evaluations": successful_evals,
            "total_clients": len(results)
        }