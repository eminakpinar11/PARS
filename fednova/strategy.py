from functools import reduce
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


class FedNovaStrategy(FedAvg):

    def aggregate_fit_custom(
        self,
        server_round: int,
        server_params: NDArrays,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        total_samples = sum([fit_res.num_examples for _, fit_res in results])
        c_fact = sum(
            [
                float(fit_res.metrics["a_i"]) * fit_res.num_examples / total_samples
                for _, fit_res in results
            ]
        )
        new_weights_results = [
            (result[0], c_fact * (fit_res.num_examples / total_samples))
            for result, (_, fit_res) in zip(weights_results, results)
        ]

        grad_updates_aggregated = aggregate_fednova(new_weights_results)
        aggregated = [
            server_param - grad_update
            for server_param, grad_update in zip(server_params, grad_updates_aggregated)
        ]

        parameters_aggregated = ndarrays_to_parameters(aggregated)
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated


def aggregate_fednova(results: List[Tuple[NDArrays, float]]) -> NDArrays:
    weighted_weights = [
        [layer * factor for layer in weights] for weights, factor in results
    ]

    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime