from logging import DEBUG, INFO
from typing import Dict, Tuple, Optional

from flwr.common import parameters_to_ndarrays, Parameters, Scalar
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.server import FitResultsAndFailures, Server, fit_clients


class FedNovaServer(Server):

    def __init__(
        self,
        *,
        client_manager: ClientManager,
        strategy: Optional[object] = None,
    ) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.strategy = strategy if strategy is not None else None

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id="0",
        )

        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        params_np = parameters_to_ndarrays(self.parameters)
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit_custom(
            server_round, params_np, results, failures
        )

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)