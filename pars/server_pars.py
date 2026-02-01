import concurrent.futures
from logging import DEBUG, INFO
from typing import OrderedDict

import torch

from flwr.common import (
    Code,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from typing import Dict, List, Tuple, Optional, Union
from flwr.common.logger import log
from flwr.common.typing import (
    Callable,
    GetParametersIns,
    NDArrays,
)
from flwr.server import Server
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from models import test

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]


class ParsServer(Server):

    def __init__(
        self,
        strategy: Strategy,
        model: DictConfig,
        client_manager: Optional[ClientManager] = None,
    ):
        if client_manager is None:
            client_manager = SimpleClientManager()
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.model_params = instantiate(model)
        self.server_cv: List[torch.Tensor] = []

    def _get_initial_parameters(self, timeout: Optional[float], server_round: int = 0) -> Parameters:
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            parameters_np = parameters_to_ndarrays(parameters)
            self.server_cv = [torch.from_numpy(t) for t in parameters_np]
            print(f"✓ Server CV initialized with {len(self.server_cv)} parameter groups from strategy")
            return parameters

        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(ins=ins, timeout=timeout, group_id="default")
        log(INFO, "Received initial parameters from one random client")
        self.server_cv = [
            torch.from_numpy(t)
            for t in parameters_to_ndarrays(get_parameters_res.parameters)
        ]
        print(f"✓ Server CV initialized with {len(self.server_cv)} parameter groups from client")
        return get_parameters_res.parameters

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=update_parameters_with_cv(self.parameters, self.server_cv),
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
        )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )
    
        if not results:
            print("No successful client results. Skipping aggregation.")
            return self.parameters, {}, (results, failures)
    
        aggregated_result: Tuple[Optional[Parameters], Dict[str, Scalar]] = (
            self.strategy.aggregate_fit(server_round, results, failures)
        )
    
        aggregated_result_arrays_combined = []
        if aggregated_result[0] is not None:
            aggregated_result_arrays_combined = parameters_to_ndarrays(aggregated_result[0])
        
        print(f"Aggregated result length: {len(aggregated_result_arrays_combined)}")
        
        if len(aggregated_result_arrays_combined) == 0:
            print("No aggregated results received. Keeping current parameters and CV.")
            parameters_updated = self.parameters
            return parameters_updated, aggregated_result[1], (results, failures)
        
        aggregated_parameters = aggregated_result_arrays_combined[: len(aggregated_result_arrays_combined) // 2]
        print(f"Aggregated parameters length: {len(aggregated_parameters)}")
        
        aggregated_cv_update = aggregated_result_arrays_combined[len(aggregated_result_arrays_combined) // 2 :]
        print(f"Aggregated CV updates length: {len(aggregated_cv_update)}")
        
        server_cv_np = [cv.cpu().numpy() if cv.is_cuda else cv.numpy() for cv in self.server_cv]
        
        print(f"Server CV length: {len(self.server_cv)}")
        print(f"Server CV numpy length: {len(server_cv_np)}")

        if len(server_cv_np) != len(aggregated_cv_update):
            print(f"WARNING: CV dimensions don't match. server_cv_np={len(server_cv_np)}, aggregated_cv_update={len(aggregated_cv_update)}")

            if len(aggregated_cv_update) == 0:
                print("aggregated_cv_update is empty, server_cv not changed")
                return self.parameters, aggregated_result[1], (results, failures)
            else:
                min_length = min(len(server_cv_np), len(aggregated_cv_update))
                print(f"Using minimum common length: {min_length}")

                total_clients = len(self._client_manager.all())
                cv_multiplier = len(results) / total_clients
                self.server_cv = [
                    torch.from_numpy(cv + cv_multiplier * aggregated_cv_update[i] if i < len(aggregated_cv_update) else cv)
                    for i, cv in enumerate(server_cv_np[:min_length])
                ]
        else:
            total_clients = len(self._client_manager.all())
            cv_multiplier = len(results) / total_clients
            self.server_cv = [
                torch.from_numpy(cv + cv_multiplier * aggregated_cv_update[i])
                for i, cv in enumerate(server_cv_np)
            ]
    
        curr_params = parameters_to_ndarrays(self.parameters)
        updated_params = [
            x + aggregated_parameters[i] for i, x in enumerate(curr_params)
        ]
        parameters_updated = ndarrays_to_parameters(updated_params)
    
        metrics_aggregated = aggregated_result[1]
        return parameters_updated, metrics_aggregated, (results, failures)


def update_parameters_with_cv(
    parameters: Parameters, s_cv: List[torch.Tensor]
) -> Parameters:
    cv_np = [cv.cpu().numpy() if cv.is_cuda else cv.numpy() for cv in s_cv]
    parameters_np = parameters_to_ndarrays(parameters)
    parameters_np.extend(cv_np)
    return ndarrays_to_parameters(parameters_np)


def fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> FitResultsAndFailures:
    print(f"fit_clients called with {len(client_instructions)} instructions")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(fit_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        print(f"Submitted {len(submitted_fs)} client fit tasks")
        
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,
        )
        print(f"Finished {len(finished_fs)} client fit tasks")

    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_fit(
            future=future, results=results, failures=failures
        )
    
    print(f"Results: {len(results)}, Failures: {len(failures)}")
    return results, failures


def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float]
) -> Tuple[ClientProxy, FitRes]:
    try:
        fit_res = client.fit(ins, timeout=timeout, group_id="")
        return client, fit_res
    except TypeError as e:
        if "missing 1 required positional argument: 'group_id'" in str(e):
            print("Handling missing group_id parameter")
            fit_res = client.fit(ins, timeout=timeout)
            return client, fit_res
        else:
            raise


def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
) -> None:
    failure = future.exception()
    if failure is not None:
        print(f"Client fit exception: {str(failure)}")
        import traceback
        traceback.print_exc()
        failures.append(failure)
        return

    result: Tuple[ClientProxy, FitRes] = future.result()
    _, res = result

    if res.status.code == Code.OK:
        results.append(result)
        return

    print(f"Client fit failure: {res.status.code} - {res.status.message}")
    failures.append(result)


def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    model: DictConfig,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        net = instantiate(model)
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        loss, accuracy = test(net, testloader, device=device)
        return loss, {"accuracy": accuracy}

    return evaluate