from typing import List, Optional, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split

from dataset_preparation import (
    partition_data,
    partition_data_dirichlet,
    partition_data_label_quantity,
    partition_data_stratified_iid,
)


def load_datasets(
    config: DictConfig,
    num_clients: int,
    val_ratio: float = 0.1,
    seed: Optional[int] = 42,
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:

    print(f"Dataset partitioning config: {config}")
    partitioning = ""
    if "partitioning" in config:
        partitioning = config.partitioning
    if partitioning == "dirichlet":
        alpha = 0.5
        if "alpha" in config:
            alpha = config.alpha
        datasets, testset = partition_data_dirichlet(
            num_clients,
            alpha=alpha,
            seed=seed,
            dataset_name=config.name,
        )
    elif partitioning == "label_quantity":
        labels_per_client = 2
        if "labels_per_client" in config:
            labels_per_client = config.labels_per_client
        datasets, testset = partition_data_label_quantity(
            num_clients,
            labels_per_client=labels_per_client,
            seed=seed,
            dataset_name=config.name,
        )
    elif partitioning == "iid":
        datasets, testset = partition_data_stratified_iid(
            num_clients,
            seed=seed,
            dataset_name=config.name,
        )
    elif partitioning == "non_stratified_iid":
        datasets, testset = partition_data(
            num_clients,
            similarity=1.0,
            seed=seed,
            dataset_name=config.name,
        )
    elif partitioning == "iid_noniid":
        similarity = 0.5
        if "similarity" in config:
            similarity = config.similarity
        datasets, testset = partition_data(
            num_clients,
            similarity=similarity,
            seed=seed,
            dataset_name=config.name,
        )

    batch_size = -1
    if "batch_size" in config:
        batch_size = config.batch_size
    elif "batch_size_ratio" in config:
        batch_size_ratio = config.batch_size_ratio
    else:
        raise ValueError

    trainloaders = []
    valloaders = []
    for dataset in datasets:
        len_val = int(len(dataset) / (1 / val_ratio)) if val_ratio > 0 else 0
        lengths = [len(dataset) - len_val, len_val]
        ds_train, ds_val = random_split(
            dataset, lengths, torch.Generator().manual_seed(seed)
        )
        if batch_size == -1:
            batch_size = int(len(ds_train) * batch_size_ratio)
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    return trainloaders, valloaders, DataLoader(testset, batch_size=len(testset))