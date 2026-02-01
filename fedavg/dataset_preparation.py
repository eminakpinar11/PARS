import os

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST


def _download_data(dataset_name="emnist") -> Tuple[Dataset, Dataset]:
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

    trainset, testset = None, None
    if dataset_name == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: F.pad(
                        Variable(x.unsqueeze(0), requires_grad=False),
                        (4, 4, 4, 4),
                        mode="reflect",
                    ).data.squeeze()
                ),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        trainset = CIFAR10(
            root= data_root,
            train=True,
            download=True,
            transform=transform_train,
        )
        testset = CIFAR10(
            root= data_root,
            train=False,
            download=True,
            transform=transform_test,
        )
    elif dataset_name == "mnist":
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        trainset = MNIST(
            root= data_root,
            train=True,
            download=True,
            transform=transform_train,
        )
        testset = MNIST(
            root= data_root,
            train=False,
            download=True,
            transform=transform_test,
        )
    elif dataset_name == "fmnist":
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        trainset = FashionMNIST(
            root= data_root,
            train=True,
            download=True,
            transform=transform_train,
        )
        testset = FashionMNIST(
            root= data_root,
            train=False,
            download=True,
            transform=transform_test,
        )
    else:
        raise NotImplementedError

    return trainset, testset


def partition_data(
    num_clients, similarity=1.0, seed=42, dataset_name="cifar10"
) -> Tuple[List[Dataset], Dataset]:
    trainset, testset = _download_data(dataset_name)
    trainsets_per_client = []
    s_fraction = int(similarity * len(trainset))
    prng = np.random.default_rng(seed)
    idxs = prng.choice(len(trainset), s_fraction, replace=False)
    iid_trainset = Subset(trainset, idxs)
    rem_trainset = Subset(trainset, np.setdiff1d(np.arange(len(trainset)), idxs))

    all_ids = np.arange(len(iid_trainset))
    splits = np.array_split(all_ids, num_clients)
    for i in range(num_clients):
        c_ids = splits[i]
        d_ids = iid_trainset.indices[c_ids]
        trainsets_per_client.append(Subset(iid_trainset.dataset, d_ids))

    if similarity == 1.0:
        return trainsets_per_client, testset

    tmp_t = rem_trainset.dataset.targets
    if isinstance(tmp_t, list):
        tmp_t = np.array(tmp_t)
    if isinstance(tmp_t, torch.Tensor):
        tmp_t = tmp_t.numpy()
    targets = tmp_t[rem_trainset.indices]
    num_remaining_classes = len(set(targets))
    remaining_classes = list(set(targets))
    client_classes: List[List] = [[] for _ in range(num_clients)]
    times = [0 for _ in range(num_remaining_classes)]

    for i in range(num_clients):
        client_classes[i] = [remaining_classes[i % num_remaining_classes]]
        times[i % num_remaining_classes] += 1
        j = 1
        while j < 2:
            index = prng.choice(num_remaining_classes)
            class_t = remaining_classes[index]
            if class_t not in client_classes[i]:
                client_classes[i].append(class_t)
                times[index] += 1
                j += 1

    rem_trainsets_per_client: List[List] = [[] for _ in range(num_clients)]

    for i in range(num_remaining_classes):
        class_t = remaining_classes[i]
        idx_k = np.where(targets == class_t)[0]
        prng.shuffle(idx_k)
        idx_k_split = np.array_split(idx_k, times[i])
        ids = 0
        for j in range(num_clients):
            if class_t in client_classes[j]:
                act_idx = rem_trainset.indices[idx_k_split[ids]]
                rem_trainsets_per_client[j].append(
                    Subset(rem_trainset.dataset, act_idx)
                )
                ids += 1

    for i in range(num_clients):
        trainsets_per_client[i] = ConcatDataset(
            [trainsets_per_client[i]] + rem_trainsets_per_client[i]
        )

    return trainsets_per_client, testset


def partition_data_dirichlet(
    num_clients, alpha, seed=42, dataset_name="cifar10"
) -> Tuple[List[Dataset], Dataset]:
    trainset, testset = _download_data(dataset_name)
    min_required_samples_per_client = 10
    min_samples = 0
    prng = np.random.default_rng(seed)

    tmp_t = trainset.targets
    if isinstance(tmp_t, list):
        tmp_t = np.array(tmp_t)
    if isinstance(tmp_t, torch.Tensor):
        tmp_t = tmp_t.numpy()
    num_classes = len(set(tmp_t))
    total_samples = len(tmp_t)
    while min_samples < min_required_samples_per_client:
        idx_clients: List[List] = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(tmp_t == k)[0]
            prng.shuffle(idx_k)
            proportions = prng.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array(
                [
                    p * (len(idx_j) < total_samples / num_clients)
                    for p, idx_j in zip(proportions, idx_clients)
                ]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_k_split = np.split(idx_k, proportions)
            idx_clients = [
                idx_j + idx.tolist() for idx_j, idx in zip(idx_clients, idx_k_split)
            ]
            min_samples = min([len(idx_j) for idx_j in idx_clients])

    trainsets_per_client = [Subset(trainset, idxs) for idxs in idx_clients]
    return trainsets_per_client, testset


def partition_data_label_quantity(
    num_clients, labels_per_client, seed=42, dataset_name="cifar10"
) -> Tuple[List[Dataset], Dataset]:
    trainset, testset = _download_data(dataset_name)
    prng = np.random.default_rng(seed)

    targets = trainset.targets
    if isinstance(targets, list):
        targets = np.array(targets)
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    num_classes = len(set(targets))
    times = [0 for _ in range(num_classes)]
    contains = []

    for i in range(num_clients):
        current = [i % num_classes]
        times[i % num_classes] += 1
        j = 1
        while j < labels_per_client:
            index = prng.choice(num_classes, 1)[0]
            if index not in current:
                current.append(index)
                times[index] += 1
                j += 1
        contains.append(current)
    idx_clients: List[List] = [[] for _ in range(num_clients)]
    for i in range(num_classes):
        idx_k = np.where(targets == i)[0]
        prng.shuffle(idx_k)
        idx_k_split = np.array_split(idx_k, times[i])
        ids = 0
        for j in range(num_clients):
            if i in contains[j]:
                idx_clients[j] += idx_k_split[ids].tolist()
                ids += 1
    trainsets_per_client = [Subset(trainset, idxs) for idxs in idx_clients]
    return trainsets_per_client, testset

def partition_data_stratified_iid(
    num_clients, seed=42, dataset_name="cifar10"
) -> Tuple[List[Dataset], Dataset]:

    trainset, testset = _download_data(dataset_name)
    prng = np.random.default_rng(seed)
    
    tmp_t = trainset.targets
    if isinstance(tmp_t, list):
        tmp_t = np.array(tmp_t)
    if isinstance(tmp_t, torch.Tensor):
        tmp_t = tmp_t.numpy()
    
    num_classes = len(set(tmp_t))
    idx_clients: List[List] = [[] for _ in range(num_clients)]
    
    for k in range(num_classes):
        idx_k = np.where(tmp_t == k)[0]
        prng.shuffle(idx_k)
        
        samples_per_client = len(idx_k) // num_clients
        remainder = len(idx_k) % num_clients
        
        start_idx = 0
        for i in range(num_clients):
            end_idx = start_idx + samples_per_client + (1 if i < remainder else 0)
            idx_clients[i].extend(idx_k[start_idx:end_idx])
            start_idx = end_idx
    
    for i in range(num_clients):
        prng.shuffle(idx_clients[i])
    
    trainsets_per_client = [Subset(trainset, idxs) for idxs in idx_clients]
    return trainsets_per_client, testset