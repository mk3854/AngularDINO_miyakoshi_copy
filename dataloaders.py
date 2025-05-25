import os
import random
from typing import Any, Tuple
from PIL import Image

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
from torchvision import datasets, transforms
from utils_extra import split

# =========================
# カスタムデータセットクラス
# =========================
class MNIST(datasets.MNIST):
    """
    MNISTデータセットのカスタムラッパークラス。
    __getitem__でPIL画像に変換し、transformを適用。
    """
    def __init__(self, root, train = True, transform = None, target_transform = None, download = False):
        super().__init__(root, train, transform, target_transform, download)
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy(), mode="L")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

class FashionMNIST(datasets.FashionMNIST):
    """
    FashionMNISTデータセットのカスタムラッパークラス。
    __getitem__でPIL画像に変換し、transformを適用。
    """
    def __init__(self, root, train = True, transform = None, target_transform = None, download = False):
        super().__init__(root, train, transform, target_transform, download)
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy(), mode="L")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

class SVHN(datasets.SVHN):
    """
    SVHNデータセットのカスタムラッパークラス。
    __getitem__でPIL画像に変換し、transformを適用。
    """
    def __init__(self, root, split = "train", transform = None, target_transform = None, download = False):
        super().__init__(root, split, transform, target_transform, download)
    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.labels[index])
        img = Image.fromarray(np.transpose(img, (1,2,0)))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

# =========================
# toyデータ生成関数
# =========================
def make_toy_data(
    n_normal: int = 900,
    n_labeled_anomaly: int = 20,
    n_unlabeled_anomaly: int = 80,
    is_train=True,
    batch_size: int = 128,
) -> DataLoader:
    """
    2次元のtoyデータセットを生成しDataLoaderで返す。
    """
    X_unlabeled_normal = np.zeros((n_normal, 2))
    for i, x in enumerate(np.linspace(-np.pi, np.pi, n_normal)):
        X_unlabeled_normal[i, 0] = x + np.random.normal(0, 0.1)
        X_unlabeled_normal[i, 1] = 3.0 * (np.sin(x) + np.random.normal(0, 0.2))

    X_unlabeled_anomaly = np.concatenate(
        [
            np.random.multivariate_normal(
                mean=[-np.pi / 2, np.pi / 2],
                cov=0.1 * np.eye(2),
                size=int(n_unlabeled_anomaly / 2),
            ),
            np.random.multivariate_normal(
                mean=[np.pi / 2, -np.pi / 2],
                cov=0.1 * np.eye(2),
                size=int(n_unlabeled_anomaly / 2),
            ),
        ]
    )

    X_anomaly = np.concatenate(
        [
            np.random.multivariate_normal(
                mean=[-np.pi / 2, np.pi / 2],
                cov=0.1 * np.eye(2),
                size=int(n_labeled_anomaly / 2),
            ),
            np.random.multivariate_normal(
                mean=[np.pi / 2, -np.pi / 2],
                cov=0.1 * np.eye(2),
                size=int(n_labeled_anomaly / 2),
            ),
        ]
    )

    X = np.concatenate([X_unlabeled_normal, X_unlabeled_anomaly, X_anomaly])
    if is_train:
        y = np.concatenate(
            [
                np.zeros(shape=n_normal),
                np.zeros(shape=n_unlabeled_anomaly),
                np.ones(shape=n_labeled_anomaly),
            ]
        )
    else:
        y = np.concatenate(
            [
                np.zeros(shape=n_normal),
                np.ones(shape=n_unlabeled_anomaly),
                np.ones(shape=n_labeled_anomaly),
            ]
        )

    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.int32))
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

# =========================
# メインデータローダ生成関数
# =========================
def load(
    name: str,
    normal_class: list = [0],
    unseen_anomaly: list = [9],
    labeled_anomaly_class: int | list = None,
    n_test: int = 2000,
    n_unlabeled_normal: int = 4500,
    n_unlabeled_anomaly: int = 250,
    n_labeled_anomaly: int = 250,
    train_ratio: float = 0.9, #train=4500, valid=500
    transform: transforms = None, 
    return_type: str ="train",
    seed: int = 0,
    return_id: bool = False,
) -> tuple[DataLoader, ...]:
    """
    指定したデータセット名・条件に応じて、学習/検証/テスト用のDataLoaderやSubsetを返す。
    返り値は指定フラグによって変化。
    """
    # --- シード固定 ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- データセットパスの決定 ---
    path = os.path.join("datasets", name) if name in ["CIFAR10", "SVHN"] else "datasets"
    os.makedirs(path, exist_ok=True)

    # --- transformの決定 ---
    if transform is not None:
        pass
    elif name in ["CIFAR10", "SVHN"]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32), antialias=True),
            transforms.Grayscale(),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ])
    else:  # MNIST, FashionMNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32), antialias=True),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ])
    if name=="MNIST":
        stats = ((0.1307,), (0.3081,))
    elif name=="CIFAR10":
        stats = ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    elif name=="FashionMNIST":
        stats = ((0.2860,), (0.3530,))
    elif name=="SVHN":
        stats = ((0.4377, 0.4438, 0.4728), (0.198, 0.201, 0.197))
    else:
        raise ValueError("Invalid dataset name")
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    # --- データセットのロード ---
    if name == "MNIST":
        train = MNIST(root=path, download=True, train=True, transform=transform)
        test = datasets.MNIST(root=path, download=True, train=False, transform=normalize)
        train_normalize = datasets.MNIST(root=path, download=True, train=True, transform=normalize)
    elif name == "FashionMNIST":
        train = FashionMNIST(root=path, download=True, train=True, transform=transform)
        test = datasets.FashionMNIST(root=path, download=True, train=False, transform=normalize)
        train_normalize = datasets.FashionMNIST(root=path, download=True, train=True, transform=normalize)
    elif name == "CIFAR10":
        train = datasets.CIFAR10(root=path, download=True, train=True, transform=transform)
        test = datasets.CIFAR10(root=path, download=True, train=False, transform=normalize)
        train_normalize = datasets.CIFAR10(root=path, download=True, train=True, transform=normalize)
    elif name == "SVHN":
        train = SVHN(root=path, download=True, split="train", transform=transform)
        test = datasets.SVHN(root=path, download=True, split="test", transform=normalize)
        train_normalize = datasets.SVHN(root=path, download=True, split="train", transform=normalize)
    else:
        raise ValueError("Invalid dataset name")

    # --- インデックス取得 ---
    train_indices = train.targets if name != "SVHN" else train.labels
    if not torch.is_tensor(train_indices):
        train_indices = torch.tensor(train_indices)

    # --- 正常データのindex取得 ---
    if isinstance(normal_class, list):
        mask = torch.isin(train_indices, torch.tensor(normal_class))
        train_normal_indices = mask.nonzero().squeeze().tolist()
    else:
        raise TypeError("normal_classはリストで指定してください")

    # --- 異常データのindex取得 ---
    if isinstance(unseen_anomaly, list):
        anomaly_mask = torch.isin(train_indices, torch.tensor(normal_class))
        anomaly_mask = ~anomaly_mask
        unseen_mask = torch.isin(train_indices, torch.tensor(unseen_anomaly))
        unseen_mask = ~unseen_mask
        anomaly_mask = torch.logical_and(anomaly_mask, unseen_mask)
        train_anomaly_indices = anomaly_mask.nonzero().squeeze().tolist()
    else:
        raise TypeError("unseen_anomalyはリストで指定してください")

    # --- サンプル抽出 ---
    if not labeled_anomaly_class:
        train_normal_bag = random.sample(train_normal_indices, k=n_unlabeled_normal)
        train_anomaly_bag = random.sample(train_anomaly_indices, k=n_labeled_anomaly + n_unlabeled_anomaly)
        train_positive_bag = train_anomaly_bag[:n_labeled_anomaly]
        train_unlabeled_bag = train_normal_bag + train_anomaly_bag[n_labeled_anomaly:]
    else:
        train_unlabeled_anomaly_indices = train_anomaly_indices
        mask = torch.isin(train_indices, torch.tensor(labeled_anomaly_class))
        train_labeled_anomaly_indices = mask.nonzero().squeeze().tolist()
        train_unlabeled_anomaly_bag = random.sample(train_unlabeled_anomaly_indices, k=n_unlabeled_anomaly)
        train_labeled_anomaly_bag = random.sample(train_labeled_anomaly_indices, k=n_labeled_anomaly)
        train_positive_bag = train_labeled_anomaly_bag
        train_normal_bag = random.sample(train_normal_indices, k=n_unlabeled_normal)
        train_unlabeled_bag = train_normal_bag + train_unlabeled_anomaly_bag

    # --- ラベル付与 ---
    if return_id:
        if name == "SVHN":
            new_targets = torch.ones(train.labels.shape[0], 2, dtype=int)*-1
        elif name == "CIFAR10":
            new_targets = torch.ones(len(train.targets), 2, dtype=int)*-1
        else:
            new_targets = torch.ones(train.targets.shape[0], 2, dtype=int)*-1
        for i in train_positive_bag:
            if name == "SVHN":
                new_targets[i] = torch.tensor([1, train.labels[i].item()], dtype=int)
            elif name == "CIFAR10":
                new_targets[i] = torch.tensor([1, train.targets[i]], dtype=int)
            else:
                new_targets[i] = torch.tensor([1, train.targets[i].item()], dtype=int)
        for i in train_unlabeled_bag:
            if name == "SVHN":
                new_targets[i] = torch.tensor([0, train.labels[i].item()], dtype=int)
            elif name == "CIFAR10":
                new_targets[i] = torch.tensor([0, train.targets[i]], dtype=int)
            else:
                new_targets[i] = torch.tensor([0, train.targets[i].item()], dtype=int)
        if name != "SVHN":
            train.targets = new_targets
        else:
            train.labels = new_targets
    else:
        for i in train_positive_bag:
            if name != "SVHN":
                train.targets[i] = 1
            else:
                train.labels[i] = 1
        for i in train_unlabeled_bag:
            if name != "SVHN":
                train.targets[i] = 0
            else:
                train.labels[i] = 0

    # --- サブセット返却フラグごとの分岐 ---
    if return_type=="train":
        train_unlabeled_bag, train_positive_bag, val_bag = split(train_unlabeled_bag, train_positive_bag, train_ratio)
        pu_train_subset = Subset(train, train_unlabeled_bag+train_positive_bag)
        pos_train_subset = Subset(train, train_positive_bag)
        val_subset = Subset(train, val_bag)
        unl_normalize_subset = Subset(train_normalize, train_unlabeled_bag)
        pos_normalize_subset = Subset(train_normalize, train_positive_bag)
        return pu_train_subset, pos_train_subset, unl_normalize_subset, pos_normalize_subset, val_subset

    elif return_type=="test":
        # --- テストデータのインデックス取得 ---
        test_indices = test.targets if name != "SVHN" else test.labels
        if not torch.is_tensor(test_indices):
            test_indices = torch.tensor(test_indices)
        test_normal_mask = torch.isin(test_indices, torch.tensor(normal_class))
        test_normal_indices = test_normal_mask.nonzero().squeeze().tolist()
        test_anomaly_mask = torch.isin(test_indices, torch.tensor(normal_class))
        test_anomaly_indices = (~test_anomaly_mask).nonzero().squeeze().tolist()

        # --- テストサンプル抽出 ---
        test_normal_bag = random.sample(test_normal_indices, k=min(len(test_normal_indices), 1000))
        test_anomaly_bag = random.sample(test_anomaly_indices, k=n_test - len(test_normal_bag))

        # --- テストラベル付与 ---
        if not return_id:
            for i in test_anomaly_bag:
                if name != "SVHN":
                    test.targets[i] = 1
                else:
                    test.labels[i] = 1
            for i in test_normal_bag:
                if name != "SVHN":
                    test.targets[i] = 0
                else:
                    test.labels[i] = 0

        test_subset = Subset(test, test_anomaly_bag + test_normal_bag)
        return test_subset
    else:
        raise ValueError("return_type must be 'train' or 'test'.")
# =========================
# 動作確認用スクリプト
# =========================
if __name__ == "__main__":
    for dataset_name in ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]:
        ret = load(dataset_name, batch_size=128)
        print(dataset_name)
        print("Train:", len(ret[0].dataset))
        print("Valid:", len(ret[1].dataset))
        print("Test:", len(ret[2].dataset))
