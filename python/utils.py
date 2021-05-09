import glob
import multiprocessing
import os
import random
import shutil
import sys
from collections import OrderedDict
from functools import reduce
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

MININT: int = -sys.maxsize - 1
MAXINT: int = sys.maxsize
MINFLOAT: float = np.finfo(float).min
MAXFLOAT: float = np.finfo(float).max


def recur_getattr(cls, *attrs):
    cls_attr = cls
    for attr in attrs:
        cls_attr = getattr(cls_attr, attr)
    return cls_attr


def get_num_worker() -> int:
    return multiprocessing.cpu_count()


def is_all_type(instance: type, *variables) -> bool:
    return all(list(map(lambda x: isinstance(x, instance), variables)))


def reduce_sum(x: Iterable) -> Any:
    return reduce(lambda x, y: x + y, x)


def reduce_prod(x: Iterable) -> Any:
    return reduce(lambda x, y: x * y, x)


def map_prod(x: Iterable, y: Iterable) -> list:
    assert len(x) == len(y)
    return list(map(lambda x, y: x * y, x, y))


def is_dir(directory: str) -> bool:
    return os.path.isdir(os.path.join(os.getcwd(), directory))


def not_dir_mkdir(directory: str) -> None:
    assert isinstance(directory, str)
    if not is_dir(directory):
        os.mkdir(os.path.join(os.getcwd(), directory))


def cvt_ws2bit(w: str) -> int:
    if w == "f":
        bit = 16
    elif w == "t":
        bit = 2
    elif w == "b":
        bit = 1
    else:
        raise ValueError("Not in supported.")
    return bit


def get_bit(ws: list) -> int:
    w_num_lenet5 = [
        5 * 5 * 1 * 6,
        5 * 5 * 6 * 16,
        (4 * 4 * 16) * 120,
        120 * 84,
        84 * 10,
    ]
    w_bit = list(map(cvt_ws2bit, ws))
    layer_bit = map_prod(w_bit, w_num_lenet5)
    bit = reduce_sum(layer_bit)
    return bit


def asb_score(acc: float, spar: float, bits: int, max_bits: int) -> float:
    assert max_bits >= bits
    normalized_bits = 1 - (bits / max_bits)
    score = (acc + spar + normalized_bits) / 3.0
    assert score <= 1.0
    return score


class DictList(object):
    def __init__(self, *args, verbose: int = 1):
        self.data = OrderedDict({i: [] for i in args})
        self.verbose = verbose
        self.iter = 0

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        if self.iter > len(self.keys):
            self.iter += 1
            return self.keys[self.iter]
        else:
            raise StopIteration

    @property
    def keys(self):
        return self.data.keys()

    @keys.setter
    def keys(self):
        return self.data.keys()

    def append(self, key, var: Any) -> None:
        self.data[key].append(var)

    def append_kwargs(self, **kwargs) -> None:
        _ = [self.data[key].append(kwargs[key]) for key in kwargs.keys()]

    def to_df(self, fill_var: float = np.nan) -> pd.DataFrame:
        """"""
        data = self.fill_dictlist_len_equal(self.data, fill_var=fill_var)
        df = pd.DataFrame(data)
        return df

    def to_csv(self, file_name: str, fill_var: float = np.nan) -> None:
        """Saving to csv."""
        df = self.to_df(fill_var=fill_var)
        df.to_csv(file_name, index=None)
        if self.verbose:
            logger.info(f"Save csv@{file_name}.")

    def amax(self, key):
        return np.amax(self.data[key])

    def amin(self, key):
        return np.amin(self.data[key])

    @staticmethod
    def fill_dictlist_len_equal(dictlist, fill_var: int = 0) -> dict:
        maxlen = MININT
        for k in dictlist.keys():
            if len(dictlist[k]) > maxlen:
                maxlen = len(dictlist[k])

        for k in dictlist.keys():
            if len(dictlist[k]) < maxlen:
                diff = maxlen - len(dictlist[k])
                _ = [dictlist[k].append(fill_var) for i in range(diff)]
        return dictlist


class TensorBoard(object):
    def __init__(
        self, log_dir: str = "./logs", sub_dir: str = None, rm_exists: bool = False
    ) -> None:
        # Not need for asserting. Those are in set_tensorboard.
        self.writer = self.set_tensorboard(log_dir, sub_dir, rm_exists)

    def add_scalar_from_dict(self, counter: int = None, **kwargs) -> None:
        for key in kwargs.keys():
            if counter is not None:
                self.writer.add_scalar(str(key), kwargs[key], counter)
            else:
                self.writer.add_scalar(str(key), kwargs[key])

    def add_scalar_from_kwargs(self, counter: int = None, **kwargs) -> None:
        for key in kwargs.keys():
            if counter is not None:
                self.writer.add_scalar(str(key), kwargs[key], counter)
            else:
                self.writer.add_scalar(str(key), kwargs[key])

    @staticmethod
    def set_tensorboard(
        log_dir: str = "./logs", sub_dir: str = None, rm_exists: bool = False
    ):
        assert isinstance(log_dir, str)
        assert isinstance(sub_dir, str) or sub_dir is None
        assert isinstance(rm_exists, bool)
        if sub_dir is not None:
            log_dir = os.path.join(log_dir, sub_dir)
        if rm_exists:
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
                old_boards = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
                _ = [os.remove(i) for i in old_boards]
        writer = SummaryWriter(log_dir=log_dir)
        return writer


def measure_sparse(*ws) -> float:
    if not ws:
        sparse = torch.tensor(0.0)
    else:
        total_sparity = 0
        num_params = 0
        for w in ws:
            if w is None:
                continue
            w = w.data
            device = w.device
            num_params += w.numel()
            total_sparity += torch.where(
                w == 0.0, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device)
            ).sum()
        if num_params == 0:
            sparse = torch.tensor(0.0)
        else:
            sparse = total_sparity / num_params
    return sparse.item()


def ternary_threshold(delta: float = 0.7, *ws):
    assert isinstance(delta, float)
    num_params = sum_w = 0
    if not ws:
        # In case, of all params cannot be found.
        threshold = torch.tensor(np.nan)
    else:
        for w in ws:
            w = w.data
            num_params += w.numel()
            sum_w += w.abs().sum()
        threshold = delta * (sum_w / num_params)
    return threshold


def topk_accuracy(logits, target, k: int = 1):
    assert isinstance(k, int)
    logits = logits.data
    target = target.data
    _, pred = logits.topk(k, dim=-1)
    target_expand = target.expand_as(pred.T)
    correct = target_expand.T.eq(pred)
    return correct.sum().float()


def search_params(model, is_weight: bool, *kws):
    assert hasattr(model, "named_parameters")
    assert isinstance(is_weight, bool)
    ncond = len(kws)
    params = {}
    for name, param in model.named_parameters():
        is_w = False
        is_b = False
        ncorr = 0

        name = name.replace("seq.", "")
        if name.find(".weight") >= 0:
            name = name.replace(".weight", "")
            is_w = True
        elif name.find(".bias") >= 0:
            name = name.replace(".bias", "")
            is_b = True
        for kw in kws:
            if name.find(kw) >= 0:
                ncorr += 1
        if ncond == ncorr and is_w and is_weight:
            params.update({name: param})
        elif ncond == ncorr and is_b and not is_weight:
            params.update({name: param})
    return list(params.keys()), list(params.values())


def train(
    model,
    device,
    loader,
    optim,
    scheduler=None,
    board=None,
    dictlist=None,
    epoch_count: int = 0,
):
    model.train()
    run_batch = run_corr = run_loss = 0
    thre_const = 0.7
    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    for data, tar in loader:
        data, tar = data.to(device), tar.to(device)
        optim.zero_grad()
        _, wt_all = search_params(model, True, "w", "t")
        thres = [ternary_threshold(thre_const, wt) for wt in wt_all]

        output = model.forward(data, thres)
        loss = loss_fn(output, tar)
        loss.backward()
        optim.step()
        with torch.no_grad():
            run_batch += tar.shape[0]
            run_loss += loss.item()
            run_corr += topk_accuracy(output, tar, 1).item()
            all_spar = model.sparse_all()
    acc = run_corr / run_batch
    loss = run_loss / run_batch
    spar = all_spar
    thre = {f"threshold_{idx}": t.item() for idx, t in enumerate(thres)}
    scalars = {"train_acc": acc, "train_loss": loss, "spar": spar}
    scalars.update(thre)

    if board is not None:
        board.add_scalar_from_dict(epoch_count, **scalars)
    if scheduler is not None:
        scheduler.step()
    logger.info(
        f'Acc: {acc:.6f}, Loss: {loss:.6f}, Sparsity: {spar:.6f}, Lr: {optim.param_groups[0]["lr"]:.6f}'
    )
    if dictlist is not None:
        dictlist.append_kwargs(train_acc=acc, train_loss=loss, spar=spar, thre=thre)


def test(
    model, device, loader, optim, board=None, dictlist=None, epoch_count: int = 0
):  # , thre_const: float) -> None:
    model.eval()
    run_batch = run_corr = run_loss = 0
    thre_const = 0.7
    for data, tar in loader:
        with torch.no_grad():
            data, tar = data.to(device), tar.to(device)
            _, wt_all = search_params(model, True, "w", "t")
            thres = [ternary_threshold(thre_const, wt) for wt in wt_all]
            output = model.forward(data, thres)
            loss = F.cross_entropy(output, tar, reduction="sum")
            run_batch += tar.shape[0]
            run_loss += loss.item()
            run_corr += topk_accuracy(output, tar, 1).item()
    acc = run_corr / run_batch
    loss = run_loss / run_batch
    logger.info(f"Acc: {acc:.6f}, Loss: {loss:.6f}.")

    thre = {f"threshold_{idx}": t.item() for idx, t in enumerate(thres)}
    scalars = {"test_acc": acc, "test_loss": loss}
    scalars.update(thre)

    if board is not None:
        board.add_scalar_from_dict(epoch_count, **scalars)
    if dictlist is not None:
        dictlist.append_kwargs(test_acc=acc, test_loss=loss)


def load_dataset(
    num_train_batch: int,
    num_test_batch: int,
    num_extra_batch: int = 0,
    num_worker: int = 8,
    dataset: str = "mnist",
    roof: str = "./dataset",
    transforms_list: list = None,
) -> tuple:
    assert isinstance(num_train_batch, int)
    assert isinstance(num_test_batch, int)
    assert isinstance(num_extra_batch, int)
    assert isinstance(num_worker, int)
    assert isinstance(dataset, str)
    assert isinstance(roof, str)
    dataset = dataset.lower()
    not_dir_mkdir(roof)
    if transforms_list is None:
        transforms_list = [transforms.ToTensor()]
    transforms_list = transforms.Compose(transforms_list)

    if dataset == "mnist":
        train_set = torchvision.datasets.MNIST(
            root=roof, train=True, download=True, transform=transforms_list
        )
        train_loader = DataLoader(
            train_set, batch_size=num_train_batch, shuffle=True, num_workers=num_worker
        )
        test_set = torchvision.datasets.MNIST(
            root=roof, train=False, download=True, transform=transforms_list
        )
        test_loader = DataLoader(
            test_set, batch_size=num_test_batch, shuffle=False, num_workers=num_worker
        )
    elif dataset == "fmnist":
        train_set = torchvision.datasets.FashionMNIST(
            root=roof, train=True, download=True, transform=transforms_list
        )
        train_loader = DataLoader(
            train_set, batch_size=num_train_batch, shuffle=True, num_workers=num_worker
        )
        test_set = torchvision.datasets.FashionMNIST(
            root=roof, train=False, download=True, transform=transforms_list
        )
        test_loader = DataLoader(
            test_set, batch_size=num_test_batch, shuffle=False, num_workers=num_worker
        )
    elif dataset == "kmnist":
        train_set = torchvision.datasets.KMNIST(
            root=roof, train=True, download=True, transform=transforms_list
        )
        train_loader = DataLoader(
            train_set, batch_size=num_train_batch, shuffle=True, num_workers=num_worker
        )
        test_set = torchvision.datasets.FashionMNIST(
            root=roof, train=False, download=True, transform=transforms_list
        )
        test_loader = DataLoader(
            test_set, batch_size=num_test_batch, shuffle=False, num_workers=num_worker
        )
    elif dataset == "emnist":
        train_set = torchvision.datasets.EMNIST(
            root=roof, train=True, download=True, transform=transforms_list
        )
        train_loader = DataLoader(
            train_set, batch_size=num_train_batch, shuffle=True, num_workers=num_worker
        )
        test_set = torchvision.datasets.FashionMNIST(
            root=roof, train=False, download=True, transform=transforms_list
        )
        test_loader = DataLoader(
            test_set, batch_size=num_test_batch, shuffle=False, num_workers=num_worker
        )
    elif dataset == "cifar10":
        train_set = torchvision.datasets.CIFAR10(
            root=roof, train=True, download=True, transform=transforms_list
        )
        train_loader = DataLoader(
            train_set, batch_size=num_train_batch, shuffle=True, num_workers=num_worker
        )
        test_set = torchvision.datasets.CIFAR10(
            root=roof, train=False, download=True, transform=transforms_list
        )
        test_loader = DataLoader(
            test_set, batch_size=num_test_batch, shuffle=False, num_workers=num_worker
        )
    elif dataset == "cifar100":
        train_set = torchvision.datasets.CIFAR100(
            root=roof, train=True, download=True, transform=transforms_list
        )
        train_loader = DataLoader(
            train_set, batch_size=num_train_batch, shuffle=True, num_workers=num_worker
        )
        test_set = torchvision.datasets.CIFAR100(
            root=roof, train=False, download=True, transform=transforms_list
        )
        test_loader = DataLoader(
            test_set, batch_size=num_test_batch, shuffle=False, num_workers=num_worker
        )
    elif dataset == "svhn":
        # The extra-section or extra_set is exist in this dataset.
        train_set = torchvision.datasets.SVHN(
            root=roof, split="train", download=True, transform=transforms_list
        )
        train_loader = DataLoader(
            train_set, batch_size=num_train_batch, shuffle=True, num_workers=num_worker
        )
        test_set = torchvision.datasets.SVHN(
            root=roof, split="test", download=True, transform=transforms_list
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=num_test_batch, shuffle=False, num_workers=num_worker
        )
        extra_set = torchvision.datasets.SVHN(
            root=roof, split="extra", download=True, transform=transforms_list
        )
        extra_loader = DataLoader(
            extra_set, batch_size=num_extra_batch, shuffle=False, num_workers=num_worker
        )
        return train_loader, test_loader, extra_loader
    else:
        raise NotImplementedError(
            "dataset must be in [mnist, fmnist, kmnist, "
            f"emnist, cifar10, cifar100, svhn] only, your input: {dataset}"
        )
    return train_loader, test_loader


def seed(seed: int = 2020) -> None:
    assert isinstance(seed, int)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    log_exists = "logger" in locals() or "logger" in globals()
    if log_exists:
        logger.info(f"Plant the random seed: {seed}.")
    else:
        print(f"Plant the random seed: {seed}.")


def init_logger(name_log: str = __file__, rm_exist: bool = False):
    if rm_exist and os.path.isfile(name_log):
        os.remove(name_log)

    logger.add(
        name_log,
        format="{time} | {level} | {message}",
        backtrace=True,
        diagnose=True,
        level="INFO",
        colorize=False,
    )
