import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import measure_sparse


def modify_lenet5(chl: int = 3, *ws):
    assert len(ws) == 5
    assert isinstance(chl, int)
    type_ws = f"{ws[0]}nrm{ws[1]}nrmv{ws[2]}nrd{ws[3]}nrd{ws[4]}s"
    type_ls = "cccccccccllllllllll"

    if chl == 1:
        feat_size = 4
    elif chl == 3:
        feat_size = 5
    else:
        raise ValueError(f"`chl` should be 1 or 3, your {chl}.")

    list_args = [
        [chl, 6, 5, 1, 0, 1, 1, False],
        [6],
        [],
        [2],
        [6, 16, 5, 1, 0, 1, 1, False],
        [16],
        [],
        [2],
        [],
        [feat_size * feat_size * 16, 120, False],
        [120],
        [],
        [0.5],
        [120, 84, False],
        [84],
        [],
        [0.5],
        [84, 10, True],
        [],
    ]
    return type_ws, type_ls, list_args


class BinQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        return w.sign()

    @staticmethod
    def backward(ctx, grad_o):
        return grad_o, None


class TerQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, threshold):
        device = w.device
        w_ter = torch.where(
            w > threshold, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device)
        )
        w_ter = torch.where(w < -threshold, torch.tensor(-1.0).to(device), w_ter)
        return w_ter

    @staticmethod
    def backward(ctx, grad_o):
        return grad_o, None


class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(Linear, self).__init__(*args, **kwargs)

    def forward(self, x, *args):
        self.weight_q = None
        y = F.linear(x, self.weight, self.bias)
        return y


class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, *args):
        self.weight_q = self.weight
        y = F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
        return y


class BinLinear(Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, *args):
        self.weight_q = BinQuant.apply(self.weight)
        y = F.linear(x, self.weight_q, self.bias)
        return y


class BinConv2d(Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, *args):
        self.weight_q = BinQuant.apply(self.weight)
        y = F.conv2d(x, self.weight_q, self.bias, self.stride, self.padding)
        return y


class TerLinear(Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        self.weight_q = TerQuant.apply(self.weight, threshold)
        x = F.linear(x, self.weight_q, self.bias)
        return x


class TerConv2d(Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        self.weight_q = TerQuant.apply(self.weight, threshold)
        x = F.conv2d(x, self.weight_q, self.bias, self.stride, self.padding)
        return x


def wl_sel(type_w: str, type_l: str, *args, **kwargs):
    assert isinstance(type_w, str)
    assert isinstance(type_l, str)

    type_w = type_w.lower()
    type_l = type_l.lower()

    assert type_w in "ftbnmrsvd"
    assert type_l in "cl"

    if type_w == "f":
        if type_l == "c":
            l = Conv2d(*args, **kwargs)
        elif type_l == "l":
            l = Linear(*args, **kwargs)
    elif type_w == "t":
        if type_l == "c":
            l = TerConv2d(*args, **kwargs)
        elif type_l == "l":
            l = TerLinear(*args, **kwargs)
    elif type_w == "b":
        if type_l == "c":
            l = BinConv2d(*args, **kwargs)
        elif type_l == "l":
            l = BinLinear(*args, **kwargs)
    elif type_w == "n":
        if type_l == "c":
            l = nn.BatchNorm2d(*args, **kwargs)
        elif type_l == "l":
            l = nn.BatchNorm1d(*args, **kwargs)
    elif type_w == "m":
        if type_l == "c":
            l = nn.MaxPool2d(*args, **kwargs)
        elif type_l == "l":
            l = nn.MaxPool1d(*args, **kwargs)
    elif type_w == "d":
        if type_l == "c":
            l = nn.Dropout2d(*args, **kwargs)
        elif type_l == "l":
            l = nn.Dropout(*args, **kwargs)
    elif type_w == "r":
        l = nn.ReLU()
    elif type_w == "s":
        l = nn.Identity()
    elif type_w == "v":
        l = nn.Flatten()
    else:
        raise NotImplementedError(f"Not in `ftbnmrs`. Your {type_w}")
    return l


class MixedPreNet(nn.Module):
    def __init__(self, type_ws: list, type_ls: list, list_args: list):
        super().__init__()
        assert len(type_ws) == len(type_ls)
        self.type_ws = type_ws
        self.type_ls = type_ls
        self.list_args = list_args

        self.thre = None
        self.seq = nn.Sequential()
        for idx, (w, l, a) in enumerate(zip(type_ws, type_ls, list_args)):
            w, l = w.lower(), l.lower()
            namestr = f"{w}{l}{idx + 1}"
            if "f" in namestr or "t" in namestr or "b" in namestr:
                namestr = "w" + namestr
                self.seq.add_module(namestr, wl_sel(w, l, *a))
            else:
                self.seq.add_module(namestr, wl_sel(w, l, *a))

    def forward(self, x, threshold=None):
        num_t = sum([1 for i in self.type_ws if i == "t"])
        assert threshold is None or num_t == len(threshold)

        counter = 0
        for w, l in zip(self.type_ws, self.seq):
            if w == "t":
                if threshold is not None:
                    x = l(x, threshold[counter])
                    counter += 1
                else:
                    x = l(x, None)
            else:
                x = l(x)
        return x

    def sparse_all(self) -> float:
        ws = []
        for w, l in zip(self.type_ws, self.seq):
            if hasattr(l, "weight_q"):
                ws.append(l.weight_q)
        return measure_sparse(*ws)

    def sparse_layerwise(self) -> list:
        spar_ws = []
        for w, l in zip(self.type_ws, self.seq):
            if hasattr(l, "weight_q"):
                spar_ws.append(measure_sparse(l.weight_q))
        return spar_ws

    def get_numel(self) -> int:
        numel = 0
        for s in self.seq:
            if hasattr(s, "weight_q") and hasattr(s, "weight"):
                numel += s.weight.numel()
        return numel

    def get_numel_layerwise(self) -> list:
        list_numel = []
        for s in self.seq:
            if hasattr(s, "weight_q") and hasattr(s, "weight"):
                numel = s.weight.numel()
                list_numel.append(numel)
        return list_numel

    def save_weight_q(self, save_loc: str) -> None:
        assert isinstance(save_loc, str)
        os.path.join(save_loc, "real.pt")
        torch.save(self.state_dict(), "")
