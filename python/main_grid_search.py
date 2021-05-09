import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from loguru import logger
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from mpwn import MixedPreNet, modify_lenet5
from utils import *

parser = argparse.ArgumentParser(
    description="Mixed Precision Weight Network for Fashion-MNIST."
)
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--step_down", type=float, default=0.1)
parser.add_argument("--train_batch", type=int, default=128)
parser.add_argument("--test_batch", type=int, default=128)
parser.add_argument("--weight_decay", type=float, default=0e-0)
parser.add_argument("--save_loc", type=str, default="save")
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--seed", "-s", type=int, default=2020)
parser.add_argument("--layer1", "-l1", type=str, default="f")
parser.add_argument("--layer2", "-l2", type=str, default="b")
parser.add_argument("--layer3", "-l3", type=str, default="t")
parser.add_argument("--layer4", "-l4", type=str, default="b")
parser.add_argument("--layer5", "-l5", type=str, default="f")
parser.add_argument("--step_size", type=int, default=75)
args = parser.parse_args()

if __name__ == "__main__":
    seed(args.seed)
    save_loc = ""

    device = torch.device("cuda" if args.cuda else "cpu")
    dictlist = DictList(
        "train_acc", "train_loss", "test_acc", "test_loss", "spar", "thre"
    )
    # ws = ['f', 'f', 'f', 'f', 'f']
    # ws = ('b', 'b', 'b', 'b', 'b')
    # ws = ('f', 'b', 't', 'b', 'f')
    ws = (args.layer1, args.layer2, args.layer3, args.layer4, args.layer5)

    name_ws = reduce_sum(ws)
    name_ws = os.path.join(save_loc, name_ws)
    init_logger(os.path.join(name_ws, "log.txt"), True)

    board = TensorBoard(name_ws, rm_exists=False)
    type_ws, type_ls, list_args = modify_lenet5(1, *ws)
    model = MixedPreNet(type_ws, type_ls, list_args).to(device)
    logger.info(model)

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.step_down)
    num_worker = get_num_worker()
    train_loader, test_loader = load_dataset(
        args.train_batch, args.test_batch, 0, num_worker, "fmnist"
    )

    for i in tqdm(range(1, args.epoch + 1)):
        train(model, device, train_loader, optimizer, scheduler, board, dictlist, i)
        test(model, device, test_loader, optimizer, board, dictlist, i)

    logger.info(model.sparse_layerwise())
    max_test_acc = dictlist.amax("test_acc")
    max_epoch = np.argmax(dictlist["test_acc"])
    max_sparse = dictlist["spar"][max_epoch]
    max_thre = dictlist["thre"][max_epoch]
    logger.info(
        f"Max test_acc@epoch{max_epoch}: {max_test_acc} "
        f"sparse: {max_sparse}, threshold: {max_thre}"
    )
    os.makedirs(name_ws, exist_ok=True)
    dictlist.to_csv(os.path.join(name_ws, "metrics.csv"))

    bits = get_bit(ws)
    max_ws = ["f", "f", "f", "f", "f"]
    max_bits = get_bit(max_ws)
    score = asb_score(max_test_acc, max_sparse, bits, max_bits)
    logger.info(f"bits: {bits}")
    logger.info(f"ASB score without normalization: {score}")
