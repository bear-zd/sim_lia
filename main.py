import argparse
from data import DataLoaderBuilder
from model import ModelSplitor
from pipeline import SIM_LIA
import random
import numpy as np
import torch
import os
import sys


def init():
    args = argparse.ArgumentParser()
    # basic train
    args.add_argument("--dataset", type=str, default="cifar10")
    args.add_argument("--model", type=str, default="resnet18")
    args.add_argument("--batch_size", type=int, default=128)
    args.add_argument("--epoch", type=int, default=10)
    # paper method
    args.add_argument("--what", type=str, choices=["smashed", "grad"])
    args.add_argument("--measure", type=str, choices=["cosine", "euclidean", "k-means"])
    args.add_argument("--split_pos", type=int, default=None)
    # log method
    args.add_argument("--print_to_stdout", action="store_true")
    args.add_argument("--log", type=str, default="logs")
    args.add_argument("--seed", type=int, default=42)

    args = args.parse_args()
    if args.log is None or args.print_to_stdout:
        dir_name = os.path.join("logs", f"{args.dataset}_{args.model}_{args.what}_{args.measure}")
        args.dir_name = dir_name
        os.makedirs(dir_name, exist_ok=True)
        if not args.print_to_stdout:
            sys.stdout = open(os.path.join(dir_name, "logs.txt"), "wt")
    set_random_seed(args.seed)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    data_builder = DataLoaderBuilder(args.dataset, args.batch_size)
    train_loader, test_loader = data_builder.get_loader()
    num_classes = data_builder.get_num_classes()

    model = ModelSplitor(args.model, num_classes)
    split_poses = [args.split_pos] if args.split_pos is not None else model.get_availabel_split()
    for split_pos in split_poses:
        bottom_model, top_model = model.split_model(split_pos)
        lia_pipelien = SIM_LIA(bottom_model, top_model, train_loader, test_loader, args.epoch)
        
    
    




if __name__ == "__main__":
    args = init()
    main(args)