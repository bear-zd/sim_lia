import argparse
from data import DataLoaderBuilder
from model import ModelSplitor
from pipeline import SIM_LIA
import random
import numpy as np
import torch
import os
import sys
from typing import Optional


def init():
    args = argparse.ArgumentParser()
    # basic train
    args.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "stl10", ])
    args.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "vgg19"])
    args.add_argument("--batch_size", type=int, default=64)
    args.add_argument("--epoch", type=int, default=30)
    # paper method
    args.add_argument("--what", type=str, choices=["smashed", "grad"])
    args.add_argument("--measure", type=str, choices=["cosine", "euclidean", "k-means"])
    args.add_argument("--split_pos", type=int, default=None)
    # log method
    args.add_argument("--print_to_stdout", action="store_true")
    args.add_argument("--log", type=str, default="logs")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--gpu", type=Optional[int], default=0)
    args.add_argument("--save", action="store_true")

    args = args.parse_args()
    if args.log is None or args.print_to_stdout:
        dir_name = os.path.join("logs", f"{args.dataset}_{args.model}_{args.what}_{args.measure}")
        args.dir_name = dir_name
        os.makedirs(dir_name, exist_ok=True)
        if not args.print_to_stdout:
            sys.stdout = open(os.path.join(dir_name, "logs.txt"), "wt")
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    set_random_seed(args.seed)
    return args


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
    split_poses = [args.split_pos] if args.split_pos is not None else model.get_available_split()
    if args.split_pos is None:
        print("Model available split positions: ", split_poses)
    for split_pos in range(len(split_poses)):
        bottom_model, top_model = model.split_model(split_pos)
        lia_pipeline = SIM_LIA(bottom_model, top_model, train_loader, test_loader, args.epoch, args.device)
        if args.save:
            save_dir = os.path.join(args.dir_name, f"split_pos{split_pos}")
            os.makedirs(save_dir, exist_ok=True)
            try:
                lia_pipeline.load_model(save_dir)
            except FileNotFoundError:
                pass
        lia_pipeline.train()
        print("=============== Train Done ===============")
        collect_data, collect_label, known_data, known_label = lia_pipeline.extract_feature(args.what)
        print("=============== Extract Done ===============")
        if args.save:
            lia_pipeline.save_model(save_dir)
            np.save(os.path.join(save_dir, "collect_data.npy"), collect_data)
            np.save(os.path.join(save_dir, "collect_label.npy"), collect_label)
            np.save(os.path.join(save_dir, "known_data.npy"), known_data)
            np.save(os.path.join(save_dir, "known_label.npy"), known_label)

        acc = lia_pipeline.attack(collect_data, collect_label, known_data, known_label, args.measure)
        print(f"split_pos {split_pos} acc: {acc}")

        
    

if __name__ == "__main__":
    args = init()
    main(args)