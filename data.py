import torch
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST

DATASETS = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "mnist": MNIST,
    "fashionmnist": FashionMNIST
}

class DataGetter():
    def __init__(self, dataset_name, batch_size, num_workers):
        self.dataset_name = dataset_name
        self.batch_size = batch_size

