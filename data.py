from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
import torch.utils
from torch.utils.data import DataLoader
from torchvision import transforms

DATASETS = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "mnist": MNIST,
    "fashionmnist": FashionMNIST
}

training_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class DataLoaderBuilder():
    def __init__(self, dataset_name, batch_size):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.train_dataset = DATASETS[self.dataset_name]("data", train=True, download=True, transform=training_transform)
        self.test_dataset = DATASETS[self.dataset_name]("data", train=False, download=True, transform=training_transform)
        self.num_classes = len(self.train_dataset.classes)

    def get_loader(self):

        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_dataloader, test_dataloader
    
    def get_num_classes(self):
        return self.num_classes
