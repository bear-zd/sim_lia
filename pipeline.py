import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

class SIM_LIA():
    def __ini__(self, bottom_model, top_model, train_loader, test_loader, num_epoch, device):
        self.bottom_model = bottom_model
        self.top_model = top_model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epoch = num_epoch
        self.optimizer = torch.optim.Adam(list(self.top_model.parameters())+list(self.bottom_model.parameters()), lr=0.01, momentum=0.9)
        epoch_steps = len(self.train_loader)
        self.lr_schedular = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = device
        
    def _train(self, data, label, train=True):
        len_d, correct, loss = 0, 0, 0
        self.optimizer.zero_grad()
        data = data.to(self.device)
        label = label.to(self.device)
        mid = self.bottom_model(data)
        pred = self.top_model(mid)
        loss = self.loss_fn(pred, label)
        if train:
            loss.backward()
            self.optimizer.step()
            self.lr_schedular.step()
    
        len_d += label.size(0)
        total_loss += loss.item() * len_d
        _, predicted = torch.max(pred, 1)
        correct += (predicted == label).sum().item()

        return len_d, correct, loss
    
    def train(self):
        for i in range(self.num_epoch):
            total = 0; correct = 0; total_loss = 0
            for j, (x, y) in enumerate(self.train_loader):
                len_d, cor, loss =self._train(x, y, train=True)
                total += len_d; correct += cor; total_loss += loss
            print('Epoch: %d, Loss: %f, Accuracy: %f' % (i, total_loss/total, correct/total))
            
            total = 0; correct = 0; total_loss = 0
            for j, (x, y) in enumerate(self.test_loader):
                len_d, cor, loss =self._train(x, y, train=False)
                total += len_d; correct += cor; total_loss += loss
            print('Epoch: %d, Loss: %f, Accuracy: %f' % (i, total_loss/total, correct/total))

    def extract_feature(self, what, part="train"):
        if what == "smashed":
            train_features, train_labels = None, None
            test_features, test_labels = None, None
            for i, (x, y) in enumerate(self.train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                mid = self.bottom_model(x)
                train_features = np.concatenate((train_features, mid.cpu().detach().numpy()), axis=0) if train_features is not None else mid.cpu().detach().numpy()
                train_labels = np.concatenate((train_labels, y.cpu().detach().numpy()), axis=0) if train_labels is not None else y.cpu().detach().numpy()
            for i, (x, y) in enumerate(self.test_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                mid = self.bottom_model(x)
                test_features = np.concatenate((test_features, mid.cpu().detach().numpy()), axis=0) if test_features is not None else mid.cpu().detach().numpy()
                test_labels = np.concatenate((test_labels, y.cpu().detach().numpy()), axis=0) if test_labels is not None else y.cpu().detach().numpy()
            if part=="train":
                return train_features, train_labels, test_features, test_labels
            else:
                return test_features, test_labels, train_features, train_labels
        
        if what == "grad":
            train_grad, train_labels = None, None
            test_grad, test_labels = None, None
            for i, (x, y) in enumerate(self.test_dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                # x.requires_grad = True
                mid = self.bottom_model(x)
                mid_detached = mid.detach()
                mid_detached.requires_grad = True
                pred = self.top_model(mid_detached)
                loss = self.loss_fn(pred, y)
                loss.backward()
                grad = mid_detached.grad.cpu().detach().numpy()
                train_grad = np.concatenate((train_grad, grad), axis=0) if train_grad is not None else grad
                train_labels = np.concatenate(train_labels, y.cpu().detach().numpy(), axis=0) if train_labels is not None else y.cpu().detach().numpy()

            for i, (x, y) in enumerate(self.train_dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                # x.requires_grad = True
                mid = self.bottom_model(x)
                mid_detached = mid.detach()
                mid_detached.requires_grad = True
                pred = self.top_model(mid_detached)
                loss = self.loss_fn(pred, y)
                loss.backward()
                grad = mid_detached.grad.cpu().detach().numpy()
                test_grad = np.concatenate((test_grad, grad), axis=0) if test_grad is not None else grad
                test_labels = np.concatenate(test_labels, y.cpu().detach().numpy(), axis=0) if test_labels is not None else y.cpu().detach().numpy()

            if part == "train":
                return train_grad, train_labels, test_grad, test_labels
            else:
                return test_grad, test_labels, train_grad, train_labels


        
        