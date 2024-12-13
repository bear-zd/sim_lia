import torch
import torch.nn as nn
import torchvision
from torchvision import models
MODELS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "vgg19": models.vgg19,
}

SPLIT_SIGN = {
    "resnet": torch.nn.Sequential,
    "vgg": torch.nn.MaxPool2d
}

class ModelSplitor():
    def __init__(self, model_name, num_classes):
        self.model = MODELS[model_name](pretrained=True)
        self.model_name = model_name
        self.model_type = "resnet" if "resnet" in model_name else "vgg"
        self._modify_fc(num_classes)
        if self.model_type == "vgg":
            self._preproccess_vgg()

    def _preproccess_vgg(self):
        model_list = []
        for layers in list(self.model.children()):
            if type(layers) == torch.nn.Sequential:
                for layer in layers:
                    model_list.append(layer)
            else:
                model_list.append(layers)
        self.model = nn.Sequential(*model_list)
        
    def _modify_fc(self, num_classes):
        if "resnet" in self.model_name:
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            self.model.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=1),
                        nn.Flatten())
        else:
            self.model.avgpool = nn.Sequential(self.model.avgpool, nn.Flatten())
            self.model.classifier.append(nn.Linear(1000, num_classes)) 

    def get_availabel_split(self):
        split_sign = SPLIT_SIGN[self.model_type]
        statistic = []
        for idx, layer in enumerate(self.model.children()):
            if type(layer) == split_sign:
                statistic.append(idx)
        statistic.append(statistic[-1]+1)
        return statistic
    
    def split_model(self, split_pos):
        statistic = self.get_availabel_split()
        if split_pos > len(statistic):
            raise ValueError(f"split_pos should be less than {len(statistic)}")
        split_pos = statistic[split_pos]
        bottom_model = nn.Sequential(*list(self.model.children())[:split_pos])
        top_model = nn.Sequential(*list(self.model.children())[split_pos:])
        return bottom_model, top_model




        