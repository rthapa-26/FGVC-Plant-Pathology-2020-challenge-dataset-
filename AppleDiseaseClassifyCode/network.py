import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib


def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    return getattr(m, class_name)


class ResNet(nn.Module):
    def __init__(self, out_dim, encoder='resnet50', pretrained=True):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        assert encoder in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], "Incorrect encoder type"

        resnet = class_for_name("torchvision.models", encoder)(pretrained=pretrained)

        # last_layer = list(resnet.children())[-1]
        # print(last_layer)
        self.resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        self.last_layer = nn.Linear(in_features=2048, out_features=out_dim, bias=True)

    def forward(self, x):
        x = self.resnet(x)
        x = x.squeeze(3).squeeze(2)
        x = self.last_layer(x)

        return x


if __name__ == '__main__':
    # import torchvision.models
    # model = torchvision.models.resnet152(pretrained=True)
    # # print(model)
    #
    # input = torch.randn(5, 3, 384, 512)
    # output = model(input)

    #
    # print('\n')
    # newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
    # print(newmodel)

    resnet = ResNet(out_dim=4)
    input = torch.randn(5, 3, 384, 512)
    output = resnet(input)
    print('input.shape: {}'.format(input.shape))
    print('output.shape: {}'.format(output.shape))
