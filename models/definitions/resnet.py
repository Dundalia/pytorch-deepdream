import os
from collections import namedtuple

import torch
from torchvision import models
from torch.hub import download_url_to_file
import open_clip

from utils.constants import *


class ResNet(torch.nn.Module):

    def __init__(self, model_name = "RN50", pretrained_weights = SupportedPretrainedWeights.IMAGENET.name, requires_grad=False, show_progress=False):
        super().__init__()

        if pretrained_weights == SupportedPretrainedWeights.IMAGENET.name:
            if model_name == 'RN50':
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT, progress=show_progress).eval()

            if model_name == 'RN101':
                model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT, progress=show_progress).eval()

            if model_name == 'RN152':
                model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT, progress=show_progress).eval()

            self.conv1 = model.conv1
            self.conv2 = torch.nn.Identity()
            self.conv3 = torch.nn.Identity()
            self.bn1 = model.bn1
            self.bn2 = torch.nn.Identity()
            self.bn3 = torch.nn.Identity()
            self.relu1 = model.relu
            self.relu2 = torch.nn.Identity()
            self.relu3 = torch.nn.Identity()
            self.pool = model.maxpool
            
            self.layer1 = model.layer1
            self.layer2 = model.layer2
            self.layer3 = model.layer3
            self.layer4 = model.layer4

        elif (pretrained_weights == SupportedPretrainedWeights.PLACES_365.name) and (model_name == "RN50"):
            model = models.resnet50(pretrained=False, progress=show_progress).eval()

            binary_name = 'resnet50_places365.pth.tar'
            resnet50_places365_binary_path = os.path.join(BINARIES_PATH, binary_name)

            if os.path.exists(resnet50_places365_binary_path):
                state_dict = torch.load(resnet50_places365_binary_path)['state_dict']
            else:
                binary_url = r'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'
                print(f'Downloading {binary_name} from {binary_url} it may take some time.')
                download_url_to_file(binary_url, resnet50_places365_binary_path)
                print('Done downloading.')
                state_dict = torch.load(resnet50_places365_binary_path)['state_dict']

            new_state_dict = {}  # modify key names and make it compatible with current PyTorch model naming scheme
            for old_key in state_dict.keys():
                new_key = old_key[7:]
                new_state_dict[new_key] = state_dict[old_key]

            model.fc = torch.nn.Linear(model.fc.in_features, 365)
            model.load_state_dict(new_state_dict, strict=True)

            self.conv1 = model.conv1
            self.conv2 = torch.nn.Identity()
            self.conv3 = torch.nn.Identity()
            self.bn1 = model.bn1
            self.bn2 = torch.nn.Identity()
            self.bn3 = torch.nn.Identity()
            self.relu1 = model.relu
            self.relu2 = torch.nn.Identity()
            self.relu3 = torch.nn.Identity()
            self.pool = model.maxpool
            
            self.layer1 = model.layer1
            self.layer2 = model.layer2
            self.layer3 = model.layer3
            self.layer4 = model.layer4
        

        elif pretrained_weights.startswith("CLIP"):
            pretrained_weights = pretrained_weights[5:].lower()

            model = open_clip.create_model(
                model_name,
                pretrained=pretrained_weights,
                require_pretrained=True
            ).visual.eval()
            
            self.conv1 = model.conv1
            self.conv2 = model.conv2
            self.conv3 = model.conv3
            self.bn1 = model.bn1
            self.bn2 = model.bn2
            self.bn3 = model.bn3
            self.relu1 = model.act1
            self.relu2 = model.act2
            self.relu3 = model.act3
            self.pool = model.avgpool
            
            self.layer1 = model.layer1
            self.layer2 = model.layer2
            self.layer3 = model.layer3
            self.layer4 = model.layer4

        else:
            raise Exception(f'Pretrained weights {pretrained_weights} not yet supported for {self.__class__.__name__} {model_name} model.')

        self.layer_names = ['layer1', 'layer2', 'layer3', 'layer4']

        # Set these to False so that PyTorch won't be including them in it's autograd engine - eating up precious memory
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    # Feel free to experiment with different layers
    def forward(self, x):
        activations = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool(x)

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        resnet_outputs = namedtuple("ResNetOutputs", self.layer_names)
        out = resnet_outputs(layer1, layer2, layer3, layer4)
        return out