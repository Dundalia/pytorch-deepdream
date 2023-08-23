import torch

import os
from collections import namedtuple

import torch
from torchvision import models
from torch.hub import download_url_to_file
import open_clip

from utils.constants import *

class OpenCLIP_ConvNeXt_base_w(torch.nn.Module):
    """Only those layers are exposed which have already proven to work nicely."""

    def __init__(self, pretrained_weights = None, requires_grad=False):
        super().__init__()
        model_name = "convnext_base_w"
        if pretrained_weights:
            pretrained = pretrained_weights.lower()
        else:
            pretrained = None

        if pretrained == SupportedPretrainedWeights.IMAGENET.name: 
        if (pretrained is None) or (pretrained in open_clip.list_pretrained_tags_by_model(model_name)):
            self.model = open_clip.create_model(model_name, pretrained=pretrained, require_pretrained=True).visual.eval().to(DEVICE)
        else:
            raise Exception(f'Pretrained weights {pretrained_weights} not yet supported for {self.__class__.__name__} model.')


        self.layer_names = ['layer1', 'layer2', 'layer3', 'layer4']

        # Set these to False so that PyTorch won't be including them in it's autograd engine - eating up precious memory
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    # Feel free to experiment with different layers
    def forward(self, x):
        activations = []
        net_outputs = namedtuple(self.__class__.__name__, self.layer_names)

        visual = self.model.visual
        x = visual.trunk.stem(x)

        for index, layer in enumerate(visual.trunk.stages):
          x = layer(x)
          if ("layer"+str(index+1) in self.layer_names):
            activations.append(x)


        # Feel free to experiment with different layers.
        out = net_outputs(*activations)
        # layer35 is my favourite
        return out