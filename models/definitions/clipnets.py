import torch
import cv2 as cv
import clip
from collections import namedtuple

from utils.constants import *


class CLIP(torch.nn.Module):
    def __init__(self, requires_grad=False, show_progress=False, checkpoint="ViT-B/16"):

        super().__init__()

        model, process_image = clip.load(checkpoint, device=DEVICE)
        self.model = model
        self.input_resolution = model.visual.input_resolution
        #self.process_image = process_image
        self.layer_names = ["logits_per_image"]
        
        
        if not requires_grad:
          for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        img, text = x
        # img = self.process_image(img).to(DEVICE)
        img = img.to(DEVICE)
        text = clip.tokenize(text).to(DEVICE)

        logits_per_image, logits_per_text = self.model(img, text)
        out = logits_per_image
        
        # Feel free to experiment with different layers.
        clip_outputs = namedtuple("CLIPOutputs", self.layer_names)
        out = clip_outputs(out)
        return out


    def process_image(self, img_tensor):
        # Ensure img_tensor is a torch tensor
        if not torch.is_tensor(img_tensor):
            raise ValueError("Input image must be a torch tensor")

        # Drop the batch dimension for easier processing
        img_tensor = img_tensor.squeeze(0)

        # Determine the aspect ratio
        _, h, w = img_tensor.shape
        aspect_ratio = w / h

        if aspect_ratio > 1:
            # If width > height, then resize height to self.input_resolution
            new_height = self.input_resolution
            new_width = int(new_height * aspect_ratio)
        else:
            # If width <= height, then resize width to self.input_resolution
            new_width = self.input_resolution
            new_height = int(new_width / aspect_ratio)

        # Resize using torch.nn.functional
        img_tensor = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=(new_height, new_width), mode='bicubic', align_corners=True)
        img_tensor = img_tensor.squeeze(0)  # Remove batch dimension

        # Center crop
        target_size = self.input_resolution
        startx = w // 2 - (target_size // 2)
        starty = h // 2 - (target_size // 2)
        img_tensor = img_tensor[:, starty:starty+target_size, startx:startx+target_size]

        # Reshape to [1, 3, w, h]
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor
