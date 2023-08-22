import enum
import os


import numpy as np
import torch

import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU

class ConstantsContext:
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    
    ACTIVE_MEAN = IMAGENET_MEAN
    ACTIVE_STD = IMAGENET_STD
    LOWER_IMAGE_BOUND = torch.tensor((-ACTIVE_MEAN / ACTIVE_STD).reshape(1, -1, 1, 1)).to(DEVICE)
    UPPER_IMAGE_BOUND = torch.tensor(((1 - ACTIVE_MEAN) / ACTIVE_STD).reshape(1, -1, 1, 1)).to(DEVICE)
    
    @classmethod
    def use_imagenet(cls):
        cls.ACTIVE_MEAN = cls.IMAGENET_MEAN
        cls.ACTIVE_STD = cls.IMAGENET_STD
        cls._update_bounds()

    @classmethod
    def use_clip(cls):
        cls.ACTIVE_MEAN = cls.CLIP_MEAN
        cls.ACTIVE_STD = cls.CLIP_STD
        cls._update_bounds()
        
    @classmethod
    def _update_bounds(cls):
        cls.LOWER_IMAGE_BOUND = torch.tensor((-cls.ACTIVE_MEAN / cls.ACTIVE_STD).reshape(1, -1, 1, 1)).to(DEVICE)
        cls.UPPER_IMAGE_BOUND = torch.tensor(((1 - cls.ACTIVE_MEAN) / cls.ACTIVE_STD).reshape(1, -1, 1, 1)).to(DEVICE)
        

class TRANSFORMS(enum.Enum):
    ZOOM = 0
    ZOOM_ROTATE = 1
    TRANSLATE = 2


class SupportedModels(enum.Enum):
    VGG16 = 0
    VGG16_EXPERIMENTAL = 1
    GOOGLENET = 2
    RESNET50 = 3
    ALEXNET = 4
    VIT = 5
    CLIPVITB32 = 6
    CLIPVITB16 = 7
    CLIPVITL14 = 8
    CLIPVITL14_336 = 9
    CLIPRN50 = 10
    CLIPRN101 = 11
    CLIPRN50x4 = 12
    CLIPRN50x16 = 13
    CLIPRN50x64 = 14


class SupportedPretrainedWeights(enum.Enum):
    IMAGENET = 0
    PLACES_365 = 1
    

FixedImageResolutions = {
    "VIT" : [224,224],
    "CLIPVITB32" : [224,224],
    "CLIPVITB16" : [224,224],
    "CLIPVITL14" : [224,224],
    "CLIPVITL14_336" : [336,336],
    "CLIPRN50" : [224,224],
    "CLIPRN101" : [224,224],
    "CLIPRN50x4" : [288,288],
    "CLIPRN50x16" : [384,384],
    "CLIPRN50x64" : [448,448],
}


FixedImageResolutionClasses = ["CLIP", "ViT"]


SUPPORTED_VIDEO_FORMATS = ['.mp4']
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']

BINARIES_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'binaries')
DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'data')

INPUT_DATA_PATH = os.path.join(DATA_DIR_PATH, 'input')
OUT_IMAGES_PATH = os.path.join(DATA_DIR_PATH, 'out-images')
OUT_VIDEOS_PATH = os.path.join(DATA_DIR_PATH, 'out-videos')
OUT_GIF_PATH = os.path.join(OUT_VIDEOS_PATH, 'GIFS')

# Make sure these exist as the rest of the code relies on it
os.makedirs(BINARIES_PATH, exist_ok=True)
os.makedirs(OUT_IMAGES_PATH, exist_ok=True)
os.makedirs(OUT_VIDEOS_PATH, exist_ok=True)
os.makedirs(OUT_GIF_PATH, exist_ok=True)




