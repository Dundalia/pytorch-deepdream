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
    ## OpenAI CLIP models
    CLIP_VITB32 = 6
    CLIP_VITB16 = 7
    CLIP_VITL14 = 8
    CLIP_VITL14_336 = 9
    CLIP_RN50 = 10
    CLIP_RN101 = 11
    CLIP_RN50x4 = 12
    CLIP_RN50x16 = 13
    CLIP_RN50x64 = 14
    ## OpenCLIP models
    OPENCLIP_COCA_BASE = 15
    OPENCLIP_COCA_ROBERTA_VIT_B_32 = 16
    OPENCLIP_COCA_VIT_B_32 = 17
    OPENCLIP_COCA_VIT_L_14 = 18
    OPENCLIP_CONVNEXT_BASE = 19
    OPENCLIP_CONVNEXT_BASE_W = 20
    OPENCLIP_CONVNEXT_BASE_W_320 = 21
    OPENCLIP_CONVNEXT_LARGE = 22
    OPENCLIP_CONVNEXT_LARGE_D = 23
    OPENCLIP_CONVNEXT_LARGE_D_320 = 24
    OPENCLIP_CONVNEXT_SMALL = 25
    OPENCLIP_CONVNEXT_TINY = 26
    OPENCLIP_CONVNEXT_XLARGE = 27
    OPENCLIP_CONVNEXT_XXLARGE = 28
    OPENCLIP_CONVNEXT_XXLARGE_320 = 29
    OPENCLIP_EVA01_G_14 = 30
    OPENCLIP_EVA01_G_14_PLUS = 31
    OPENCLIP_EVA02_B_16 = 32
    OPENCLIP_EVA02_E_14 = 33
    OPENCLIP_EVA02_E_14_PLUS = 34
    OPENCLIP_EVA02_L_14 = 35
    OPENCLIP_EVA02_L_14_336 = 36
    OPENCLIP_MT5_BASE_VIT_B_32 = 37
    OPENCLIP_MT5_XL_VIT_H_14 = 38
    OPENCLIP_RN50 = 39
    OPENCLIP_RN50_QUICKGELU = 40
    OPENCLIP_RN50X4 = 41
    OPENCLIP_RN50X16 = 42
    OPENCLIP_RN50X64 = 43
    OPENCLIP_RN101 = 44
    OPENCLIP_RN101_QUICKGELU = 45
    OPENCLIP_ROBERTA_VIT_B_32 = 46
    OPENCLIP_SWIN_BASE_PATCH4_WINDOW7_224 = 47
    OPENCLIP_VIT_B_16 = 48
    OPENCLIP_VIT_B_16_PLUS = 49
    OPENCLIP_VIT_B_16_PLUS_240 = 50
    OPENCLIP_VIT_B_32 = 51
    OPENCLIP_VIT_B_32_PLUS_256 = 52
    OPENCLIP_VIT_B_32_QUICKGELU = 53
    OPENCLIP_VIT_BIGG_14 = 54
    OPENCLIP_VIT_E_14 = 55
    OPENCLIP_VIT_G_14 = 56
    OPENCLIP_VIT_H_14 = 57
    OPENCLIP_VIT_H_16 = 58
    OPENCLIP_VIT_L_14 = 59
    OPENCLIP_VIT_L_14_280 = 60
    OPENCLIP_VIT_L_14_336 = 61
    OPENCLIP_VIT_L_16 = 62
    OPENCLIP_VIT_L_16_320 = 63
    OPENCLIP_VIT_M_16 = 64
    OPENCLIP_VIT_M_16_ALT = 65
    OPENCLIP_VIT_M_32 = 66
    OPENCLIP_VIT_M_32_ALT = 67
    OPENCLIP_VIT_S_16 = 68
    OPENCLIP_VIT_S_16_ALT = 69
    OPENCLIP_VIT_S_32 = 70
    OPENCLIP_VIT_S_32_ALT = 71
    OPENCLIP_VIT_MEDIUM_PATCH16_GAP_256 = 72
    OPENCLIP_VIT_RELPOS_MEDIUM_PATCH16_CLS_224 = 73
    OPENCLIP_XLM_ROBERTA_BASE_VIT_B_32 = 74
    OPENCLIP_XLM_ROBERTA_LARGE_VIT_H_14 = 75


class SupportedPretrainedWeights(enum.Enum):
    IMAGENET = 0
    PLACES_365 = 1
    CC12M = 2
    COMMONPOOL_L_BASIC_S1B_B8K = 3
    COMMONPOOL_L_CLIP_S1B_B8K = 4
    COMMONPOOL_L_IMAGE_S1B_B8K = 5
    COMMONPOOL_L_LAION_S1B_B8K = 6
    COMMONPOOL_L_S1B_B8K = 7
    COMMONPOOL_L_TEXT_S1B_B8K = 8
    COMMONPOOL_M_BASIC_S128M_B4K = 9
    COMMONPOOL_M_CLIP_S128M_B4K = 10
    COMMONPOOL_M_IMAGE_S128M_B4K = 11
    COMMONPOOL_M_LAION_S128M_B4K = 12
    COMMONPOOL_M_S128M_B4K = 13
    COMMONPOOL_M_TEXT_S128M_B4K = 14
    COMMONPOOL_S_BASIC_S13M_B4K = 15
    COMMONPOOL_S_CLIP_S13M_B4K = 16
    COMMONPOOL_S_IMAGE_S13M_B4K = 17
    COMMONPOOL_S_LAION_S13M_B4K = 18
    COMMONPOOL_S_S13M_B4K = 19
    COMMONPOOL_S_TEXT_S13M_B4K = 20
    COMMONPOOL_XL_CLIP_S13B_B90K = 21
    COMMONPOOL_XL_LAION_S13B_B90K = 22
    COMMONPOOL_XL_S13B_B90K = 23
    DATACOMP_L_S1B_B8K = 24
    DATACOMP_M_S128M_B4K = 25
    DATACOMP_S_S13M_B4K = 26
    DATACOMP_XL_S13B_B90K = 27
    FROZEN_LAION5B_S13B_B90K = 28
    LAION2B_E16 = 29
    LAION2B_S12B_B32K = 30
    LAION2B_S12B_B42K = 31
    LAION2B_S13B_B82K = 32
    LAION2B_S13B_B82K_AUGREG = 33
    LAION2B_S13B_B90K = 34
    LAION2B_S26B_B102K_AUGREG = 35
    LAION2B_S29B_B131K_FT = 36
    LAION2B_S29B_B131K_FT_SOUP = 37
    LAION2B_S32B_B79K = 38
    LAION2B_S32B_B82K = 39
    LAION2B_S34B_B79K = 40
    LAION2B_S34B_B82K_AUGREG = 41
    LAION2B_S34B_B82K_AUGREG_REWIND = 42
    LAION2B_S34B_B82K_AUGREG_SOUP = 43
    LAION2B_S34B_B88K = 44
    LAION2B_S39B_B160K = 45
    LAION2B_S4B_B115K = 46
    LAION2B_S9B_B144K = 47
    LAION400M_E31 = 48
    LAION400M_E32 = 49
    LAION400M_S11B_B41K = 50
    LAION400M_S13B_B51K = 51
    LAION5B_S13B_B90K = 52
    LAION_AESTHETIC_S13B_B82K = 53
    LAION_AESTHETIC_S13B_B82K_AUGREG = 54
    MERGED2B_S11B_B114K = 55
    MERGED2B_S4B_B131K = 56
    MERGED2B_S6B_B61K = 57
    MERGED2B_S8B_B131K = 58
    MSCOCO_FINETUNED_LAION2B_S13B_B90K = 59
    OPENAI = 60
    YFCC15M = 61


FixedImageResolutionClasses = ["CLIP", "OpenCLIP", "ViT"]


FixedImageResolutions = {
    "VIT" : [224,224],
    ## OpenAI CLIP Models
    "CLIP_VITB32" : (224,224),
    "CLIP_VITB16" : (224,224),
    "CLIP_VITL14" : (224,224),
    "CLIP_VITL14_336" : (336,336),
    "CLIP_RN50" : (224,224),
    "CLIP_RN101" : (224,224),
    "CLIP_RN50x4" : (288,288),
    "CLIP_RN50x16" : (384,384),
    "CLIP_RN50x64" : (448,448),
    ## OpenCLIP Models
    "OPENCLIP_COCA_BASE" : (288, 288),
    "OPENCLIP_COCA_ROBERTA_VIT_B_32" : (224, 224),
    "OPENCLIP_COCA_VIT_B_32" : (224, 224),
    "OPENCLIP_COCA_VIT_L_14" : (224, 224),
    "OPENCLIP_CONVNEXT_BASE" : (224, 224),
    "OPENCLIP_CONVNEXT_BASE_W" : (256, 256),
    "OPENCLIP_CONVNEXT_BASE_W_320" : (320, 320),
    "OPENCLIP_CONVNEXT_LARGE" : (224, 224),
    "OPENCLIP_CONVNEXT_LARGE_D" : (256, 256),
    "OPENCLIP_CONVNEXT_LARGE_D_320" : (320, 320),
    "OPENCLIP_CONVNEXT_SMALL" : (224, 224),
    "OPENCLIP_CONVNEXT_TINY" : (224, 224),
    "OPENCLIP_CONVNEXT_XLARGE" : (256, 256),
    "OPENCLIP_CONVNEXT_XXLARGE" : (256, 256),
    "OPENCLIP_CONVNEXT_XXLARGE_320" : (320, 320),
    "OPENCLIP_EVA01_G_14" : (224, 224),
    "OPENCLIP_EVA01_G_14_PLUS" : (224, 224),
    "OPENCLIP_EVA02_B_16" : (224, 224),
    "OPENCLIP_EVA02_E_14" : (224, 224),
    "OPENCLIP_EVA02_E_14_PLUS" : (224, 224),
    "OPENCLIP_EVA02_L_14" : (224, 224),
    "OPENCLIP_EVA02_L_14_336" : (336, 336),
    "OPENCLIP_MT5_BASE_VIT_B_32" : (224, 224),
    "OPENCLIP_MT5_XL_VIT_H_14" : (224, 224),
    "OPENCLIP_RN50" : (224, 224),
    "OPENCLIP_RN50_QUICKGELU" : (224, 224),
    "OPENCLIP_RN50X4" : (288, 288),
    "OPENCLIP_RN50X16" : (384, 384),
    "OPENCLIP_RN50X64" : (448, 448),
    "OPENCLIP_RN101" : (224, 224),
    "OPENCLIP_RN101_QUICKGELU" : (224, 224),
    "OPENCLIP_ROBERTA_VIT_B_32" : (224, 224),
    "OPENCLIP_SWIN_BASE_PATCH4_WINDOW7_224" : (224, 224),
    "OPENCLIP_VIT_B_16" : (224, 224),
    "OPENCLIP_VIT_B_16_PLUS" : (224, 224),
    "OPENCLIP_VIT_B_16_PLUS_240" : (240, 240),
    "OPENCLIP_VIT_B_32" : (224, 224),
    "OPENCLIP_VIT_B_32_PLUS_256" : (256, 256),
    "OPENCLIP_VIT_B_32_QUICKGELU" : (224, 224),
    "OPENCLIP_VIT_BIGG_14" : (224, 224),
    "OPENCLIP_VIT_E_14" : (224, 224),
    "OPENCLIP_VIT_G_14" : (224, 224),
    "OPENCLIP_VIT_H_14" : (224, 224),
    "OPENCLIP_VIT_H_16" : (224, 224),
    "OPENCLIP_VIT_L_14" : (224, 224),
    "OPENCLIP_VIT_L_14_280" : (280, 280),
    "OPENCLIP_VIT_L_14_336" : (336, 336),
    "OPENCLIP_VIT_L_16" : (224, 224),
    "OPENCLIP_VIT_L_16_320" : (320, 320),
    "OPENCLIP_VIT_M_16" : (224, 224),
    "OPENCLIP_VIT_M_16_ALT" : (224, 224),
    "OPENCLIP_VIT_M_32" : (224, 224),
    "OPENCLIP_VIT_M_32_ALT" : (224, 224),
    "OPENCLIP_VIT_S_16" : (224, 224),
    "OPENCLIP_VIT_S_16_ALT" : (224, 224),
    "OPENCLIP_VIT_S_32" : (224, 224),
    "OPENCLIP_VIT_S_32_ALT" : (224, 224),
    "OPENCLIP_VIT_MEDIUM_PATCH16_GAP_256" : (256, 256),
    "OPENCLIP_VIT_RELPOS_MEDIUM_PATCH16_CLS_224" : (224, 224),
    "OPENCLIP_XLM_ROBERTA_BASE_VIT_B_32" : (224, 224),
    "OPENCLIP_XLM_ROBERTA_LARGE_VIT_H_14" : (224, 224),
}


SupportedModel_to_ModelName = {
    ## OpenAI CLIP models
    "CLIP_VITB32" : "ViT-B/32",
    "CLIP_VITB16" : "ViT-B/16",
    "CLIP_VITL14" : "ViT-L/14",
    "CLIP_VITL14_336" : "ViT-L/14@336px",
    "CLIP_RN50" : "RN50",
    "CLIP_RN101" : "RN101",
    "CLIP_RN50x4" : "RN50x4",
    "CLIP_RN50x16" : "RN50x16",
    "CLIP_RN50x64" : "RN50x64",
    ## OpenCLIP models
    "OPENCLIP_COCA_BASE" : "coca_base",
    "OPENCLIP_COCA_ROBERTA_VIT_B_32" : "coca_roberta-ViT-B-32",
    "OPENCLIP_COCA_VIT_B_32" : "coca_ViT-B-32",
    "OPENCLIP_COCA_VIT_L_14" : "coca_ViT-L-14",
    "OPENCLIP_CONVNEXT_BASE" : "convnext_base",
    "OPENCLIP_CONVNEXT_BASE_W" : "convnext_base_w",
    "OPENCLIP_CONVNEXT_BASE_W_320" : "convnext_base_w_320",
    "OPENCLIP_CONVNEXT_LARGE" : "convnext_large",
    "OPENCLIP_CONVNEXT_LARGE_D" : "convnext_large_d",
    "OPENCLIP_CONVNEXT_LARGE_D_320" : "convnext_large_d_320",
    "OPENCLIP_CONVNEXT_SMALL" : "convnext_small",
    "OPENCLIP_CONVNEXT_TINY" : "convnext_tiny",
    "OPENCLIP_CONVNEXT_XLARGE" : "convnext_xlarge",
    "OPENCLIP_CONVNEXT_XXLARGE" : "convnext_xxlarge",
    "OPENCLIP_CONVNEXT_XXLARGE_320" : "convnext_xxlarge_320",
    "OPENCLIP_EVA01_G_14" : "EVA01-g-14",
    "OPENCLIP_EVA01_G_14_PLUS" : "EVA01-g-14-plus",
    "OPENCLIP_EVA02_B_16" : "EVA02-B-16",
    "OPENCLIP_EVA02_E_14" : "EVA02-E-14",
    "OPENCLIP_EVA02_E_14_PLUS" : "EVA02-E-14-plus",
    "OPENCLIP_EVA02_L_14" : "EVA02-L-14",
    "OPENCLIP_EVA02_L_14_336" : "EVA02-L-14-336",
    "OPENCLIP_MT5_BASE_VIT_B_32" : "mt5-base-ViT-B-32",
    "OPENCLIP_MT5_XL_VIT_H_14" : "mt5-xl-ViT-H-14",
    "OPENCLIP_RN50" : "RN50",
    "OPENCLIP_RN50_QUICKGELU" : "RN50-quickgelu",
    "OPENCLIP_RN50X4" : "RN50x4",
    "OPENCLIP_RN50X16" : "RN50x16",
    "OPENCLIP_RN50X64" : "RN50x64",
    "OPENCLIP_RN101" : "RN101",
    "OPENCLIP_RN101_QUICKGELU" : "RN101-quickgelu",
    "OPENCLIP_ROBERTA_VIT_B_32" : "roberta-ViT-B-32",
    "OPENCLIP_SWIN_BASE_PATCH4_WINDOW7_224" : "swin_base_patch4_window7_224",
    "OPENCLIP_VIT_B_16" : "ViT-B-16",
    "OPENCLIP_VIT_B_16_PLUS" : "ViT-B-16-plus",
    "OPENCLIP_VIT_B_16_PLUS_240" : "ViT-B-16-plus-240",
    "OPENCLIP_VIT_B_32" : "ViT-B-32",
    "OPENCLIP_VIT_B_32_PLUS_256" : "ViT-B-32-plus-256",
    "OPENCLIP_VIT_B_32_QUICKGELU" : "ViT-B-32-quickgelu",
    "OPENCLIP_VIT_BIGG_14" : "ViT-bigG-14",
    "OPENCLIP_VIT_E_14" : "ViT-e-14",
    "OPENCLIP_VIT_G_14" : "ViT-g-14",
    "OPENCLIP_VIT_H_14" : "ViT-H-14",
    "OPENCLIP_VIT_H_16" : "ViT-H-16",
    "OPENCLIP_VIT_L_14" : "ViT-L-14",
    "OPENCLIP_VIT_L_14_280" : "ViT-L-14-280",
    "OPENCLIP_VIT_L_14_336" : "ViT-L-14-336",
    "OPENCLIP_VIT_L_16" : "ViT-L-16",
    "OPENCLIP_VIT_L_16_320" : "ViT-L-16-320",
    "OPENCLIP_VIT_M_16" : "ViT-M-16",
    "OPENCLIP_VIT_M_16_ALT" : "ViT-M-16-alt",
    "OPENCLIP_VIT_M_32" : "ViT-M-32",
    "OPENCLIP_VIT_M_32_ALT" : "ViT-M-32-alt",
    "OPENCLIP_VIT_S_16" : "ViT-S-16",
    "OPENCLIP_VIT_S_16_ALT" : "ViT-S-16-alt",
    "OPENCLIP_VIT_S_32" : "ViT-S-32",
    "OPENCLIP_VIT_S_32_ALT" : "ViT-S-32-alt",
    "OPENCLIP_VIT_MEDIUM_PATCH16_GAP_256" : "vit_medium_patch16_gap_256",
    "OPENCLIP_VIT_RELPOS_MEDIUM_PATCH16_CLS_224" : "vit_relpos_medium_patch16_cls_224",
    "OPENCLIP_XLM_ROBERTA_BASE_VIT_B_32" : "xlm-roberta-base-ViT-B-32",
    "OPENCLIP_XLM_ROBERTA_LARGE_VIT_H_14" : "xlm-roberta-large-ViT-H-14",
}


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




