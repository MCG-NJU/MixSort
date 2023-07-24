import torch
from segmentation.models.model import build_seg_model
from segmentation.config.model.config import cfg

def get_seg_model():
    model = build_seg_model(cfg)
    try:
        ckpt = torch.load("/data0/cyt/experiments/MixFormerPP/models/deeplab.ckpt", map_location='cpu')
        missing, unexpected = model.load_state_dict(ckpt['state_dict'])
        print("Segmentation model missing keys: {}".format(missing))
        print("Segmentation model unexpected keys: {}".format(unexpected))
    except:
        print("Segmentation model checkpoint is not loaded!")
    return model
