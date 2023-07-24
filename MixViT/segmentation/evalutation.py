import os
from sklearn.utils import shuffle
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader

from segmentation.dataset.youtube_image import Youtube_VOS
from segmentation.models.model import build_seg_model
from segmentation.config.model.config import cfg

os.environ['CUDA_VISIBLE_DEVICES']='7'

if __name__ == "__main__":
    valid_dataset = Youtube_VOS(cfg, split="valid")
    loader_args = dict(batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, **loader_args)
    
    model = build_seg_model(cfg)
    
    trainer = pl.Trainer(gpus=1)
    
    for i in range(80):
        ckpt_path = "/data1/songtianhui/segmentation/lightning_logs/version_2/checkpoints/segmentation_epoch={}.ckpt".format(i)
        if os.path.isfile(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path)
                model.load_state_dict(ckpt['state_dict'])
                valid_metrics = trainer.validate(model, dataloaders=valid_dataloader)
                print(ckpt_path + ":")
                print(valid_metrics)
            except:
                pass
            
    print("Done!")