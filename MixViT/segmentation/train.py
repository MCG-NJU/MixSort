import os
from sklearn.utils import shuffle
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader

from segmentation.dataset.youtube_image import Youtube_VOS
from segmentation.models.model import build_seg_model
from segmentation.config.model.config import cfg


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_epoch_frequency,
        prefix="segmentation",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_epoch_frequency = save_epoch_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_epoch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        if self.use_modelcheckpoint_filename:
            filename = trainer.checkpoint_callback.filename
        else:
            filename = f"{self.prefix}_{epoch=}.ckpt"
        ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
        trainer.save_checkpoint(ckpt_path)


def main():
    train_dataset = Youtube_VOS(cfg, split="train")
    valid_dataset = Youtube_VOS(cfg, split="valid")
    # test_dataset  = Youtube_VOS(split="test")

    loader_args = dict(batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.TRAIN.WORKERS)

    train_dataloader = DataLoader(train_dataset, persistent_workers=True, shuffle=True, **loader_args) # {"img": [b, 3, w, h], "mask": [b, 3, w, h]}
    valid_dataloader = DataLoader(valid_dataset, persistent_workers=True, shuffle=False, **loader_args)
    # test_dataloader  = DataLoader(test_dataset, **loader_args)
    print("Dataloader created!")

    model = build_seg_model(cfg)
    print("Model created!")

    trainer = pl.Trainer(
        gpus=7, 
        strategy="ddp",
        max_epochs=cfg.TRAIN.EPOCHS,
        callbacks=[CheckpointEveryNSteps(save_epoch_frequency=1)],
        default_root_dir='/data1/songtianhui/segmentation',
    )
    print("Trainer created!")

    print("Start training...")
    trainer.fit(
        model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=valid_dataloader,
    )


if __name__ == "__main__":
    main()