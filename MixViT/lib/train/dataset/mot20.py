# adapted from tracking_net.py
import torch
import os
import os.path
import numpy as np
import pandas
import random
from collections import OrderedDict

from lib.train.data import jpeg4py_loader
from .base_video_dataset import BaseVideoDataset
from lib.train.admin import env_settings


class MOT20(BaseVideoDataset):
    """ MOT20 dataset.
    """
    def __init__(self, root=None, annotations=None, image_loader=jpeg4py_loader, split='train', data_fraction=None):
        """
        args:
            root        - The path to the MOT20 folder, containing the train, val, test sets.
            annotations - The path to annotation folder, containing train, val, test folders.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split (train) - dataset split
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        self.root = env_settings().mot20_dir if root is None else root
        self.annotations=os.path.join(root,'annotations') if annotations is None else annotations
        super().__init__('MOT20', root, image_loader)

        self.split=split
        self.sequence_list=[f[:-4] for f in os.listdir(os.path.join(annotations,split))]

        if data_fraction is not None:
            self.sequence_list = random.sample(
                self.sequence_list, int(len(self.sequence_list) * data_fraction))

    def get_name(self):
        return 'mot20'

    def _get_frame(self, seq_id, frame_id):
        seq_name=self.sequence_list[seq_id][:8]
        img1_path=os.path.join(self.root,self.split,seq_name,'img1')
        if self.split=='val_half':
            num_images=len([i for i in os.listdir(img1_path) if 'jpg' in i])
            frame_id=frame_id+num_images // 2 + 1
        frame_path = os.path.join(self.root,self.split,seq_name,'img1','{:0>6d}.jpg'.format(frame_id))
        return self.image_loader(frame_path)

    def _read_bb_anno(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        bb_anno_file = os.path.join(self.annotations,self.split,seq_name+'.txt')
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)

    def get_sequence_info(self, seq_id):
        bbox = self._read_bb_anno(seq_id)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def get_frames(self, seq_id, frame_ids, anno=None):
        frame_list = [self._get_frame(seq_id, f) for f in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': 'player',
                                   'seq':self.sequence_list[seq_id][:8],
                                   'id':self.sequence_list[seq_id][9:],
                                   'frames':frame_ids,})

        return frame_list, anno_frames, object_meta
