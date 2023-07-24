import os
import os.path
import torch
import numpy as np
import pandas
import csv
import random
import json
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class Tao(BaseVideoDataset):
    """ LaSOT dataset.

    Publication:
        LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
        Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao and Haibin Ling
        CVPR, 2019
        https://arxiv.org/pdf/1809.07845.pdf

    Download the dataset from https://cis.temple.edu/lasot/download.html
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split=None, data_fraction=None):
        """
        args:
            root - path to the lasot dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        # root = env_settings().lasot_dir if root is None else root
        root = env_settings().tao_dir if root is None else root
        super().__init__('Tao', root, image_loader)

        # Keep a list of all classes
        self.class_list = [f for f in os.listdir(self.root)]
        self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}

        self.sequence_list = self._build_sequence_list(vid_ids, split)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

        self.seq_per_class = self._build_class_list()

    def _build_sequence_list(self, vid_ids=None, split=None):
        # if split is not None:
        #     if vid_ids is not None:
        #         raise ValueError('Cannot set both split_name and vid_ids.')
        #     ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        #     if split == 'train':
        #         file_path = os.path.join(ltr_path, 'data_specs', 'lasot_train_split.txt')
        #     else:
        #         raise ValueError('Unknown split name.')
        #     sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
        # elif vid_ids is not None:
        #     sequence_list = [c+'-'+str(v) for c in self.class_list for v in vid_ids]
        # else:
        #     raise ValueError('Set either split_name or vid_ids.')

        # return sequence_list
        assert split == "train"
        self.annotation_path = os.path.join(self.root, "annotations", "{}_with_freeform.json".format(split))
        all_infos = json.load(open(self.annotation_path, 'r'))
        video_infos = all_infos['videos']
        annotation_infos = all_infos['annotations']
        tracks_infos = all_infos['tracks']
        img_infos = all_infos['images']

        trackid_to_anno_list = {}  # track_id: [anno1, anno2]
        for anno in annotation_infos:
            track_id = anno['track_id']
            if track_id in trackid_to_anno_list.keys():
                trackid_to_anno_list[track_id].append(anno)
            else:
                trackid_to_anno_list[track_id] = [anno]

        imgid_to_img_infos = {}
        for info in img_infos:
            image_id = info['id']
            imgid_to_img_infos[image_id] = info

        output = []
        for track in tracks_infos:
            track_id = track['id']
            anno_list = trackid_to_anno_list[track_id]
            track_item = {}
            frame_id_to_pathAndBbox = {}
            for i, anno in enumerate(anno_list):
                image_id = anno['image_id']
                imginfo = imgid_to_img_infos[image_id]
                if i == 0:
                    track_item['width'] = imginfo['width']
                    track_item['height'] = imginfo['height']

                frame_id = imginfo['frame_index']
                frame_path = imginfo['file_name']
                frame_bbox = anno['bbox']
                frame_id_to_pathAndBbox[frame_id] = [frame_path, frame_bbox]

            all_frame_numbers = max(frame_id_to_pathAndBbox.keys())
            visible_list = [False] * (all_frame_numbers + 1)  # including id 0
            for frame_id in frame_id_to_pathAndBbox.keys():
                visible_list[frame_id] = True

            track_item['visible'] = visible_list
            track_item['infos'] = frame_id_to_pathAndBbox
            # output[track_id] = track_item
            output.append(track_item)

        return output  # output [{visible:[true, false...], path_dict:{id: path, ...}, bbox_dict:{id: bbox, ...}}]
        # output {seq_id: {visible:[true, false...], path_dict:{id: path, ...}, bbox_dict:{id: bbox, ...}}}

    def _build_class_list(self):
        return None
        # seq_per_class = {}
        # for seq_id, seq_name in enumerate(self.sequence_list):
        #     class_name = seq_name.split('-')[0]
        #     if class_name in seq_per_class:
        #         seq_per_class[class_name].append(seq_id)
        #     else:
        #         seq_per_class[class_name] = [seq_id]

        # return seq_per_class

    def get_name(self):
        return 'tao'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        # return len(self.class_list)
        return 0

    def get_sequences_in_class(self, class_name):
        # return self.seq_per_class[class_name]
        return None

    # def _read_bb_anno(self, seq_path):
    #     bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
    #     gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
    #     return torch.tensor(gt)

    # def _read_target_visible(self, seq_path):
    #     # Read full occlusion and out_of_view
    #     occlusion_file = os.path.join(seq_path, "full_occlusion.txt")
    #     out_of_view_file = os.path.join(seq_path, "out_of_view.txt")

    #     with open(occlusion_file, 'r', newline='') as f:
    #         occlusion = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
    #     with open(out_of_view_file, 'r') as f:
    #         out_of_view = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])

    #     target_visible = ~occlusion & ~out_of_view

    #     return target_visible

    # def _get_sequence_path(self, seq_id):
    #     seq_name = self.sequence_list[seq_id]
    #     class_name = seq_name.split('-')[0]
    #     vid_id = seq_name.split('-')[1]

    #     return os.path.join(self.root, class_name, class_name + '-' + vid_id)

    def get_sequence_info(self, seq_id):
        # seq_path = self._get_sequence_path(seq_id)
        seq_info = self.sequence_list[seq_id]  # {visible:[true, false...], infos:{frame_id: [path, bbox]...}}
        # bbox = self._read_bb_anno(seq_path)
        infos = seq_info["infos"]
        visible = seq_info['visible']
        valid = [False] * len(visible)
        bbox = torch.zeros(len(visible), 4)
        for frame_id in infos.keys():
            bbox_anno = infos[frame_id][1]
            bbox[frame_id] = torch.tensor(bbox_anno)
            valid[frame_id] = (bbox_anno[2] > 0) & (bbox_anno[3] > 0)

        return {'bbox': bbox, 'valid': torch.ByteTensor(valid), 'visible': torch.torch.ByteTensor(visible)}

    # def _get_frame_path(self, seq_path, frame_id):

    #     return os.path.join(seq_path, 'img', '{:08}.jpg'.format(frame_id+1))    # frames start from 1

    # def _get_frame(self, seq_path, frame_id):
    #     return self.image_loader(self._get_frame_path(seq_path, frame_id))
    def _get_frame(self, seq_info, frame_id):
        infos = seq_info['infos']
        frame_path = infos[frame_id][0]
        frame_path = os.path.join(self.root, "frames", frame_path)
        return self.image_loader(frame_path)

    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-2]
        return raw_class

    def get_class_name(self, seq_id):
        return "padding"
        # seq_path = self._get_sequence_path(seq_id)
        # obj_class = self._get_class(seq_path)

        # return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        # seq_path = self._get_sequence_path(seq_id)
        seq_info = self.sequence_list[seq_id]  # {visible:[true, false...], infos:{frame_id: [path, bbox]...}}

        # obj_class = self._get_class(seq_path)
        obj_class = "padding"
        frame_list = [self._get_frame(seq_info, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            # print(type(value))
            # print(key, type(key))
            # print(frame_ids[0], type(frame_ids[0]))
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
