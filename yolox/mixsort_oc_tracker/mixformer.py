from MixViT.lib.models.mixformer_vit import build_mixformer_deit
from MixViT.lib.train.data.processing import MixformerProcessing as MP
from MixViT.lib.train.data.transforms import Transform, ToTensor, Normalize
import MixViT.lib.train.data.processing_utils as prutils
import MixViT.lib.train.admin.settings as ws_settings
import importlib
from MixViT.lib.train.base_functions import update_settings
import MixViT.lib.train.data.transforms as tfm
from typing import List, Union, Tuple
import torch,os
from torchvision.transforms.functional import resized_crop, normalize
import math
import numpy as np

class MixFormer:
    def __init__(self,args) -> None:
        # mixformer setting & cfg
        # adapted from lib/train/run_training.py & train_script_mixformer.py
        self.settings = ws_settings.Settings()
        self.settings.script_name = args.script
        self.settings.config_name = args.config
        self.radius=args.radius
        prj_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../MixViT")
        )
        self.settings.cfg_file = os.path.join(
            prj_dir, f"experiments/{args.script}/{args.config}.yaml"
        )
        config_module = importlib.import_module(
            "lib.config.%s.config" % self.settings.script_name
        )
        self.cfg = config_module.cfg
        config_module.update_config_from_file(self.settings.cfg_file)
        update_settings(self.settings, self.cfg)

        # need modification, for distributed
        network = build_mixformer_deit(self.cfg)
        self.network = network.cuda(torch.device(f"cuda:{args.local_rank}"))
        self.network.eval()

    def crop_and_resize(
        self, img: torch.Tensor, center: np.ndarray, s: str, annos: torch.Tensor = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """crop&resize the `img` centered at `center` and transform `annos` to the cropped position.

        Args:
            img (torch.Tensor): image to be cropped
            center (np.ndarray): center coord
            s (str): 'template' or 'search'
            annos (torch.Tensor, optional): boxes to be transformed. Defaults to None.

        Returns:
            Union[Tuple[torch.Tensor,torch.Tensor],torch.Tensor]: transfromed image (and boxes)
        """
        # compute params
        center = torch.from_numpy(center[:4].astype(np.int))
        search_area_factor = self.settings.search_area_factor[s]
        output_sz = self.settings.output_sz[s]
        x, y, w, h = [int(i) for i in center]
        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

        # x:left, y:top
        x = int(round(x + 0.5 * w - crop_sz * 0.5))
        y = int(round(y + 0.5 * h - crop_sz * 0.5))

        try:
            resized_img = resized_crop(
                img, y, x, crop_sz, crop_sz, [output_sz, output_sz]
            )
        except:  # too small box
            zero_img = torch.zeros((3, output_sz, output_sz)).cuda()
            return zero_img if annos is None else zero_img, []

        if annos is not None:
            # (origin_x - x, origin_y - y, origin_w, origin_h)/factor
            transforemd_coord = torch.cat(
                (annos[:, 0:2] - torch.tensor([x, y]), annos[:, 2:4]), dim=1
            )
            return resized_img, transforemd_coord / (crop_sz / output_sz)
        else:
            return resized_img

    def compute_vit_sim(self,detections,trackers,img,templates)-> np.ndarray:
        # x1,y1,w,h
        dets=np.concatenate((detections[:,0:2],detections[:,2:4]-detections[:,0:2]),axis=1).astype(np.int32)
        stracks=np.concatenate((trackers[:,0:2],trackers[:,2:4]-trackers[:,0:2]),axis=1).astype(np.int32)
        vit=np.zeros((len(stracks),len(dets)),dtype=np.float64)
        if min(vit.shape) ==0:
            return vit
        template_imgs=[templates[int(t[-1])] for t in trackers]
        search_imgs = []
        search_boxes = []
        search_bbox=torch.from_numpy(dets)
        for strack in stracks:
            # centered at predicted position
            # crop search area & transform det coord
            s_img, s_box = self.crop_and_resize(img, strack, "search", search_bbox)
            search_imgs.append(s_img)
            search_boxes.append(s_box)

        # img transform & compute
        template_imgs = normalize(
            torch.stack(template_imgs).float().div(255),
            self.cfg.DATA.MEAN,
            self.cfg.DATA.STD,
        )
        search_imgs = normalize(
            torch.stack(search_imgs).float().div(255),
            self.cfg.DATA.MEAN,
            self.cfg.DATA.STD,
        )
        heatmap = self.network(template_imgs, search_imgs).cpu().detach().numpy()
        # linear transform to [0,1]
        for i in range(heatmap.shape[0]):
            heatmap[i][0] = heatmap[i][0] - heatmap[i][0].min()
            heatmap[i][0] = heatmap[i][0] / heatmap[i][0].max()

        # compute similarity
        search_size = s_img[0].shape[-1]
        heatmap_size = heatmap.shape[-1]
        factor = search_size // heatmap_size
        for i, boxes in enumerate(search_boxes):
            # correspond to strack[i]
            for j, (t, l, w, h) in enumerate(boxes):
                cx, cy = t + w / 2, l + h / 2
                # don't consider outsiders
                if cx > 0 and cy > 0 and cx < search_size and cy < search_size:
                    cx, cy = int(cx) // factor, int(cy) // factor
                    top = max(0, cy - self.radius)
                    bottom = min(heatmap_size, cy + self.radius + 1)
                    left = max(0, cx - self.radius)
                    right = min(heatmap_size, cx + self.radius + 1)
                    vit[i][j] = heatmap[i][0][top:bottom, left:right].mean()

        return vit
