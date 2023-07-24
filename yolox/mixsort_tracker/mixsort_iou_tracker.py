import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resized_crop, normalize
import math
from torch.utils.tensorboard import SummaryWriter

from .kalman_filter import KalmanFilter
from . import matching
from .basetrack import BaseTrack, TrackState
from MixViT.lib.models.mixformer_vit import build_mixformer_deit
from MixViT.lib.train.data.processing import MixformerProcessing as MP
from MixViT.lib.train.data.transforms import Transform, ToTensor, Normalize
import MixViT.lib.train.data.processing_utils as prutils
import MixViT.lib.train.admin.settings as ws_settings
import importlib
from MixViT.lib.train.base_functions import update_settings
import MixViT.lib.train.data.transforms as tfm
from typing import List, Union, Tuple


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, iou):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.tlwh=self._tlwh
        self.is_activated = False
        self.template = None
        self._iou = iou

        self.score = score
        self.tracklet_len = 0

    def activate(self, frame_id, template):
        """Start a new tracklet"""
        self.track_id = self.next_id()
        self.template = template
        self.tlwh=self._tlwh

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False, template=None):
        self.tlwh=new_track.tlwh
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        if template is not None:
            self.template = template

    def update(self, new_track, frame_id, template=None):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.tlwh = new_track.tlwh

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

        if template is not None:
            self.template = template

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return "OT_{}_({}-{})".format(self.track_id, self.start_frame, self.end_frame)


class MIXTracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        # self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        self.last_img = None
        self.alpha = args.alpha
        self.radius = args.radius
        self.iou_thresh = args.iou_thresh

        # mixformer setting & cfg
        # adapted from lib/train/run_training.py & train_script_mixformer.py
        self.settings = ws_settings.Settings()
        self.settings.script_name = args.script
        self.settings.config_name = args.config
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

    def re_init(self, args, frame_rate=30):
        BaseTrack._count = 0 # set to 0 for new video
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        # self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        self.last_img = None
        self.alpha = args.alpha
        self.radius = args.radius
        self.iou_thresh = args.iou_thresh

    def visualize(self, logger: SummaryWriter, template, search, search_box):
        # utils for debugging
        logger.add_image("template", template)
        for box in search_box:
            box[2:4] = box[0:2] + box[2:4]
        logger.add_image_with_boxes("search", search, search_box)

    def visualize_box(self, logger: SummaryWriter, img, dets:List[STrack], name):
        # utils for debugging
        logger.add_image_with_boxes(name, img, np.array([s.tlbr for s in dets]),labels=[str(i) for i in range(len(dets))])

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
        center = torch.from_numpy(center.astype(np.int))
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

    @torch.no_grad()
    def compute_mix_dist(
        self,
        stracks: List[STrack],
        dets: List[STrack],
        img: torch.Tensor,
        fuse: bool = False,
    ) -> np.ndarray:
        """compute mix distance between stracks and dets.

        Args:
            stracks (List[STrack]): len = m
            dets (List[STrack]): len = n
            img (torch.Tensor): current image
            fuse (bool, optional): whether to fuse det score into iou. Defaults to False.

        Returns:
            np.ndarray: m x n
        """

        # Predict the current location with KF
        #STrack.multi_predict(stracks)
        # compute iou dist
        iou = matching.iou_distance(stracks, dets)
        if fuse:
            iou = matching.fuse_score(iou, dets)

        if len(stracks) * len(dets) == 0:
            return iou

        # for every strack, compute its vit-dist with dets
        search_bbox = torch.stack(
            [torch.from_numpy(det.tlwh.astype(np.int)) for det in dets]
        )
        search_imgs = []
        search_boxes = []
        # vit dist
        # self.logger=SummaryWriter('./debug_tensorboard')
        # self.visualize(self.logger,template_imgs[0],search_img,search_box.clone())
        # self.visualize_box(self.logger,img,stracks,"stracks")
        # self.visualize_box(self.logger,img,dets,"dets")
        vit = np.zeros((len(stracks), len(dets)), dtype=np.float64)
        template_imgs = [s.template for s in stracks]
        for strack in stracks:
            # centered at predicted position
            center = strack.tlwh
            # crop search area & transform det coord
            s_img, s_box = self.crop_and_resize(img, center, "search", search_bbox)
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
                    # vit[i][j] = heatmap[i][0][cy][cx]

        # fuse iou&vit cost
        return self.alpha * iou + (1 - self.alpha) * (1 - vit)
        # if iou.min()<self.args.fuse_iou_thresh:
        #     return iou
        # else:
        #     return 1-vit
        # vit=1-vit
        # for i in range(iou.shape[0]):
        #     for j in range(iou.shape[1]):
        #         if iou[i][j]>self.args.fuse_iou_thresh and vit[i][j]<self.args.fuse_vit_thresh:
        #             iou[i][j]=vit[i][j]
        # return iou

    def update(self, output_results, img_info, img_size, img):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        # compute all ious
        all_dets = [x for x in bboxes]
        iou = matching.ious(all_dets, all_dets)
        # compute max iou for every det
        max_iou = []
        for i in range(len(all_dets)):
            iou[i][i] = 0
            max_iou.append(iou[i].max())
        max_iou = np.array(max_iou)

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        max_iou_keep = max_iou[remain_inds]
        max_iou_second = max_iou[inds_second]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            """Detections"""
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, u)
                for (tlbr, s, u) in zip(dets, scores_keep, max_iou_keep)
            ]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        dists = self.compute_mix_dist(strack_pool, detections, img, fuse=True)
        # if not self.args.mot20:
        #     dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.args.match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            template = (
                self.crop_and_resize(img, det.tlwh, "template")
                if det._iou < self.iou_thresh
                else None
            )
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id, template)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, template=template)
                refind_stracks.append(track)

        """ Step 3: Second association, with low score detection boxes"""
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, u)
                for (tlbr, s, u) in zip(dets_second, scores_second, max_iou_second)
            ]
        else:
            detections_second = []
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        # dists = matching.iou_distance(r_tracked_stracks, detections_second)
        dists = self.compute_mix_dist(r_tracked_stracks, detections_second, img)
        matches, u_track, u_detection_second = matching.linear_assignment(
            dists, thresh=0.5
        )
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            template = (
                self.crop_and_resize(img, det.tlwh, "template")
                if det._iou < self.iou_thresh
                else None
            )
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, template)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, template=template)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        dists = self.compute_mix_dist(unconfirmed, detections, img, fuse=True)
        # dists = matching.iou_distance(unconfirmed, detections)
        # if not self.args.mot20:
        #     dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.7
        )
        for itracked, idet in matches:
            det = detections[idet]
            template = (
                self.crop_and_resize(img, det.tlwh, "template")
                if det._iou < self.iou_thresh
                else None
            )
            unconfirmed[itracked].update(detections[idet], self.frame_id, template)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            # do not consider iou constraint
            track.activate(
                self.frame_id,
                self.crop_and_resize(img, track._tlwh, "template"),
            )
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        self.last_img = img
        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.05)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
