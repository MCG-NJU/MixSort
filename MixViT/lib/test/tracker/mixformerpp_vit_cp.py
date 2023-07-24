from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
from lib.models.mixformer_vit import build_mixformerpp_vit_multi_score
from lib.test.tracker.tracker_utils import Preprocessor_wo_mask, hann2d, max2d
from lib.utils.box_ops import clip_box, box_xyxy_to_cxcywh
import math


class MixFormerPP(BaseTracker):
    def __init__(self, params, dataset_name):
        super(MixFormerPP, self).__init__(params)
        network = build_mixformerpp_vit_multi_score(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor_wo_mask()
        self.state = None
        # for debug
        self.debug = True
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        # self.z_dict1 = {}

        # Set the update interval
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.DATA.MAX_SAMPLE_INTERVAL
        print("Update interval is: ", self.update_intervals)

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        template = self.preprocessor.process(z_patch_arr)
        self.template = template
        self.template_mask = None

        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        # print("frame id: {}".format(self.frame_id))
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        # print("search shape: {}".format(search.shape))
        with torch.no_grad():
            out_dict, _, self.template_mask = self.network(self.template, self.template, search, run_score_head=False, template_mask=self.template_mask)
            # print("out_dict: {}".format(out_dict))

        pred_boxes = out_dict['pred_boxes']  # [1, 4, h, w]
        pred_cls = out_dict['pred_cls']  # [1, 1, h, w]
        pred_box = self.box_decode(pred_boxes, pred_cls)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_box * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # update template
        # for idx, update_i in enumerate(self.update_intervals):
        #     if self.frame_id % update_i == 0:
        #         z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
        #                                                     output_sz=self.params.template_size)  # (x1, y1, w, h)
        #         self.online_template = self.preprocessor.process(z_patch_arr)


        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def box_decode(self, pred_boxes, pred_cls):
        max_disp, _ = self.localize_target(pred_cls)
        max_disp = max_disp.long()
        feat_sz = pred_boxes.size(-1)
        pred_boxes_lrtb = torch.relu(pred_boxes.squeeze())
        # pred_box_lrtb = pred_boxes_lrtb[..., max_disp[0], max_disp[1]]
        pred_box_lrtb = pred_boxes_lrtb[..., max_disp[1], max_disp[0]]
        center_norm = max_disp / feat_sz
        # pred_box_xyxy = torch.tensor([center_norm[1] - pred_box_lrtb[0],
        #                               center_norm[0] - pred_box_lrtb[2],
        #                               center_norm[1] + pred_box_lrtb[1],
        #                               center_norm[0] + pred_box_lrtb[3]]).to(pred_boxes.device)
        pred_box_xyxy = torch.tensor([center_norm[0] - pred_box_lrtb[0],
                                      center_norm[1] - pred_box_lrtb[2],
                                      center_norm[0] + pred_box_lrtb[1],
                                      center_norm[1] + pred_box_lrtb[3]]).to(pred_boxes.device)
        pred_box_cxcywh = box_xyxy_to_cxcywh(pred_box_xyxy)
        return pred_box_cxcywh

    def localize_target(self, scores, sample_pos=None, sample_scales=None):
        """Run the target localization."""

        scores = scores.squeeze(1)

        preprocess_method = self.params.get('score_preprocess', 'none')
        if preprocess_method == 'none':
            pass
        else:
            raise Exception('Unknown score_preprocess in params.')

        if self.params.get('advanced_localization', False):
            return self.localize_advanced(scores, sample_pos, sample_scales)

        # Get maximum
        max_score, max_disp = max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp[scale_ind,...].float().cpu().view(-1)

        return max_disp, None

    def localize_advanced(self, scores, sample_scales, feature_sz):
        """Run the target advanced localization (as in ATOM).
        """
        if scores.dim() == 4:
            scores.squeeze(1)

        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz))
        score_center = (score_sz - 1) / 2

        self.output_window = hann2d(score_sz.long(), centered=True).to(scores.device)
        self.output_window = self.output_window.squeeze(0)

        scores_hn = scores
        if self.output_window is not None and getattr(self.params, 'perform_hn_without_windowing', False):
            scores_hn = scores.clone()
            scores *= self.output_window

        max_score1, max_disp1 = max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        sample_scale = sample_scales[scale_ind]
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind, ...].float().cpu().view(-1)
        target_disp1 = max_disp1 - score_center
        translation_vec1 = target_disp1 * (self.img_support_sz / feature_sz) * sample_scale

        # print("max_score_{}: {}".format(feature_sz, max_score1.item()))

        # Mask out target neighborhood
        target_neigh_sz = self.params.target_neighborhood_scale * (self.target_sz / sample_scale) * (
                    feature_sz / self.img_support_sz)

        tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
        scores_masked = scores_hn[scale_ind:scale_ind + 1, ...].clone()
        scores_masked[..., tneigh_top:tneigh_bottom, tneigh_left:tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = max2d(scores_masked)
        # print("max_score2: {}".format(max_score2.item()))

        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - score_center
        translation_vec2 = target_disp2 * (self.img_support_sz / feature_sz) * sample_scale

        if max_score1.item() < self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'not_found', max_disp1, max_disp2, None

        # Handle the different cases
        if max_score2 > self.params.distractor_threshold * max_score1:
            disp_norm1 = torch.sqrt(torch.sum(target_disp1 ** 2))
            disp_norm2 = torch.sqrt(torch.sum(target_disp2 ** 2))
            disp_threshold = self.params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'hard_negative', max_disp1, max_disp2, max_score1.item()
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores_hn, 'hard_negative', max_disp2, max_disp1, max_score2.item()
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'uncertain', max_disp1, max_disp2, max_score1.item()

            # If also the distractor is close, return with highest score
            return translation_vec1, scale_ind, scores_hn, 'uncertain', max_disp1, max_disp2, max_score1.item()

        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'hard_negative', max_disp1, max_disp2, max_score1.item()

        return translation_vec1, scale_ind, scores_hn, 'normal', max_disp1, max_disp2, max_score1.item()


def get_tracker_class():
    return MixFormerPP
