from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch


class MixFormerppActor(BaseActor):
    """ Actor for training the TSP_online and TSP_cls_online"""
    def __init__(self, net, objective, loss_weight, settings, run_score_head=False):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.run_score_head = run_score_head

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox',
                    for two-branch head, also contain 'reg_label', 'cls_label', 'reg_mask' with shape (b, 4, w, h), (b, w, h), (b, w, h)
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data, run_score_head=self.run_score_head)

        # process the groundtruth
        gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)
        reg_labels = data['reg_labels'][0]  # (b, 4, w, h)
        reg_mask = data['reg_mask'][0] # (b, w, h)
        cls_labels = data['cls_labels'][0] # (b, w, h)

        labels = None
        if 'pred_scores' in out_dict:
            try:
                labels = data['label'].view(-1)  # (batch, ) 0 or 1
            except:
                raise Exception("Please setting proper labels for score branch.")

        # compute losses
        loss, status = self.compute_losses(out_dict, reg_labels, reg_mask, cls_labels, labels=labels)

        return loss, status

    def forward_pass(self, data, run_score_head):
        search_bboxes = box_xywh_to_xyxy(data['search_anno'][0].clone())
        out_dict, _ = self.net(data['template_images'][0], data['template_images'][1], data['search_images'],
                               run_score_head=run_score_head, gt_bboxes=search_bboxes)
        # out_dict: (B, N, C), outputs_coord: (1, B, N, C), target_query: (1, B, N, C)
        return out_dict

    def compute_losses(self, pred_dict, reg_labels, reg_mask, cls_labels, return_status=True, labels=None):
        # gt_bbox: (b, 4) xywh

        # Get boxes
        pred_boxes_map = pred_dict['pred_boxes']    # (b, 4, w, h) lrtb
        pred_cls_map = pred_dict['pred_cls']    # (b, 1, w, h)
        if torch.isnan(pred_boxes_map).any():
            raise ValueError("Network outputs is NAN! Stop Training")

        # compute ciou and iou
        try:
            ciou_loss, iou = self.objective['ciou'](pred_boxes_map, reg_labels, reg_mask)  # (b, 4, w, h) (b, 4, w, h)
        except:
            ciou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_map, reg_labels, reg_mask)  # (b, 4, w, h) (b, 4, w, h)

        # compute hinge loss
        hinge_loss = self.objective['hinge'](pred_cls_map.view(-1), cls_labels.view(-1)) # (b, 1, w, h), (b, w, h)

        # weighted sum
        loss = self.loss_weight['ciou'] * ciou_loss + self.loss_weight['l1'] * l1_loss + hinge_loss * self.loss_weight['hinge']


        # compute SPM loss if neccessary
        if 'pred_scores' in pred_dict:
            score_loss = self.objective['score'](pred_dict['pred_scores'].view(-1), labels)
            loss = score_loss * self.loss_weight['score']


        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            if 'pred_scores' in pred_dict:
                status = {"Loss/total": loss.item(),
                          "Loss/scores": score_loss.item()}
                # status = {"Loss/total": loss.item(),
                #           "Loss/scores": score_loss.item(),
                #           "Loss/giou": giou_loss.item(),
                #           "Loss/l1": l1_loss.item(),
                #           "IoU": mean_iou.item()}
            else:
                status = {"Loss/total": loss.item(),
                          "Loss/ciou": ciou_loss.item(),
                          "Loss/l1": l1_loss.item(),
                          "Loss/hinge": hinge_loss.item(),
                          "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
