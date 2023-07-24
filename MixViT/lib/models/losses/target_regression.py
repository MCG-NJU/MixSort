import torch
import math
from torch import nn


class IOULoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(IOULoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 2]
        pred_right = pred[:, 1]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 2]
        target_right = target[:, 1]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            if self.reduction == 'mean':
                return losses.mean()
            else:
                return losses.sum()


class CIOULoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target, weight=None):
        '''
        pred.shape: (n, 4)
        target.shape: (n, 4)
        值为该点到 gt 的四条边的距离
        '''
        pred_left = pred[:, 0]
        pred_top = pred[:, 2]
        pred_right = pred[:, 1]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 2]
        target_right = target[:, 1]
        target_bottom = target[:, 3]

        pred_w = pred_left + pred_right
        pred_h = pred_top + pred_bottom
        target_w = target_left + target_right
        target_h = target_top + target_bottom

        target_area = target_w * target_h
        pred_area = pred_w * pred_h

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        center_pred_x = (pred_right - pred_left) / 2
        center_pred_y = (pred_top - pred_bottom) / 2
        center_target_x = (target_right - target_left) / 2
        center_target_y = (target_top - target_left) / 2

        inter_diag = (center_pred_x - center_target_x) ** 2 + (center_pred_y - center_target_y) ** 2
        c_diag = (torch.max(pred_left, target_left) + torch.max(pred_right, target_right)) ** 2 + (
                    torch.max(pred_top, target_top) + torch.max(pred_bottom, target_bottom)) ** 2

        u = inter_diag / c_diag
        iou = area_intersect / area_union

        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(pred_w / pred_h) - torch.atan(pred_w / pred_h)), 2)
        with torch.no_grad():
            S = (iou > 0.5).float()
            alpha = S * v / (1 - iou + v)
        cious = iou - u - alpha * v
        cious = torch.clamp(cious, min=-1.0, max=1.0)
        losses = 1 - cious

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            if self.reduction == 'mean':
                return losses.mean(), iou
            else:
                return losses.sum(), iou


class REGLoss(nn.Module):
    def __init__(self, dim=4, loss_type='iou', reduction='mean'):
        super(REGLoss, self).__init__()
        self.dim = dim
        assert loss_type == 'l1' or loss_type == 'iou' or loss_type == 'ciou'
        self.loss_type = loss_type
        self.reduction = reduction
        if loss_type == 'iou':
            self.loss = IOULoss(reduction=reduction)
        elif loss_type == 'ciou':
            self.loss = CIOULoss(reduction=reduction)
        elif loss_type == 'l1':
            self.loss = nn.L1Loss(reduction=reduction)
        else:
            raise ValueError

    def forward(self, output, target, mask=None, radius=2):
        '''
        output.shape: (b, c, w, h)
        target.shape: (b, c, w, h)
        '''
        b, h, w = output.size(1), output.size(-1), output.size(-2)
        # num_reduce_nodes = b * (((radius-1)*2+1) * ((radius-1)*2+1) - 4)
        output = output.view(-1, self.dim, h, w).permute(0, 2, 3, 1)  # (b, w, h, c)
        target = target.view(-1, self.dim, h, w).permute(0, 2, 3, 1)
        if mask is not None:
            mask = mask.view(-1, h, w, 1).repeat(1, 1, 1, self.dim).bool()
            output = output[mask].view(-1, self.dim)
            target = target[mask].view(-1, self.dim)
            # print("output: {}".format(output))
            # print("target: {}".format(target))

        if 'iou' in self.loss_type:
            loss, iou = self.loss(output, target)
            return loss, iou
        else:
            loss = self.loss(output, target)
            return loss


def generate_regression_label(boxes_xyxy, size=20):
    '''
    Generate a gt boxes map from a single gt bbox
    boxes_xyxy: gt box xyxy, shape (b, 4) normalized value in [0, 1)
    '''
    boxes_xyxy = boxes_xyxy * size
    x = torch.arange(size, dtype=torch.float32, device=boxes_xyxy.device).view(1, 1, -1)
    y = torch.arange(size, dtype=torch.float32, device=boxes_xyxy.device).view(1, -1, 1)

    l = (x - boxes_xyxy[:, 0].view(-1, 1, 1)).repeat(1, size, 1)  # / spatial_scale # (n, 1, size) --> (n, size, size)
    r = (boxes_xyxy[:, 2].view(-1, 1, 1) - x).repeat(1, size, 1)  # / spatial_scale # (n, 1, size) --> (n, size, size)
    t = (y - boxes_xyxy[:, 1].view(-1, 1, 1)).repeat(1, 1, size)  # / spatial_scale # (n, size, 1) --> (n, size, size)
    b = (boxes_xyxy[:, 3].view(-1, 1, 1) - y).repeat(1, 1, size)  # / spatial_scale # (n, size, 1) --> (n, size, size)

    reg_labels = torch.stack([l, r, t, b], dim=1)  # (n, 4, size, size)
    return reg_labels


def generate_regression_mask(box_xywh, reg_labels=None, radius=2, mask_size=20):
    """
    NHW format
    :param box_xywh: gt_box, shape (b, 4), norm
    :param radius:
    :param mask_size: size of feature map
    :return:
    """
    target_center = (box_xywh[:, 0:2] + 0.5 * box_xywh[:, 2:4]) * mask_size
    target_center = target_center.int().float()
    k0 = torch.arange(mask_size, dtype=torch.float32, device=target_center.device).view(1, 1, -1)
    k1 = torch.arange(mask_size, dtype=torch.float32, device=target_center.device).view(1, -1, 1)

    d0 = k0 - target_center[:, 0].view(-1, 1, 1)
    d1 = k1 - target_center[:, 1].view(-1, 1, 1)
    dist = d0.abs() + d1.abs()

    rPos = radius + 0.1
    mask = torch.where(dist <= rPos, torch.ones_like(dist), torch.zeros_like(dist))  # (n, size, size)

    # remove the points outside the target box.
    if reg_labels is not None:
        mask = mask.unsqueeze(1).repeat(1, reg_labels.size(1), 1, 1)
        mask = torch.where(reg_labels <= 0, torch.zeros_like(mask), mask)
        mask = mask.min(dim=1)[0]  # (n, size, size)

    return mask
