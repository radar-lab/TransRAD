import math
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_bbox import dist2bbox, make_anchors, make_3Danchors, DecodeBox
from utils.helper import NMS_3D_Lei, NMS_2D_Lei,torch_iou3d

def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9, roll_out=False):
    """select the positive anchor center in gt

    Args:
        xy_centers (Tensor): shape(h*w, 4)
        gt_bboxes (Tensor): shape(b, n_boxes, 4)
    Return:
        (Tensor): shape(b, n_boxes, h*w)
    """
    n_anchors       = xy_centers.shape[0]
    bs, n_boxes, _  = gt_bboxes.shape

    if roll_out:
        bbox_deltas = torch.empty((bs, n_boxes, n_anchors), device=gt_bboxes.device)
        for b in range(bs):
            lt, rb          = gt_bboxes[b].view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
            bbox_deltas[b]  = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]),
                                       dim=2).view(n_boxes, n_anchors, -1).amin(2).gt_(eps)
        return bbox_deltas
    else:

        lt, rb      = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  

        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
        return bbox_deltas.amin(3).gt_(eps)


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.

    Args:
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
        overlaps (Tensor): shape(b, n_max_boxes, h*w)
    Return:
        target_gt_idx (Tensor): shape(b, h*w)
        fg_mask (Tensor): shape(b, h*w)
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
    """
    # b, n_max_boxes, 8400 -> b, 8400
    fg_mask = mask_pos.sum(-2) 
    if fg_mask.max() > 1:  

        # b, n_max_boxes, 8400
        mask_multi_gts      = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])  

        # b, 8400
        max_overlaps_idx    = overlaps.argmax(1)  
        # b, 8400, n_max_boxes
        is_max_overlaps     = F.one_hot(max_overlaps_idx, n_max_boxes)  
        # b, n_max_boxes, 8400
        is_max_overlaps     = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)  

        # b, n_max_boxes, 8400
        mask_pos            = torch.where(mask_multi_gts, is_max_overlaps, mask_pos) 

        fg_mask             = mask_pos.sum(-2)

    target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
    return target_gt_idx, fg_mask, mask_pos


class TaskAlignedAssigner(nn.Module):

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9, roll_out_thr=0):
        super().__init__()
        self.topk           = topk
        self.num_classes    = num_classes
        self.bg_idx         = num_classes
        self.alpha          = alpha
        self.beta           = beta
        self.eps            = eps
        # roll_out_thr为64
        self.roll_out_thr   = roll_out_thr

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor)  : shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor)  : shape(bs, num_total_anchors, 4)
            anc_points (Tensor) : shape(num_total_anchors, 2)
            gt_labels (Tensor)  : shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor)  : shape(bs, n_max_boxes, 4)
            mask_gt (Tensor)    : shape(bs, n_max_boxes, 1)
        Returns:
            target_labels (Tensor)  : shape(bs, num_total_anchors)
            target_bboxes (Tensor)  : shape(bs, num_total_anchors, 4)
            target_scores (Tensor)  : shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor)        : shape(bs, num_total_anchors)
        """

        self.bs             = pd_scores.size(0)

        self.n_max_boxes    = gt_bboxes.size(1)

        self.roll_out       = self.n_max_boxes > self.roll_out_thr if self.roll_out_thr else False
    
        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))

        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)


        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)


        target_labels, target_bboxes, target_scores,target_gt_idx = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)


        align_metric        *= mask_pos

        pos_align_metrics   = align_metric.amax(axis=-1, keepdim=True) 

        pos_overlaps        = (overlaps * mask_pos).amax(axis=-1, keepdim=True)

        norm_align_metric   = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        # Normalize norm_align_metric to a range that does not excessively reduce the scores
        norm_align_metric = torch.clamp(norm_align_metric, min=0.1, max=1.0)
        #target_scores       = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):

        align_metric, overlaps  = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        

        mask_in_gts             = select_candidates_in_gts(anc_points, gt_bboxes, roll_out=self.roll_out)
        # get topk_metric mask      b, max_num_obj, 8400
        mask_topk               = self.select_topk_candidates(align_metric * mask_in_gts, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())

        mask_pos                = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        if self.roll_out:
            align_metric    = torch.empty((self.bs, self.n_max_boxes, pd_scores.shape[1]), device=pd_scores.device)
            overlaps        = torch.empty((self.bs, self.n_max_boxes, pd_scores.shape[1]), device=pd_scores.device)
            ind_0           = torch.empty(self.n_max_boxes, dtype=torch.long)
            for b in range(self.bs):
                ind_0[:], ind_2 = b, gt_labels[b].squeeze(-1).long()
                # bs, max_num_obj, 8400
                bbox_scores     = pd_scores[ind_0, :, ind_2]  
                # bs, max_num_obj, 8400
                overlaps[b]     = bbox_iou(gt_bboxes[b].unsqueeze(1), pd_bboxes[b].unsqueeze(0), xywh=False, CIoU=True)[0].squeeze(2).clamp(0)
                align_metric[b] = bbox_scores.pow(self.alpha) * overlaps[b].pow(self.beta)
        else:
            # 2, b, max_num_obj
            ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)       

            ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)  

            ind[1] = gt_labels.long().squeeze(-1) 

            # b, max_num_obj, 8400
            bbox_scores = pd_scores[ind[0], :, ind[1]]  


            # bs, max_num_obj, 8400
            overlaps        = bbox_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1), xywh=False, CIoU=True)[0].squeeze(3).clamp(0)#.clamp(0)将张量中的所有元素限制为不小于0
            align_metric    = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Args:
            metrics     : (b, max_num_obj, h*w).
            topk_mask   : (b, max_num_obj, topk) or None
        """
        # 8400
        num_anchors             = metrics.shape[-1] 
        # b, max_num_obj, topk
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.topk])
        # b, max_num_obj, topk
        topk_idxs[~topk_mask] = 0
        # b, max_num_obj, topk, 8400 -> b, max_num_obj, 8400

        if self.roll_out:
            is_in_topk = torch.empty(metrics.shape, dtype=torch.long, device=metrics.device)
            for b in range(len(topk_idxs)):
                is_in_topk[b] = F.one_hot(topk_idxs[b], num_anchors).sum(-2)
        else:
            is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(-2)

        is_in_topk = torch.where(is_in_topk > 1, 0, is_in_topk)
        return is_in_topk.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Args:
            gt_labels       : (b, max_num_obj, 1)
            gt_bboxes       : (b, max_num_obj, 4)
            target_gt_idx   : (b, h*w)
            fg_mask         : (b, h*w)
        """

        batch_ind       = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        # b, h*w   
        target_gt_idx   = target_gt_idx + batch_ind * self.n_max_boxes
        # b, h*w    
        target_labels   = gt_labels.long().flatten()[target_gt_idx]
        # b, h*w, 4 
        target_bboxes   = gt_bboxes.view(-1, 4)[target_gt_idx]
        
        # assigned target scores
        target_labels.clamp(0)
        # one_hot
        target_scores   = F.one_hot(target_labels, self.num_classes)  # (b, h*w, 80)
        fg_scores_mask  = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores   = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores,target_gt_idx

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps
    
    # center_distance-Lei
    cent_dist = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha), cent_dist  # CIoU
            return iou - rho2 / c2, cent_dist  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area, cent_dist  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou, cent_dist  # IoU

def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)  # dist (lt, rb)

class BboxLoss(nn.Module):
    def __init__(self, reg_max=16, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):

        weight      = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)

        iou, cent_dist         = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)

        loss_iou    = ((1.0 - iou) * weight).sum() / target_scores_sum
        
        loss_cent    = (cent_dist * weight).sum() / target_scores_sum


        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl, loss_cent

    @staticmethod
    def _df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right

        return (F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr).mean(-1, keepdim=True)

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def xyzwhd2xyzxyz(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 3] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 4] / 2  # top left y
    y[..., 2] = x[..., 2] - x[..., 5] / 2  # top left z
    y[..., 3] = x[..., 0] + x[..., 3] / 2  # bottom right x
    y[..., 4] = x[..., 1] + x[..., 4] / 2  # bottom right y
    y[..., 5] = x[..., 2] + x[..., 5] / 2  # bottom right y
    return y

def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x, y) is the
    center point and width and height are the dimensions.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # center x
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # center y
    y[..., 2] = x[..., 2] - x[..., 0]        # width
    y[..., 3] = x[..., 3] - x[..., 1]        # height
    return y

def xyzxyz2xyzwhd(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x, y) is the
    center point and width and height are the dimensions.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 3]) / 2  # center x
    y[..., 1] = (x[..., 1] + x[..., 4]) / 2  # center y
    y[..., 2] = (x[..., 2] + x[..., 5]) / 2  # center z
    y[..., 3] = x[..., 3] - x[..., 0]        # width
    y[..., 4] = x[..., 4] - x[..., 1]        # height
    y[..., 5] = x[..., 5] - x[..., 2]        # depth
    return y

def compute_class_weights(labels,min_weight = 0.05):
    '''
    labels = torch.tensor([
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]])
    '''
    class_counts = labels.sum(dim=0)
    class_weights = class_counts.sum() - class_counts
    class_weights = class_weights / class_weights.sum()

    # Ensure minimum weight   
    class_weights = torch.clamp(class_weights, min=min_weight)
    
    # Normalize the adjusted weights
    class_weights = class_weights / class_weights.sum()
    
    return class_weights

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='sum'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, class_weights= None):

        p_t = inputs * targets + (1 - inputs) * (1 - targets)  
        if class_weights is None or class_weights.numel() == 0:
            ce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')  
        else:
            ce_loss = F.binary_cross_entropy(inputs, targets, weight=class_weights, reduction='none')  
        loss = ce_loss * ((1 - p_t) ** self.gamma)  

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)  
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()  
        elif self.reduction == 'sum':
            return loss.sum()  
        else:
            return loss



# Criterion class for computing training losses
class Loss:
    def __init__(self, model): 
        self.bce    = nn.BCEWithLogitsLoss(reduction='none')
        self.classif_loss = nn.CrossEntropyLoss()
        self.bce_org    = nn.BCELoss()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.stride = model.stride  # model strides
        self.nc     = model.num_classes  # number of classes
        self.no     = model.no
        self.reg_max = model.reg_max
        self.box_attri   = model.box_attri   #4--xywh ; 6--xyzwhd
        self.input_shape = model.input_shape
        
        self.use_dfl = model.reg_max > 1
        roll_out_thr = 64

        self.assigner = TaskAlignedAssigner(topk=10,
                                            num_classes=self.nc,
                                            alpha=0.5,
                                            beta=6.0,
                                            roll_out_thr=roll_out_thr)
        self.bbox_loss  = BboxLoss(model.reg_max - 1, use_dfl=self.use_dfl)
        self.proj       = torch.arange(model.reg_max, dtype=torch.float)
        self.bbox_util   = DecodeBox(self.nc, (self.input_shape[0], self.input_shape[1]))

    def preprocess(self, targets, batch_size,box_attri, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, box_attri+1, device=targets.device)
        else:

            i           = targets[:, 0]  
            _, counts   = i.unique(return_counts=True)
            out         = torch.zeros(batch_size, counts.max(), box_attri+1, device=targets.device)

            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]

            if box_attri == 4:
                out[..., 1:(box_attri+1)] = xywh2xyxy(out[..., 1:(box_attri+1)].mul_(scale_tensor))
            else:
                out[..., 1:(box_attri+1)] = xyzwhd2xyzxyz(out[..., 1:(box_attri+1)].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist, box_attri):
        if self.use_dfl:
            # batch, anchors, channels
            b, a, c     = pred_dist.shape  

            pred_dist   = pred_dist.view(b, a, box_attri, c // box_attri).softmax(3).matmul(self.proj.to(pred_dist.device).type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)

        return dist2bbox(pred_dist, anchor_points, xywh=False)
    
    def point_decode(self, anchor_points, pred_dist, box_attri):
        if self.use_dfl:
            # batch, anchors, channels
            b, a, c     = pred_dist.shape  

            pred_dist   = pred_dist.view(b, a, box_attri, c // box_attri).softmax(3).matmul(self.proj.to(pred_dist.device).type(pred_dist.dtype))
        return pred_dist    

    def compute_loss(self, preds, batch, mode='RA'):

        device  = preds[1].device
        # box, cls, dfl, cent, obj,points 
        loss    = torch.zeros(9, device=device) #torch.zeros(3, device=device)  

        if mode=='RA':
            feats   = preds[2] if isinstance(preds, tuple) else preds
        else:
            feats   = preds[3] if isinstance(preds, tuple) else preds
        #pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split((self.reg_max * 4, self.nc), 1)
        #refer to yolo.py
        pred_distri, pred_scores, pred_objness, pred_points = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split((self.reg_max * self.box_attri, self.nc, 1, 2), 1)
       
        # bs, num_classes + self.reg_max * 4 , 8400 =>  cls bs, num_classes, 8400; 
        #                                               box bs, self.reg_max * 4, 8400;
        #                                               obj bs, 1,                8400
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_objness = pred_objness.permute(0, 2, 1).contiguous()
        pred_points = pred_points.permute(0, 2, 1).contiguous()


        dtype       = pred_scores.dtype
        batch_size  = pred_scores.shape[0]

        imgsz       = torch.tensor(feats[0].shape[2:], device=device, dtype=dtype) * self.stride[0]  

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        #anchor_points_3D, stride_tensor_3D = make_3Danchors(feats, self.stride, 0.5)


        targets                 = torch.cat((batch[:, 0].view(-1, 1), batch[:, 1].view(-1, 1), batch[:, 2:]), 1)

        # bs, max_boxes_num, 5; 5 is cxywh
        targets                 = self.preprocess(targets.to(device), batch_size,6, scale_tensor=imgsz[[1,1, 0, 1,1, 0]]) #scale_tensor=torch.tensor(self.input_shape)[[0, 1, 2, 0, 1, 2]].to(device)
        # bs, max_boxes_num, 5 => bs, max_boxes_num, 1 ; bs, max_boxes_num, 4
        gt_labels, gt_bboxes_3D    = targets.split((1, 6), 2)  # cls, xyxy
 
        # bs, max_boxes_num
        mask_gt                 = gt_bboxes_3D.sum(2, keepdim=True).gt_(0)
        gt_bboxes               = gt_bboxes_3D[..., [0, 1, 3, 4]]
        # pboxes

        # bs, 8400, 4
        pred_bboxes   = self.bbox_decode(anchor_points, pred_distri, self.box_attri)


        # target_bboxes     bs, 8400, 4
        # target_scores     bs, 8400, 80
        # fg_mask           bs, 8400
        _, target_bboxes, target_scores, fg_mask,target_gt_idx = self.assigner(
            pred_scores.detach(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt
        )
        
        target_bboxes_3D    = gt_bboxes_3D.view(-1, 6)[target_gt_idx]
        #target_bboxes_ra = target_bboxes_3D[..., [0, 1, 3, 4]] # =target_bboxes
        target_bboxes_rd = target_bboxes_3D[..., [0, 2, 3, 5]]
        target_points = target_bboxes_3D[..., [2, 5]]
                
        target_bboxes       /= stride_tensor
        target_scores_sum   = max(target_scores.sum(), 1)
        

        loss[5] = self.smooth_l1_loss(pred_points[fg_mask], (target_points[fg_mask]/imgsz[0]))


        x1 = pred_bboxes[..., 0]
        x2 = pred_bboxes[..., 2]
        y1 = pred_bboxes[..., 1]
        y2 = pred_bboxes[..., 3]
        pred_points = pred_points * imgsz[0] # / stride_tensor
        z1 = pred_points[..., 0]
        z2 = pred_points[..., 1]        
        # Combine the extracted values into a new tensor
        pred_bboxes_rd = torch.stack((x1, z1, x2, z2), dim=-1)
        pred_bboxes_3D = torch.stack((x1, y1, z1, x2, y2, z2), dim=-1)
        target_bboxes_rd[...,[0,2]]       /= stride_tensor
        iou, cent_dist   = bbox_iou(pred_bboxes_rd[fg_mask], target_bboxes_rd[fg_mask], xywh=False, CIoU=True)

        loss[6]    = ((1.0 - iou)).sum() / target_scores_sum * 5.
        loss[7]    =(cent_dist).sum() / target_scores_sum / imgsz[0] * 5.
        #loss[5] += (loss_iou_rd+cent_dist)
        
        # 3D IOU Loss
        target_bboxes_3D[...,[0,1,3,4]]       /= stride_tensor
        iou_3D = torch_iou3d(pred_bboxes_3D[fg_mask], target_bboxes_3D[fg_mask], input_size=imgsz[[1,1, 0]])
        loss[8]    = ((1.0 - iou_3D)).sum() / target_scores_sum * 40.
        

        # loss[4] = self.varifocal_loss(pred_objness, fg_mask, target_labels) / target_scores_sum  # VFL way
        #loss[4] = self.bce(pred_objness, (fg_mask.unsqueeze(-1)).to(dtype)).sum()  # BCE
        #loss[4] = self.bce_org(pred_objness, (fg_mask.unsqueeze(-1)).to(dtype)).sum()  # BCE
        loss[4] = self.focal_loss(pred_objness, (fg_mask.unsqueeze(-1)).to(dtype)).sum()/ max(fg_mask.sum(), 1)


       
        pred_mask = (pred_objness.squeeze(-1)) > 0.5
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        #loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        #loss[1] = self.classif_loss(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        #loss[1] = self.bce_org(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        class_weights = compute_class_weights(target_scores.view(-1, self.nc))
        loss[1] = self.focal_loss(pred_scores, target_scores.to(dtype), class_weights).sum()/ target_scores_sum
        #loss[1] = self.focal_loss(pred_scores, target_scores.to(dtype)).sum()/ target_scores_sum
        #loss[1] = self.focal_loss(pred_scores[fg_mask], target_scores[fg_mask].to(dtype)).sum()/ target_scores_sum
        class_weights = compute_class_weights(target_scores[fg_mask])
        temp_loss1 = self.focal_loss(pred_scores[fg_mask], target_scores[fg_mask].to(dtype), class_weights).sum()/ max(target_scores[fg_mask].sum(), 1)
        # class_weights = compute_class_weights(target_scores[pred_mask])
        # temp_loss2 = self.focal_loss(pred_scores[pred_mask], target_scores[pred_mask].to(dtype), class_weights).sum()/ max(target_scores[pred_mask].sum(), 1)
        # loss[1] = (loss[1] + temp_loss1 + temp_loss2) / 3.0 * 10.0 
        loss[1] = (loss[1] + temp_loss1 ) / 2.0 * 10.0


        

        if fg_mask.sum():
            # # batch, anchors, channels
            # b, a, c     = pred_distri.shape  
            # pred_distri_ra   = pred_distri.view(b, a, self.box_attri, c // self.box_attri)[:,:, [0, 1, 3, 4],:].view(b,a,-1)
            # pred_distri_rd   = pred_distri.view(b, a, self.box_attri, c // self.box_attri)[:,:, [0, 2, 3, 5],:].view(b,a,-1)
            
            loss[0], loss[2], loss[3] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)


        loss[0] *= 7.5  # box gain
        loss[1] *= 15 #7.5  # cls gain
        loss[2] *= 1.5  # dfl gain
        loss[3] *= 0.5  # cent gain
        loss[4] *= 40 #30  # obj gain
        loss[5] *= 80 # points z1z2 gain
        
        print('\n myLoss: ',round(loss[0].item(),3),' ',round(loss[1].item(),3),' ',round(loss[2].item(),3),' ',round(loss[3].item(),3),' ',round(loss[4].item(),3),' ',round(loss[5].item(),3),' ',round(loss.sum().item(),3))
        print('\n RDLoss: ',round(loss[6].item(),3),' ',round(loss[7].item(),3),' ',round(loss[8].item(),3),' ')
        return loss.sum() # loss[:3].sum() # loss(box, cls, dfl, cent, obj) # * batch_size

    def __call__(self, preds, batch):
        # batch_ra = batch[:, [0, 1, 2, 3, 5, 6]]
        # batch_rd = batch[:, [0, 1, 2, 4, 5, 7]]
        # # preds= [dbox, cls, x, x_rd, anchors, strides]
        # loss_ra    = self.compute_loss(preds, batch_ra, mode='RA')
        # loss_rd    = self.compute_loss(preds, batch_rd, mode='RD')
        # loss       = (loss_ra + loss_rd)/2.0
        loss    = self.compute_loss(preds, batch, mode='RA')
        #print('\n myLoss: ',round(loss[0].item(),3),' ',round(loss[1].item(),3),' ',round(loss[2].item(),3),' ',round(loss[3].item(),3),' ',round(loss[4].item(),3),' ',round(loss.sum().item(),3))
        return loss

def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model
    
def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)  #1e-6
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
