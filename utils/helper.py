import numpy as np
#import tensorflow as torch
import torch
import matplotlib.pyplot as plt
from sklearn import mixture
import math
################ coordinates transformation ################
def cartesianToPolar(x, y):
    """ Cartesian to Polar """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def polarToCartesian(rho, phi):
    """ Polar to Cartesian """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

################ functions of RAD processing ################
def complexTo2Channels(target_array):
    """ transfer complex a + bi to [a, b]"""
    assert target_array.dtype == np.complex64
    ### NOTE: transfer complex to (magnitude) ###
    output_array = getMagnitude(target_array)
    output_array = getLog(output_array)
    return output_array

def getMagnitude(target_array, power_order=2):
    """ get magnitude out of complex number """
    target_array = np.abs(target_array)
    target_array = pow(target_array, power_order)
    return target_array 

def getLog(target_array, scalar=1., log_10=True):
    """ get Log values """
    if log_10:
        return scalar * np.log10(target_array + 1.)
    else:
        return target_array

def getSumDim(target_array, target_axis):
    """ sum up one dimension """
    output = np.sum(target_array, axis=target_axis)
    return output 

def switchCols(target_array, cols):
    """ switch columns """
    assert isinstance(cols, tuple) or isinstance(cols, list)
    assert len(cols) == 2
    assert np.max(cols) <= target_array.shape[-1] - 1
    cols = np.sort(cols)
    output_axes = []
    for i in range(target_array.shape[-1]):
        if i == cols[0]:
            idx = cols[1]
        elif i == cols[1]:
            idx = cols[0]
        else:
            idx = i
        output_axes.append(idx)
    return target_array[..., output_axes]

def switchAxes(target_array, axes):
    """ switch axes """
    assert isinstance(axes, tuple) or isinstance(axes, list)
    assert len(axes) == 2
    assert np.max(axes) <= len(target_array.shape) - 1
    return np.swapaxes(target_array, axes[0], axes[1])

def norm2Image(array):
    """ normalize to image format (uint8) """
    norm_sig = plt.Normalize()
    img = plt.cm.viridis(norm_sig(array))
    img *= 255.
    img = img.astype(np.uint8)
    return img

def toCartesianMask(RA_mask, radar_config, gapfill_interval_num=1):
    """ transfer RA mask to Cartesian mask for plotting """
    output_mask = np.ones([RA_mask.shape[0], RA_mask.shape[0]*2]) * np.amin(RA_mask)
    point_angle_previous = None
    for i in range(RA_mask.shape[0]):
        for j in range(1, RA_mask.shape[1]):
            if RA_mask[i, j] > 0:
                point_range = ((radar_config["range_size"]-1) - i) * \
                                radar_config["range_resolution"]
                point_angle = (j * (2*np.pi/radar_config["azimuth_size"]) - np.pi) / \
                                (2*np.pi*0.5*radar_config["config_frequency"]/ \
                                radar_config["designed_frequency"])
                point_angle_current = np.arcsin(point_angle)
                if point_angle_previous is None:
                    point_angle_previous = point_angle_current
                for point_angle in np.linspace(point_angle_previous, point_angle_current, \
                                                gapfill_interval_num):
                    point_zx = polarToCartesian(point_range, point_angle)
                    new_i = int(output_mask.shape[0] - \
                            np.round(point_zx[0]/radar_config["range_resolution"])-1)
                    new_j = int(np.round((point_zx[1]+50)/radar_config["range_resolution"])-1)
                    output_mask[new_i,new_j] = RA_mask[i, j] 
                point_angle_previous = point_angle_current
    return output_mask

def GaussianModel(pcl):
    """ Get the center and covariance from gaussian model. """
    model = mixture.GaussianMixture(n_components=1, covariance_type='full')
    model.fit(pcl)
    return model.means_[0], model.covariances_[0]

################ ground truth manipulation ################
def boxLocationsToWHD(boxes):
    """ Transfer boxes from [x_min, x_max, y_min, y_max, z_min, z_max] to
    [x_center, y_center, z_center, width, height, depth] """
    new_boxes = np.zeros(boxes.shape)
    if len(boxes.shape) == 2:
        assert boxes.shape[-1] == 6
        new_boxes[:, 0] = np.round((boxes[:, 0] + boxes[:, 1]) / 2)
        new_boxes[:, 1] = np.round((boxes[:, 2] + boxes[:, 3]) / 2)
        new_boxes[:, 2] = np.round((boxes[:, 4] + boxes[:, 5]) / 2)
        new_boxes[:, 3] = np.round(boxes[:, 1] - boxes[:, 0])
        new_boxes[:, 4] = np.round(boxes[:, 3] - boxes[:, 2])
        new_boxes[:, 5] = np.round(boxes[:, 5] - boxes[:, 4])
        return new_boxes
    elif len(boxes.shape) == 1:
        assert boxes.shape[0] == 6
        new_boxes[0] = np.round((boxes[0] + boxes[1]) / 2)
        new_boxes[1] = np.round((boxes[2] + boxes[3]) / 2)
        new_boxes[2] = np.round((boxes[4] + boxes[5]) / 2)
        new_boxes[3] = np.round(boxes[1] - boxes[0])
        new_boxes[4] = np.round(boxes[3] - boxes[2])
        new_boxes[5] = np.round(boxes[5] - boxes[4])
        return new_boxes
    else:
        raise ValueError("Wrong input boxes, please check the input")
  
def iou2d(box_xywh_1, box_xywh_2):
    """ Numpy version of 3D bounding box IOU calculation 
    Args:
        box_xywh_1        ->      box1 [x, y, w, h]
        box_xywh_2        ->      box2 [x, y, w, h]"""
    assert box_xywh_1.shape[-1] == 4
    assert box_xywh_2.shape[-1] == 4
    ### areas of both boxes
    box1_area = box_xywh_1[..., 2] * box_xywh_1[..., 3]
    box2_area = box_xywh_2[..., 2] * box_xywh_2[..., 3]
    ### find the intersection box
    box1_min = box_xywh_1[..., :2] - box_xywh_1[..., 2:] * 0.5
    box1_max = box_xywh_1[..., :2] + box_xywh_1[..., 2:] * 0.5
    box2_min = box_xywh_2[..., :2] - box_xywh_2[..., 2:] * 0.5
    box2_max = box_xywh_2[..., :2] + box_xywh_2[..., 2:] * 0.5

    left_top = np.maximum(box1_min, box2_min)
    bottom_right = np.minimum(box1_max, box2_max)
    ### get intersection area
    intersection = np.maximum(bottom_right - left_top, 0.0)
    intersection_area = intersection[..., 0] * intersection[..., 1]
    ### get union area
    union_area = box1_area + box2_area - intersection_area
    ### get iou
    iou = np.nan_to_num(intersection_area / (union_area + 1e-10))
    return iou

def iou3d(box_xyzwhd_1, box_xyzwhd_2, input_size):
    """ Numpy version of 3D bounding box IOU calculation 
    Args:
        box_xyzwhd_1        ->      box1 [x, y, z, w, h, d]
        box_xyzwhd_2        ->      box2 [x, y, z, w, h, d]"""
    assert box_xyzwhd_1.shape[-1] == 6
    assert box_xyzwhd_2.shape[-1] == 6
    fft_shift_implement = np.array([0, 0, input_size[2]/2])
    ### areas of both boxes
    box1_area = box_xyzwhd_1[..., 3] * box_xyzwhd_1[..., 4] * box_xyzwhd_1[..., 5]
    box2_area = box_xyzwhd_2[..., 3] * box_xyzwhd_2[..., 4] * box_xyzwhd_2[..., 5]
    ### find the intersection box
    box1_min = box_xyzwhd_1[..., :3] + fft_shift_implement - box_xyzwhd_1[..., 3:] * 0.5
    box1_max = box_xyzwhd_1[..., :3] + fft_shift_implement + box_xyzwhd_1[..., 3:] * 0.5
    box2_min = box_xyzwhd_2[..., :3] + fft_shift_implement - box_xyzwhd_2[..., 3:] * 0.5
    box2_max = box_xyzwhd_2[..., :3] + fft_shift_implement + box_xyzwhd_2[..., 3:] * 0.5

    # box1_min = box_xyzwhd_1[..., :3] - box_xyzwhd_1[..., 3:] * 0.5
    # box1_max = box_xyzwhd_1[..., :3] + box_xyzwhd_1[..., 3:] * 0.5
    # box2_min = box_xyzwhd_2[..., :3] - box_xyzwhd_2[..., 3:] * 0.5
    # box2_max = box_xyzwhd_2[..., :3] + box_xyzwhd_2[..., 3:] * 0.5

    left_top = np.maximum(box1_min, box2_min)
    bottom_right = np.minimum(box1_max, box2_max)
    ### get intersection area
    intersection = np.maximum(bottom_right - left_top, 0.0)
    intersection_area = intersection[..., 0] * intersection[..., 1] * intersection[..., 2]
    ### get union area
    union_area = box1_area + box2_area - intersection_area
    ### get iou
    iou = np.nan_to_num(intersection_area / (union_area + 1e-10))
    return iou
 
def giou3d(box_xyzwhd_1, box_xyzwhd_2):
    """ Numpy version of 3D bounding box GIOU (Generalized IOU) calculation 
    Args:
        box_xyzwhd_1        ->      box1 [x, y, z, w, h, d]
        box_xyzwhd_2        ->      box2 [x, y, z, w, h, d]"""
    assert len(box_xyzwhd_1.shape) == 2
    assert len(box_xyzwhd_2.shape) == 2
    assert box_xyzwhd_1.shape[-1] == 6
    assert box_xyzwhd_2.shape[-1] == 6
    ### areas of both boxes
    box1_area = box_xyzwhd_1[:, 3] * box_xyzwhd_1[:, 4] * box_xyzwhd_1[:, 5]
    box2_area = box_xyzwhd_2[:, 3] * box_xyzwhd_2[:, 4] * box_xyzwhd_2[:, 5]
    ### find the intersection box
    box1_min = box_xyzwhd_1[:, :3] - box_xyzwhd_1[:, 3:] * 0.5
    box1_max = box_xyzwhd_1[:, :3] + box_xyzwhd_1[:, 3:] * 0.5
    box2_min = box_xyzwhd_2[:, :3] - box_xyzwhd_2[:, 3:] * 0.5
    box2_max = box_xyzwhd_2[:, :3] + box_xyzwhd_2[:, 3:] * 0.5
    left_top = np.maximum(box1_min, box2_min)
    bottom_right = np.minimum(box1_max, box2_max)
    ### get normal area
    intersection = np.maximum(bottom_right - left_top, 0.0)
    intersection_area = intersection[:, 0] * intersection[:, 1] * intersection[:, 2]
    union_area = box1_area + box2_area - intersection_area
    iou = np.nan_to_num(intersection_area / union_area)
    ### get enclose area
    enclose_left_top = np.minimum(box1_min, box2_min)
    enclose_bottom_right = np.maximum(box1_max, box2_max)
    enclose_section = enclose_bottom_right - enclose_left_top
    enclose_area = enclose_section[:, 0] * enclose_section[:, 1] * enclose_section[:, 2]
    ### get giou
    giou = iou - np.nan_to_num((enclose_area - union_area) / (enclose_area + 1e-10))
    return giou
  
def torch_iou2d(box_xywh_1, box_xywh_2):
    """ 3D bounding box IOU calculation 
    Args:
        box_xywh_1        ->      box1 [x, y, w, h]
        box_xywh_2        ->      box2 [x, y, w, h]"""
    assert box_xywh_1.shape[-1] == 4
    assert box_xywh_2.shape[-1] == 4
    ### areas of both boxes
    box1_area = box_xywh_1[..., 2] * box_xywh_1[..., 3]
    box2_area = box_xywh_2[..., 2] * box_xywh_2[..., 3]
    ### find the intersection box
    box1_min = box_xywh_1[..., :2] - box_xywh_1[..., 2:] * 0.5
    box1_max = box_xywh_1[..., :2] + box_xywh_1[..., 2:] * 0.5
    box2_min = box_xywh_2[..., :2] - box_xywh_2[..., 2:] * 0.5
    box2_max = box_xywh_2[..., :2] + box_xywh_2[..., 2:] * 0.5

    left_top = torch.maximum(box1_min, box2_min)
    bottom_right = torch.minimum(box1_max, box2_max)
    ### get intersection area
    intersection = torch.maximum(bottom_right - left_top, 0.0)
    intersection_area = intersection[..., 0] * intersection[..., 1]
    ### get union area
    union_area = box1_area + box2_area - intersection_area
    ### get iou
    iou = intersection_area / (union_area + 1e-10)
    return iou

def torch_iou3d(box_xyzwhd_1, box_xyzwhd_2, input_size):
    """ Torch version of 3D bounding box IOU calculation 
    Args:
        box_xyzwhd_1        ->      box1 [x, y, z, w, h, d]
        box_xyzwhd_2        ->      box2 [x, y, z, w, h, d]"""
    assert box_xyzwhd_1.shape[-1] == 6
    assert box_xyzwhd_2.shape[-1] == 6
    fft_shift_implement = torch.tensor([0, 0, input_size[2]/2]).to(input_size.device)
    ### areas of both boxes
    box1_area = box_xyzwhd_1[..., 3] * box_xyzwhd_1[..., 4] * box_xyzwhd_1[..., 5]
    box2_area = box_xyzwhd_2[..., 3] * box_xyzwhd_2[..., 4] * box_xyzwhd_2[..., 5]
    ### find the intersection box
    box1_min = box_xyzwhd_1[..., :3] + fft_shift_implement - box_xyzwhd_1[..., 3:] * 0.5
    box1_max = box_xyzwhd_1[..., :3] + fft_shift_implement + box_xyzwhd_1[..., 3:] * 0.5
    box2_min = box_xyzwhd_2[..., :3] + fft_shift_implement - box_xyzwhd_2[..., 3:] * 0.5
    box2_max = box_xyzwhd_2[..., :3] + fft_shift_implement + box_xyzwhd_2[..., 3:] * 0.5

    # box1_min = box_xyzwhd_1[..., :3] - box_xyzwhd_1[..., 3:] * 0.5
    # box1_max = box_xyzwhd_1[..., :3] + box_xyzwhd_1[..., 3:] * 0.5
    # box2_min = box_xyzwhd_2[..., :3] - box_xyzwhd_2[..., 3:] * 0.5
    # box2_max = box_xyzwhd_2[..., :3] + box_xyzwhd_2[..., 3:] * 0.5

    left_top = torch.maximum(box1_min, box2_min)
    bottom_right = torch.minimum(box1_max, box2_max)
    ### get intersection area
    intersection = torch.clamp(bottom_right - left_top, min=0.0)
    intersection_area = intersection[..., 0] * intersection[..., 1] * intersection[..., 2]
    ### get union area
    union_area = box1_area + box2_area - intersection_area
    ### get iou
    iou = intersection_area / torch.clamp(union_area, min=1e-10)
    return iou
 
def torch_giou3d(box_xyzwhd_1, box_xyzwhd_2, input_size):
    """ Tensorflow version of 3D bounding box GIOU (Generalized IOU) calculation 
    Args:
        box_xyzwhd_1        ->      box1 [x, y, z, w, h, d]
        box_xyzwhd_2        ->      box2 [x, y, z, w, h, d]"""
    assert box_xyzwhd_1.shape[-1] == 6
    assert box_xyzwhd_2.shape[-1] == 6
    fft_shift_implement = np.array([0, 0, input_size[2]/2])
    ### areas of both boxes
    box1_area = box_xyzwhd_1[..., 3] * box_xyzwhd_1[..., 4] * box_xyzwhd_1[..., 5]
    box2_area = box_xyzwhd_2[..., 3] * box_xyzwhd_2[..., 4] * box_xyzwhd_2[..., 5]
    ### find the intersection box
    # box1_min = box_xyzwhd_1[..., :3] - box_xyzwhd_1[..., 3:] * 0.5
    # box1_max = box_xyzwhd_1[..., :3] + box_xyzwhd_1[..., 3:] * 0.5
    # box2_min = box_xyzwhd_2[..., :3] - box_xyzwhd_2[..., 3:] * 0.5
    # box2_max = box_xyzwhd_2[..., :3] + box_xyzwhd_2[..., 3:] * 0.5

    box1_min = box_xyzwhd_1[..., :3] + fft_shift_implement - box_xyzwhd_1[..., 3:] * 0.5
    box1_max = box_xyzwhd_1[..., :3] + fft_shift_implement + box_xyzwhd_1[..., 3:] * 0.5
    box2_min = box_xyzwhd_2[..., :3] + fft_shift_implement - box_xyzwhd_2[..., 3:] * 0.5
    box2_max = box_xyzwhd_2[..., :3] + fft_shift_implement + box_xyzwhd_2[..., 3:] * 0.5

    left_top = torch.maximum(box1_min, box2_min)
    bottom_right = torch.minimum(box1_max, box2_max)
    ### get normal area
    intersection = torch.maximum(bottom_right - left_top, 0.0)
    intersection_area = intersection[..., 0] * intersection[..., 1] * intersection[..., 2]
    union_area = box1_area + box2_area - intersection_area
    iou = torch.math.divide_no_nan(intersection_area, union_area)
    ### get enclose area
    enclose_left_top = torch.minimum(box1_min, box2_min)
    enclose_bottom_right = torch.maximum(box1_max, box2_max)
    enclose_section = enclose_bottom_right - enclose_left_top
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1] * enclose_section[..., 2]
    ### get giou
    giou = iou - ((enclose_area - union_area) / (enclose_area + 1e-10))
    return giou

#### Lei
def box_iou_3D(box_xyzwhd_1, box_xyzwhd_2, flag=1, ret_iou = True):
    """ PyTorch version of 3D bounding box IOU calculation by Lei
    Args:
        box_xyzwhd_1        ->      box1 [x, y, z, w, h, d]
        box_xyzwhd_2        ->      box2 [x, y, z, w, h, d]
    Test case:
        box1 = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]], dtype=torch.float32)  # [A, 6]
	    box2 = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8]], dtype=torch.float32)  # [B, 6]
		flag = 1

        iou output shape is [A,B]
    """
    # make input_size as input parameter
    #fft_shift_implement = torch.tensor([0, 0, input_size[2] / 2.0])

    if flag == 1:        
        box_xyzwhd_1 = box_xyzwhd_1.unsqueeze(1) #[4,6]->[4,1,6]
        box_xyzwhd_2 = box_xyzwhd_2.unsqueeze(0) #[9,6]->[1,9,6]
        
    b1_xyz = box_xyzwhd_1[..., :3]
    b1_whd = box_xyzwhd_1[..., 3:]
        
    b2_xyz = box_xyzwhd_2[..., :3]
    b2_whd = box_xyzwhd_2[..., 3:]

    ### areas of both boxes
    box1_area = box_xyzwhd_1[..., 3] * box_xyzwhd_1[..., 4] * box_xyzwhd_1[..., 5]
    box2_area = box_xyzwhd_2[..., 3] * box_xyzwhd_2[..., 4] * box_xyzwhd_2[..., 5]

    ### Calculate the left_top(min) and bottom_right(max) corner coordinates of the real box and the predicted box
    box1_min = b1_xyz  - b1_whd * 0.5
    box1_max = b1_xyz  + b1_whd * 0.5
    box2_min = b2_xyz  - b2_whd * 0.5
    box2_max = b2_xyz  + b2_whd * 0.5

    ### find the intersection box
    left_top = torch.max(box1_min, box2_min) # intersect_mins
    bottom_right = torch.min(box1_max, box2_max) # intersect_maxes
    
    ### get intersection area
    intersection = torch.clamp(bottom_right - left_top, min=0.0)
    intersection_area = intersection[..., 0] * intersection[..., 1] * intersection[..., 2]
    
    ### get union area
    union_area = box1_area + box2_area - intersection_area
    
    ### get iou
    iou = intersection_area / torch.clamp(union_area, min=1e-6)
    if ret_iou:
        return iou
    else:
        return iou, b1_xyz, b2_xyz, b1_whd, b2_whd, box1_min, box2_min, box1_max, box2_max


def box3D_iou_loss(b1, b2, iou_type='ciou'):
    """
    Implemented by Lei
    输入为：
    ----------
    b1: tensor, shape=(batch, anchor_num, feat_w, feat_h, 6), xyzwhd
    b2: tensor, shape=(batch, anchor_num, feat_w, feat_h, 6), xyzwhd

    返回为：
    -------
    out: tensor, shape=(batch, anchor_num, feat_w, feat_h)
    """

    iou, b1_xyz, b2_xyz, b1_whd, b2_whd, b1_mins, b2_mins, b1_maxes, b2_maxes = box_iou_3D(b1, b2, flag=0, ret_iou = False)

    center_whd = b1_xyz - b2_xyz
        
    
    enclose_mins = torch.min(b1_mins, b2_mins)
    enclose_maxes = torch.max(b1_maxes, b2_maxes)
    enclose_whd = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(enclose_maxes))

    if iou_type == 'ciou':

        center_distance = torch.sum(torch.pow(center_whd, 2), axis=-1)

        enclose_diagonal = torch.sum(torch.pow(enclose_whd, 2), axis=-1)
        ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal, min=1e-6)
        
 
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b1_whd[..., 0] / torch.clamp(b1_whd[..., 1], min=1e-6)) - torch.atan(b2_whd[..., 0] / torch.clamp(b2_whd[..., 1], min=1e-6))), 2)
        alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
        

        v1 = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b1_whd[..., 0] / torch.clamp(b1_whd[..., 2], min=1e-6)) - torch.atan(b2_whd[..., 0] / torch.clamp(b2_whd[..., 2], min=1e-6))), 2)
        alpha1 = v1 / torch.clamp((1.0 - iou + v1), min=1e-6)
        
        out = ciou - (alpha * v + alpha1 * v1)
        
    elif iou_type == 'siou':

        sigma = torch.pow(torch.sum(torch.pow(center_whd, 2), axis=-1), 0.5)
        

        sin_alpha_1 = torch.clamp(torch.abs(center_whd[..., 0]) / torch.clamp(sigma, min=1e-6), min=0, max=1)
        sin_alpha_2 = torch.clamp(torch.abs(center_whd[..., 1]) / torch.clamp(sigma, min=1e-6), min=0, max=1)
        sin_alpha_3 = torch.clamp(torch.abs(center_whd[..., 2]) / torch.clamp(sigma, min=1e-6), min=0, max=1)
        

        threshold = pow(2, 0.5) / 2
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
        sin_alpha = torch.where(sin_alpha > threshold, sin_alpha_3, sin_alpha)


        angle_cost = torch.cos(torch.asin(sin_alpha) * 2 - math.pi / 2)
        gamma = 2 - angle_cost

        # Distance cost
        rho_x = (center_whd[..., 0] / torch.clamp(enclose_whd[..., 0], min=1e-6)) ** 2
        rho_y = (center_whd[..., 1] / torch.clamp(enclose_whd[..., 1], min=1e-6)) ** 2
        rho_z = (center_whd[..., 2] / torch.clamp(enclose_whd[..., 2], min=1e-6)) ** 2
        distance_cost = 2 - torch.exp(-gamma * rho_x) - torch.exp(-gamma * rho_y) - torch.exp(-gamma * rho_z)
        
        # Shape cost 
        omiga_w = torch.abs(b1_whd[..., 0] - b2_whd[..., 0]) / torch.clamp(torch.max(b1_whd[..., 0], b2_whd[..., 0]), min=1e-6)
        omiga_h = torch.abs(b1_whd[..., 1] - b2_whd[..., 1]) / torch.clamp(torch.max(b1_whd[..., 1], b2_whd[..., 1]), min=1e-6)
        omiga_d = torch.abs(b1_whd[..., 2] - b2_whd[..., 2]) / torch.clamp(torch.max(b1_whd[..., 2], b2_whd[..., 2]), min=1e-6)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4) + torch.pow(1 - torch.exp(-1 * omiga_d), 4)
        out = iou - 0.5 * (distance_cost + shape_cost)
        
    return out


##########


def smoothOnehot(class_num, hm_classes, smooth_coef=0.01):
    """ Transfer class index to one hot class (smoothed) """
    assert isinstance(class_num, int)
    assert isinstance(hm_classes, int)
    assert class_num < hm_classes
    ### building onehot 
    onehot = np.zeros(hm_classes, dtype=np.float32) 
    onehot[class_num] = 1.
    ### smoothing onehot
    uniform_distribution = np.full(hm_classes, 1.0/hm_classes)
    smooth_onehot = (1-smooth_coef) * onehot + smooth_coef * uniform_distribution
    return smooth_onehot


def smooth2onehot(true_labels, num_classes, smoothing=0.1):
    """
    Convert class labels to smooth one-hot encoding.
    
    Args:
    - true_labels (Tensor): tensor of class labels, shape (batch_size,)
    - num_classes (int): number of classes
    - smoothing (float): smoothing factor
    
    Returns:
    - Tensor: smooth one-hot encoded labels, shape (batch_size, num_classes)
    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), num_classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist

def smooth_onehot(onehot_labels, smoothing=0.1):
    """
    Smooth already one-hot encoded labels.
    
    Args:
    - onehot_labels (Tensor): one-hot encoded labels, shape (batch_size, num_classes)
    - smoothing (float): smoothing factor
    
    Returns:
    - Tensor: smooth one-hot encoded labels, shape (batch_size, num_classes)
    """
    assert 0 <= smoothing < 1
    num_classes = onehot_labels.size(1)
    confidence = 1.0 - smoothing
    smoothing_value = smoothing / (num_classes - 1)
    
    # Apply smoothing
    smoothed_labels = onehot_labels * confidence + (1.0 - onehot_labels) * smoothing_value
    return smoothed_labels

def yoloheadToPredictions(yolohead_output, conf_threshold=0.5):
    """ Transfer YOLO HEAD output to [:, 8], where 8 means
    [x, y, z, w, h, d, score, class_index]"""
    prediction = yolohead_output.numpy().reshape(-1, yolohead_output.shape[-1])
    prediction_class = np.argmax(prediction[:, 7:], axis=-1)
    predictions = np.concatenate([prediction[:, :7], \
                    np.expand_dims(prediction_class, axis=-1)], axis=-1)
    conf_mask = (predictions[:, 6] >= conf_threshold)
    predictions = predictions[conf_mask]
    return predictions

def yoloheadToPredictions2D(yolohead_output, conf_threshold=0.5):
    """ Transfer YOLO HEAD output to [:, 6], where 6 means
    [x, y, w, h, score, class_index]"""
    prediction = yolohead_output.numpy().reshape(-1, yolohead_output.shape[-1])
    prediction_class = np.argmax(prediction[:, 5:], axis=-1)
    predictions = np.concatenate([prediction[:, :5], \
                    np.expand_dims(prediction_class, axis=-1)], axis=-1)
    conf_mask = (predictions[:, 4] >= conf_threshold)
    predictions = predictions[conf_mask]
    return predictions

###### Lei
def NMS_3D(boxes,box_scores,box_confidence, box_class_probs,num_classes,input_size,iou_threshold,confidence= 0.5, sigma=0.3, method='nms'):
    """ bboxes= [x, y, z, w, h, d,score]"""
    """ best_bboxes = [x, y, z, w, h, d,score]"""

    assert method in ['nms', 'soft-nms']
    #-----------------------------------------------------------#
    #   判断得分是否大于score_threshold
    #-----------------------------------------------------------#
    mask             = box_scores >= confidence # mask=[numBBox,num_classes]


    if len(boxes) == 0: 
        best_bboxes = np.zeros([0, 7])#[x, y, z, w, h, d,score]
        best_classes = np.zeros([0, num_classes])
    else:
        best_bboxes  = []
        best_classes = []
        for c in range(num_classes):        
            class_boxes      = boxes[mask[:, c]]
            class_box_scores = box_scores[:, c][mask[:, c]]
            cls_bboxes =  np.concatenate((class_boxes, np.expand_dims(class_box_scores, axis=1)), axis = -1)
            classes          = np.ones_like(class_box_scores, 'int32') * c
            ### NOTE: start looping over boxes to find the best one ###
            while len(cls_bboxes) > 0: 
                max_ind = np.argmax(cls_bboxes[:, 6]) # cls_bboxes[:, 6]==class_box_scores
                best_bbox = cls_bboxes[max_ind]
                
                best_bboxes.append(best_bbox)
                best_classes.append(classes)
                
                ## only keep the bboxes which are not corresponding to the max_ind
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                iou = iou3d(best_bbox[np.newaxis, :6], cls_bboxes[:, :6], \
                            input_size)
                weight = np.ones((len(iou),), dtype=np.float32)
                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0
                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))
                cls_bboxes[:, 6] = cls_bboxes[:, 6] * weight
                score_mask = cls_bboxes[:, 6] > 0.
                cls_bboxes = cls_bboxes[score_mask]
                
        if len(best_bboxes) != 0:
            best_bboxes = np.array(best_bboxes)
            best_classes = np.array(best_classes)
        else:
            best_bboxes = np.zeros([0, 7])
            best_classes = np.zeros([0, num_classes])
    return best_bboxes[:, :6], best_bboxes[:, 6],best_classes
##
def NMS_3D_2(bboxes, input_size, iou_threshold,confidence= 0.5, sigma=0.3, method='nms'):
    """ bboxes= [x, y, z, w, h, d,score]"""
    """ best_bboxes = [x, y, z, w, h, d,score]"""
    assert method in ['nms', 'soft-nms']
    #### yoloheadToPredictions
    prediction_class = np.argmax(bboxes[:, 7:], axis=-1)
    bboxes = np.concatenate([bboxes[:, :7], \
                    np.expand_dims(prediction_class, axis=-1)], axis=-1)
    conf_mask = (bboxes[:, 6] >= confidence)
    bboxes = bboxes[conf_mask]
    ## NMS
    if len(bboxes) == 0:
        best_bboxes = np.zeros([0, 8])
    else:
        all_pred_classes = list(set(bboxes[:, 7]))
        unique_classes = list(np.unique(all_pred_classes))
        best_bboxes = []
        for cls in unique_classes:
            cls_mask = (bboxes[:, 7] == cls)
            cls_bboxes = bboxes[cls_mask]
            ### NOTE: start looping over boxes to find the best one ###
            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[:, 6])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                iou = iou3d(best_bbox[np.newaxis, :6], cls_bboxes[:, :6], \
                            input_size)
                weight = np.ones((len(iou),), dtype=np.float32)
                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0
                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))
                cls_bboxes[:, 6] = cls_bboxes[:, 6] * weight
                score_mask = cls_bboxes[:, 6] > 0.
                cls_bboxes = cls_bboxes[score_mask]
        if len(best_bboxes) != 0:
            best_bboxes = np.array(best_bboxes)
        else:
            best_bboxes = np.zeros([0, 8])
    return best_bboxes

def NMS_3D_Lei(bboxes, input_size, iou_threshold, confidence=0.5, sigma=0.3, method='nms'):
    """
    Perform Non-Maximum Suppression (NMS) or Soft-NMS on 3D bounding boxes for one image.
    
    Args:
    - bboxes (np.array): Array of bounding boxes with shape (N, 7+num_classes).
                         Each bbox has format [x, y, z, w, h, d, confidence, class_scores...].
    - input_size (tuple): Size of the input image/volume.
    - iou_threshold (float): IOU threshold for NMS.
    - confidence (float): Confidence threshold to filter out low-score bboxes.
    - sigma (float): Sigma value for Soft-NMS.
    - method (str): Method for suppression ('nms' or 'soft-nms').
    
    Returns:
    - best_bboxes (np.array): Array of best bounding boxes after NMS/Soft-NMS.
    """
    
    # Ensure the method is either 'nms' or 'soft-nms'
    assert method in ['nms', 'soft-nms']
    
    # Apply objectness confidence threshold to filter out the most possible existed target
    conf_mask = (bboxes[..., 6] >= confidence)
    bboxes = bboxes[conf_mask]
    
    # Select/Assign the class with the highest cls_score for/to each bbox
    cls_scores = bboxes[:, 7:]
    prediction_class = np.argmax(cls_scores, axis=-1)
    
    cls_scores = cls_scores[np.arange(cls_scores.shape[0]), prediction_class]
    
    ## Concatenate the selected class and cls_scores to the bbox array
    #bboxes = np.concatenate([bboxes[:, :7], np.expand_dims(prediction_class, axis=-1), np.expand_dims(cls_scores, axis=-1)], axis=-1)
    
    ## Concatenate the selected class to the bbox array
    bboxes = np.concatenate([bboxes[:, :7], np.expand_dims(prediction_class, axis=-1)], axis=-1)
    ## Replace the column of 'objectness confidence' with ' cls_scores' 
    bboxes[..., 6] = cls_scores
    
    

    
    # If no bboxes remain after filtering, return an empty array
    if len(bboxes) == 0:
        best_bboxes = np.zeros([0, 8])
    else:
        # Get the unique classes present in the filtered bboxes
        all_pred_classes = list(set(bboxes[:, 7]))
        unique_classes = list(np.unique(all_pred_classes))
        
        best_bboxes = []
        
        # Iterate over each unique class
        for cls in unique_classes:
            # Filter bboxes by the current class
            cls_mask = (bboxes[:, 7] == cls)
            cls_bboxes = bboxes[cls_mask]
            
            # Perform NMS/Soft-NMS
            while len(cls_bboxes) > 0:
                # Select the bbox with the highest classification score
                max_ind = np.argmax(cls_bboxes[:, 6])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                
                # Remove the selected bbox from the list
                cls_bboxes = np.concatenate([cls_bboxes[:max_ind], cls_bboxes[max_ind + 1:]])
                
                # Calculate IOU between the selected bbox and remaining bboxes
                iou = iou3d(best_bbox[np.newaxis, :6], cls_bboxes[:, :6], input_size)
                
                # Initialize weights for Soft-NMS
                weight = np.ones((len(iou),), dtype=np.float32)
                
                if method == 'nms':
                    # For NMS, set weight to 0 for bboxes with IOU greater than threshold
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0
                
                if method == 'soft-nms':
                    # For Soft-NMS, apply Gaussian decay to weights
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))
                
                # Update bbox scores with weights
                cls_bboxes[:, 6] = cls_bboxes[:, 6] * weight
                
                # Filter out bboxes with scores <= 0
                score_mask = cls_bboxes[:, 6] > 0.
                cls_bboxes = cls_bboxes[score_mask]
        
        # If best_bboxes is not empty, convert it to a numpy array
        if len(best_bboxes) != 0:
            best_bboxes = np.array(best_bboxes)
        else:
            best_bboxes = np.zeros([0, 8])
    
    return best_bboxes



def NMS_2D_Lei(bboxes, input_size, iou_threshold, confidence=0.5, sigma=0.3, method='nms'):
    """
    Perform Non-Maximum Suppression (NMS) or Soft-NMS on 2D bounding boxes for one image.
    
    Args:
    - bboxes (np.array): Array of bounding boxes with shape (N, 5+num_classes). N is the number of BBOX
                         Each bbox has format [x, y, w, h, score, class_scores...].
    - iou_threshold (float): IOU threshold for NMS.
    - confidence (float): Confidence threshold to filter out low-score bboxes.
    - sigma (float): Sigma value for Soft-NMS.
    - method (str): Method for suppression ('nms' or 'soft-nms').
    
    Returns:
    - best_bboxes (np.array): Array of best bounding boxes after NMS/Soft-NMS.
    """
    
    # Ensure the method is either 'nms' or 'soft-nms'
    assert method in ['nms', 'soft-nms']
    
    # Apply confidence threshold
    conf_mask = (bboxes[..., 4] >= confidence)
    bboxes = bboxes[conf_mask]
    
    # Select the class with the highest score for each bbox
    cls_scores = bboxes[:, 5:]
    prediction_class = np.argmax(cls_scores, axis=-1)
    
    cls_scores = cls_scores[np.arange(cls_scores.shape[0]), prediction_class]
    
    # Concatenate the selected class to the bbox array
    bboxes = np.concatenate([bboxes[..., :5], np.expand_dims(prediction_class, axis=-1)], axis=-1)
    ## Replace the column of 'objectness confidence' with ' cls_scores' 
    bboxes[..., 4] = cls_scores    

    
    # If no bboxes remain after filtering, return an empty array
    if len(bboxes) == 0:
        best_bboxes = np.zeros([0, 6])
    else:
        # Get the unique classes present in the filtered bboxes
        all_pred_classes = list(set(bboxes[..., 5]))
        unique_classes = list(np.unique(all_pred_classes))
        
        best_bboxes = []
        
        # Iterate over each unique class
        for cls in unique_classes:
            # Filter bboxes by the current class
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]
            
            # Perform NMS/Soft-NMS
            while len(cls_bboxes) > 0:
                # Select the bbox with the highest score
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                
                # Remove the selected bbox from the list
                cls_bboxes = np.concatenate([cls_bboxes[:max_ind], cls_bboxes[max_ind + 1:]])
                
                # Calculate IOU between the selected bbox and remaining bboxes
                iou = iou2d(best_bbox[np.newaxis, :4], cls_bboxes[..., :4])
                
                # Initialize weights for Soft-NMS
                weight = np.ones((len(iou),), dtype=np.float32)
                
                if method == 'nms':
                    # For NMS, set weight to 0 for bboxes with IOU greater than threshold
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0
                
                if method == 'soft-nms':
                    # For Soft-NMS, apply Gaussian decay to weights
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))
                
                # Update bbox scores with weights
                cls_bboxes[:, 4] = cls_bboxes[..., 4] * weight
                
                # Filter out bboxes with scores <= 0
                score_mask = cls_bboxes[..., 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]
        
        # If best_bboxes is not empty, convert it to a numpy array
        if len(best_bboxes) != 0:
            best_bboxes = np.array(best_bboxes)
        else:
            best_bboxes = np.zeros([0, 6])
    
    return best_bboxes

def NMS_3D_Lei_overlap(bbox, input_size, iou_threshold):
    """
    Perform Second Non-Maximum Suppression (NMS) based on IoU for different classes with using the First NMS results.
    
    Parameters:
    bbox (numpy.ndarray): Array of bounding boxes with shape (N, 8) where each box is [x, y, z, w, h, d, class_scores, class].
    iou_threshold (float): IoU threshold for suppression.
    
    Returns:
    numpy.ndarray: Array of bounding boxes after NMS.
    """
    if len(bbox) == 0:
        return bbox
    
    # Sort the boxes by class scores in descending order
    bbox = bbox[np.argsort(-bbox[:, 6])]
    
    keep = []
    
    while len(bbox) > 0:
        # Select the box with the highest classification score and remove it from the list
        current_box = bbox[0]
        keep.append(current_box)
        bbox = bbox[1:]
        
        # Compute IoU of the remaining boxes with the current box
        #iou = np.array([compute_iou(current_box, box) for box in bbox])
        iou = iou3d(current_box[np.newaxis, :6], bbox[:, :6], input_size)
        
        # Filter out boxes with ( IoU >= thr and different class )
        keep_indices = [i for i, box in enumerate(bbox) if iou[i] < iou_threshold or box[7] == current_box[7]]
        
        bbox = bbox[keep_indices]
    
    return np.array(keep)
#####

def nms(bboxes, iou_threshold, input_size, sigma=0.3, method='nms'):
    """ Bboxes format [x, y, z, w, h, d, score, class_index] """
    """ Implemented the same way as YOLOv4 """ 
    assert method in ['nms', 'soft-nms']
    if len(bboxes) == 0:
        best_bboxes = np.zeros([0, 8])
    else:
        all_pred_classes = list(set(bboxes[:, 7]))
        unique_classes = list(np.unique(all_pred_classes))
        best_bboxes = []
        for cls in unique_classes:
            cls_mask = (bboxes[:, 7] == cls)
            cls_bboxes = bboxes[cls_mask]
            ### NOTE: start looping over boxes to find the best one ###
            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[:, 6])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                iou = iou3d(best_bbox[np.newaxis, :6], cls_bboxes[:, :6], \
                            input_size)
                weight = np.ones((len(iou),), dtype=np.float32)
                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0
                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))
                cls_bboxes[:, 6] = cls_bboxes[:, 6] * weight
                score_mask = cls_bboxes[:, 6] > 0.
                cls_bboxes = cls_bboxes[score_mask]
        if len(best_bboxes) != 0:
            best_bboxes = np.array(best_bboxes)
        else:
            best_bboxes = np.zeros([0, 8])
    return best_bboxes

def nmsOverClass(bboxes, iou_threshold, input_size, sigma=0.3, method='nms'):
    """ Bboxes format [x, y, z, w, h, d, score, class_index] """
    """ Implemented the same way as YOLOv4 """ 
    assert method in ['nms', 'soft-nms']
    if len(bboxes) == 0:
        best_bboxes = np.zeros([0, 8])
    else:
        best_bboxes = []
        ### NOTE: start looping over boxes to find the best one ###
        while len(bboxes) > 0:
            max_ind = np.argmax(bboxes[:, 6])
            best_bbox = bboxes[max_ind]
            best_bboxes.append(best_bbox)
            bboxes = np.concatenate([bboxes[: max_ind], bboxes[max_ind + 1:]])
            iou = iou3d(best_bbox[np.newaxis, :6], bboxes[:, :6], \
                        input_size)
            weight = np.ones((len(iou),), dtype=np.float32)
            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
            bboxes[:, 6] = bboxes[:, 6] * weight
            score_mask = bboxes[:, 6] > 0.
            bboxes = bboxes[score_mask]
        if len(best_bboxes) != 0:
            best_bboxes = np.array(best_bboxes)
        else:
            best_bboxes = np.zeros([0, 8])
    return best_bboxes

def nms2D(bboxes, iou_threshold, input_size, sigma=0.3, method='nms'):
    """ Bboxes format [x, y, w, h, score, class_index] """
    """ Implemented the same way as YOLOv4 """ 
    assert method in ['nms', 'soft-nms']
    if len(bboxes) == 0:
        best_bboxes = np.zeros([0, 6])
    else:
        all_pred_classes = list(set(bboxes[:, 5]))
        unique_classes = list(np.unique(all_pred_classes))
        best_bboxes = []
        for cls in unique_classes:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]
            ### NOTE: start looping over boxes to find the best one ###
            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], \
                                            cls_bboxes[max_ind + 1:]])
                iou = iou2d(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou),), dtype=np.float32)
                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0
                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))
                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]
        if len(best_bboxes) != 0:
            best_bboxes = np.array(best_bboxes)
        else:
            best_bboxes = np.zeros([0, 6])
    return best_bboxes

def nms2DOverClass(bboxes, iou_threshold, input_size, sigma=0.3, method='nms'):
    """ Bboxes format [x, y, w, h, score, class_index] """
    """ Implemented the same way as YOLOv4 """ 
    assert method in ['nms', 'soft-nms']
    if len(bboxes) == 0:
        best_bboxes = np.zeros([0, 6])
    else:
        best_bboxes = []
        ### NOTE: start looping over boxes to find the best one ###
        while len(bboxes) > 0:
            max_ind = np.argmax(bboxes[:, 4])
            best_bbox = bboxes[max_ind]
            best_bboxes.append(best_bbox)
            bboxes = np.concatenate([bboxes[: max_ind], bboxes[max_ind + 1:]])
            iou = iou2d(best_bbox[np.newaxis, :4], bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)
            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
            bboxes[:, 4] = bboxes[:, 4] * weight
            score_mask = bboxes[:, 4] > 0.
            bboxes = bboxes[score_mask]
        if len(best_bboxes) != 0:
            best_bboxes = np.array(best_bboxes)
        else:
            best_bboxes = np.zeros([0, 6])
    return best_bboxes

