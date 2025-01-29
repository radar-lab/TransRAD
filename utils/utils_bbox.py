import numpy as np
import torch
from torchvision.ops import nms
import pkg_resources as pkg

def check_version(current: str = "0.0.0",
                  minimum: str = "0.0.0",
                  name: str = "version ",
                  pinned: bool = False) -> bool:
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    return result

TORCH_1_10 = check_version(torch.__version__, '1.10.0')

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w  = feats[i].shape
        sx          = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy          = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx      = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def make_3Danchors(feats, strides, grid_cell_offset=0.5):
    """Generate 3D anchors from features."""
    anchor_points, stride_tensor         = [], []
    # anchor_points_z_y, anchor_points_y_x = [], []
    # stride_tensor_z_y, stride_tensor_y_x = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape  # assuming feats is a list of 5D tensors (batch, channels, depth, height, width)
        sz = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift z
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sz, sy, sx = torch.meshgrid(sz, sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sz, sy, sx)
        
        anchor_points.append(torch.stack((sx, sy, sz), -1).view(-1, 3))
        stride_tensor.append(torch.full((h * w * w, 1), stride, dtype=dtype, device=device))
        
        # anchor_points_z_y.append(torch.stack((sz, sy), -1).view(-1, 2))
        # anchor_points_y_x.append(torch.stack((sy, sx), -1).view(-1, 2))
        
        # stride_tensor_z_y.append(torch.full((d * h * w, 1), stride, dtype=dtype, device=device))
        # stride_tensor_y_x.append(torch.full((d * h * w, 1), stride, dtype=dtype, device=device))
    
    a = torch.cat(anchor_points) #(torch.cat(anchor_points_z_y)
    b = torch.cat(stride_tensor)
    return a, b

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""

    lt, rb  = torch.split(distance, distance.shape[dim]//2, dim)
    x1y1    = anchor_points - lt
    x2y2    = anchor_points + rb
    if xywh:
        c_xy    = (x1y1 + x2y2) / 2
        wh      = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

class DecodeBox():
    def __init__(self, num_classes, input_shape):
        super(DecodeBox, self).__init__()
        self.num_classes    = num_classes
        self.box_attri      = 4
        self.bbox_attrs     = self.box_attri + num_classes
        self.input_shape    = input_shape
        self.reg_max        = 16
        self.no         = num_classes + self.reg_max * self.box_attri + 1 +2  # number of outputs per anchor: classes+BBox+Objness
        
    def decode_box(self, inputs):
        # dbox  batch_size, 4, 8400
        # cls   batch_size, 20, 8400
        dbox, cls, feats, anchors, strides = inputs
        pred_distri, pred_scores, pred_objness = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split((self.reg_max * self.box_attri, self.num_classes, 1), 1)
     
        # 获得中心宽高坐标
        dbox    = dist2bbox(dbox, anchors.unsqueeze(0), xywh=True, dim=1) * strides
        y       = torch.cat((dbox, cls.sigmoid()), 1).permute(0, 2, 1)
        # 进行归一化，到0~1之间
        y[:, :, :4] = y[:, :, :4] / torch.Tensor([self.input_shape[1], self.input_shape[0], self.input_shape[1], self.input_shape[0]]).to(y.device)
        return y

    def decode_box_Lei(self, inputs, mode='RA'):
        # dbox  batch_size, 4, 8400
        # cls   batch_size, 20, 8400  # same with pred_scores
        dbox, cls, feats, anchors, strides = inputs
        pred_distri, pred_scores, pred_objness, pred_points = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split((self.reg_max * self.box_attri, self.num_classes, 1, 2), 1)


        dbox    = dist2bbox(dbox, anchors.unsqueeze(0), xywh=True, dim=1) * strides
        pred_points = pred_points  * self.input_shape[-1]
        z1 = pred_points[:, 0, :]
        z2 = pred_points[:, 1, :]
        d = z2 - z1
        z = (z2 + z1)/2.
        # Convert pred_points to zd format
        zd = torch.stack((z, d), dim=1)  # Shape: [1, 2, 1344]
        # Combine dbox and zd to form xyzwhd
        x = dbox[:, 0, :]
        y = dbox[:, 1, :]
        w = dbox[:, 2, :]
        h = dbox[:, 3, :]
        z = zd[:, 0, :]
        d = zd[:, 1, :]
        xyzwhd = torch.stack((x, y, z, w, h, d), dim=1)  # Shape: [1, 6, 1344]

        y       = torch.cat((xyzwhd, pred_objness, cls), 1).permute(0, 2, 1)
        
     
        #y[:, :, :4] = y[:, :, :4] / torch.Tensor([self.input_shape[1], self.input_shape[0], self.input_shape[1], self.input_shape[0]]).to(y.device)
        return y

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):

        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:

            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset  = (input_shape - new_shape)/2./input_shape
            scale   = input_shape/new_shape

            box_yx  = (box_yx - offset) * scale
            box_hw *= scale

        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):

        box_corner          = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            class_conf, class_pred = torch.max(image_pred[:, 4:4 + num_classes], 1, keepdim=True)


            conf_mask = (class_conf[:, 0] >= conf_thres).squeeze()
            

            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not image_pred.size(0):
                continue

            detections = torch.cat((image_pred[:, :4], class_conf.float(), class_pred.float()), 1)


            unique_labels = detections[:, -1].cpu().unique()

            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            for c in unique_labels:

                detections_class = detections[detections[:, -1] == c]

                keep = nms(
                    detections_class[:, :4],
                    detections_class[:, 4],
                    nms_thres
                )
                max_detections = detections_class[keep]
                

                
                # Add max detections to outputs
                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))
            
            if output[i] is not None:
                output[i]           = output[i].cpu().numpy()
                box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4]    = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np


    def get_anchors_and_decode(input, input_shape, anchors, anchors_mask, num_classes):
        #-----------------------------------------------#
        #   input   batch_size, 3 * (4 + 1 + num_classes), 20, 20
        #-----------------------------------------------#
        batch_size      = input.size(0)
        input_height    = input.size(2)
        input_width     = input.size(3)

        #-----------------------------------------------#
        #   input_shape = [640, 640]  input_height = 20, input_width = 20
        #   640 / 20 = 32
        #   stride_h = stride_w = 32
        #-----------------------------------------------#
        stride_h = input_shape[0] / input_height
        stride_w = input_shape[1] / input_width
        #-------------------------------------------------#
        #   anchor_width, anchor_height / stride_h, stride_w
        #-------------------------------------------------#
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in anchors[anchors_mask[2]]]

        #-----------------------------------------------#
        #   batch_size, 3 * (4 + 1 + num_classes), 20, 20 => 
        #   batch_size, 3, 5 + num_classes, 20, 20  => 
        #   batch_size, 3, 20, 20, 4 + 1 + num_classes
        #-----------------------------------------------#
        prediction = input.view(batch_size, len(anchors_mask[2]),
                                num_classes + 5, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()


        #-----------------------------------------------#
        x = torch.sigmoid(prediction[..., 0])  
        y = torch.sigmoid(prediction[..., 1])

        #-----------------------------------------------#
        w = torch.sigmoid(prediction[..., 2]) 
        h = torch.sigmoid(prediction[..., 3]) 

        #-----------------------------------------------#
        conf        = torch.sigmoid(prediction[..., 4])

        #-----------------------------------------------#
        pred_cls    = torch.sigmoid(prediction[..., 5:])

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor  = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor


        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            batch_size * len(anchors_mask[2]), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            batch_size * len(anchors_mask[2]), 1, 1).view(y.shape).type(FloatTensor)

        #----------------------------------------------------------#
        #   batch_size, 3, 20 * 20 => batch_size, 3, 20, 20
        #   batch_size, 3, 20 * 20 => batch_size, 3, 20, 20
        #----------------------------------------------------------#
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

        #----------------------------------------------------------#
        #   x  0 ~ 1 => 0 ~ 2 => -0.5 ~ 1.5 + grid_x
        #   y  0 ~ 1 => 0 ~ 2 => -0.5 ~ 1.5 + grid_y
        #   w  0 ~ 1 => 0 ~ 2 => 0 ~ 4 * anchor_w
        #   h  0 ~ 1 => 0 ~ 2 => 0 ~ 4 * anchor_h 
        #----------------------------------------------------------#
        pred_boxes          = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0]  = x.data * 2. - 0.5 + grid_x
        pred_boxes[..., 1]  = y.data * 2. - 0.5 + grid_y
        pred_boxes[..., 2]  = (w.data * 2) ** 2 * anchor_w
        pred_boxes[..., 3]  = (h.data * 2) ** 2 * anchor_h

        point_h = 5
        point_w = 5
        
        box_xy          = pred_boxes[..., 0:2].cpu().numpy() * 32
        box_wh          = pred_boxes[..., 2:4].cpu().numpy() * 32
        grid_x          = grid_x.cpu().numpy() * 32
        grid_y          = grid_y.cpu().numpy() * 32
        anchor_w        = anchor_w.cpu().numpy() * 32
        anchor_h        = anchor_h.cpu().numpy() * 32
        
        fig = plt.figure()
        ax  = fig.add_subplot(121)
        from PIL import Image
        img = Image.open("img/street.jpg").resize([640, 640])
        plt.imshow(img, alpha=0.5)
        plt.ylim(-30, 650)
        plt.xlim(-30, 650)
        plt.scatter(grid_x, grid_y)
        plt.scatter(point_h * 32, point_w * 32, c='black')
        plt.gca().invert_yaxis()

        anchor_left = grid_x - anchor_w / 2
        anchor_top  = grid_y - anchor_h / 2
        
        rect1 = plt.Rectangle([anchor_left[0, 0, point_h, point_w],anchor_top[0, 0, point_h, point_w]], \
            anchor_w[0, 0, point_h, point_w],anchor_h[0, 0, point_h, point_w],color="r",fill=False)
        rect2 = plt.Rectangle([anchor_left[0, 1, point_h, point_w],anchor_top[0, 1, point_h, point_w]], \
            anchor_w[0, 1, point_h, point_w],anchor_h[0, 1, point_h, point_w],color="r",fill=False)
        rect3 = plt.Rectangle([anchor_left[0, 2, point_h, point_w],anchor_top[0, 2, point_h, point_w]], \
            anchor_w[0, 2, point_h, point_w],anchor_h[0, 2, point_h, point_w],color="r",fill=False)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        ax  = fig.add_subplot(122)
        plt.imshow(img, alpha=0.5)
        plt.ylim(-30, 650)
        plt.xlim(-30, 650)
        plt.scatter(grid_x, grid_y)
        plt.scatter(point_h * 32, point_w * 32, c='black')
        plt.scatter(box_xy[0, :, point_h, point_w, 0], box_xy[0, :, point_h, point_w, 1], c='r')
        plt.gca().invert_yaxis()

        pre_left    = box_xy[...,0] - box_wh[...,0] / 2
        pre_top     = box_xy[...,1] - box_wh[...,1] / 2

        rect1 = plt.Rectangle([pre_left[0, 0, point_h, point_w], pre_top[0, 0, point_h, point_w]],\
            box_wh[0, 0, point_h, point_w,0], box_wh[0, 0, point_h, point_w,1],color="r",fill=False)
        rect2 = plt.Rectangle([pre_left[0, 1, point_h, point_w], pre_top[0, 1, point_h, point_w]],\
            box_wh[0, 1, point_h, point_w,0], box_wh[0, 1, point_h, point_w,1],color="r",fill=False)
        rect3 = plt.Rectangle([pre_left[0, 2, point_h, point_w], pre_top[0, 2, point_h, point_w]],\
            box_wh[0, 2, point_h, point_w,0], box_wh[0, 2, point_h, point_w,1],color="r",fill=False)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        plt.show()
        #
    feat            = torch.from_numpy(np.random.normal(0.2, 0.5, [4, 255, 20, 20])).float()
    anchors         = np.array([[116, 90], [156, 198], [373, 326], [30,61], [62,45], [59,119], [10,13], [16,30], [33,23]])
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    get_anchors_and_decode(feat, [640, 640], anchors, anchors_mask, 80)
