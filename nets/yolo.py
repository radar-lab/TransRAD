import numpy as np
import torch
import torch.nn as nn

from nets.backbone import Backbone, C2f, Conv
from nets.yolo_training import weights_init
from utils.utils_bbox import make_anchors

from RMT_Det import RMT


def fuse_conv_and_bn(conv, bn):
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)


    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))


    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

class DFL(nn.Module):

    # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16, box_attri=4):
        super().__init__()
        self.conv   = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x           = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1     = c1  #c1 = self.reg_max
        self.box_attri = box_attri

    def forward(self, x):

        b, c, a = x.shape

        # bs, 4, self.reg_max, 8400 => bs, self.reg_max, 4, 8400 => b, 4, 8400
        return self.conv(x.view(b, self.box_attri, self.c1, a).transpose(2, 1).softmax(1)).view(b, self.box_attri, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)
        
#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, input_shape, num_classes, phi, pretrained=False):
        super(YoloBody, self).__init__()
        depth_dict          = {'n' : 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.00,}
        width_dict          = {'n' : 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        deep_width_dict     = {'n' : 1.00, 's' : 1.00, 'm' : 0.75, 'l' : 0.50, 'x' : 0.50,}
        dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]

        input_channels      = input_shape[0]  # RA-64; RD-256; AD-256
        base_channels       = int(wid_mul * 64)  # 64
        base_depth          = max(round(dep_mul * 3), 1)  # 3

       
        self.backbone   = RMT(in_chans=input_channels, out_indices=(0, 1, 2, 3),
                             embed_dims=[32, 64, 128, 256], depths=[2, 2, 8, 2], num_heads=[4, 4, 8, 16],
                             init_values=[2, 2, 2, 2], heads_ranges=[4, 4, 6, 6], mlp_ratios=[3, 3, 3, 3], drop_path_rate=0.1, norm_layer=nn.LayerNorm, 
                             patch_norm=True, use_checkpoint=False, chunkwise_recurrents=[True, True, False, False], projection=1024,
                             layerscales=[False, False, False, False], layer_init_values=1e-6, norm_eval=True)

        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        # 1024 * deep_mul + 512, 40, 40 => 512, 40, 40
        self.conv3_for_upsample1    = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, base_channels * 8, base_depth, shortcut=False)
        # 768, 80, 80 => 256, 80, 80
        self.conv3_for_upsample2    = C2f(base_channels * 8 + base_channels * 4, base_channels * 4, base_depth, shortcut=False)
        
        # 256, 80, 80 => 256, 40, 40
        self.down_sample1           = Conv(base_channels * 4, base_channels * 4, 3, 2)
        # 512 + 256, 40, 40 => 512, 40, 40
        self.conv3_for_downsample1  = C2f(base_channels * 8 + base_channels * 4, base_channels * 8, base_depth, shortcut=False)

        # 512, 40, 40 => 512, 20, 20
        self.down_sample2           = Conv(base_channels * 8, base_channels * 8, 3, 2)
        # 1024 * deep_mul + 512, 20, 20 =>  1024 * deep_mul, 20, 20
        self.conv3_for_downsample2  = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, int(base_channels * 16 * deep_mul), base_depth, shortcut=False)

        
        ch              = [base_channels * 4, base_channels * 8, int(base_channels * 16 * deep_mul)]
        self.shape      = None
        self.nl         = len(ch)  # number of output feat layers
        self.stride     = torch.tensor([input_shape[0] / x.shape[-2] for x in self.backbone.forward(torch.zeros(1, input_shape[0], input_shape[0], input_shape[0]))])  # forward
        self.obj_ness   = 1
        self.reg_max    = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.box_attri  = 4   #4--xywh ; 6--xyzwhd 
        self.no         = num_classes + self.reg_max * self.box_attri + self.obj_ness + 2  # number of outputs per anchor: classes+BBox+Objness
        self.num_classes = num_classes
        self.input_shape = input_shape
        
        ## YOLO head
        c2, c3   = max((16, ch[0] // self.box_attri, self.reg_max * self.box_attri)), max(ch[0], num_classes)  # channels

        # Regression head for Bbox: [self.reg_max * 4, 80, 80]
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, self.box_attri * self.reg_max, 1)) for x in ch)
        # Classification head: [num_classes, 80, 80]
        #self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, num_classes, 1), nn.Softmax(dim=1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, num_classes, 1), nn.Sigmoid()) for x in ch)          
        # Objectness head: [1, 80, 80]
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.obj_ness, 1), nn.Sigmoid()) for x in ch)
        # Regression head for two points: [self.reg_max * 2, 80, 80]; [z1,z2]
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, 2, 1), nn.Sigmoid()) for x in ch)
        if not pretrained:
            weights_init(self)
        self.dfl = DFL(self.reg_max, self.box_attri) if self.reg_max > 1 else nn.Identity()


    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        return self
    
    def forward(self, x):
        #  backbone
        feat1, feat2, feat3 = self.backbone.forward(x)
        

        P5 = feat3
        P4 = feat2
        P3 = feat1
        # ######## upsample
        # # 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 40, 40
        P5_upsample = self.upsample(feat3)
        # 1024 * deep_mul, 40, 40 cat 512, 40, 40 => 1024 * deep_mul + 512, 40, 40
        P4          = torch.cat([P5_upsample, feat2], 1)
        # 1024 * deep_mul + 512, 40, 40 => 512, 40, 40
        P4          = self.conv3_for_upsample1(P4)

        # 512, 40, 40 => 512, 80, 80
        P4_upsample = self.upsample(P4)
        # 512, 80, 80 cat 256, 80, 80 => 768, 80, 80
        P3          = torch.cat([P4_upsample, feat1], 1)
        # 768, 80, 80 => 256, 80, 80
        P3          = self.conv3_for_upsample2(P3)
        

        shape = P3.shape  # BCHW
        
        # P3 256, 80, 80 => [self.reg_max * 4, 80, 80] and [num_classes, 80, 80]  and [objectness, 80, 80]
                         #=> [(self.reg_max * 4   + num_classes + 1), 80, 80]
        # P4 512, 40, 40 => num_classes + self.reg_max * 4, 40, 40
        # P5 1024 * deep_mul, 20, 20 => num_classes + self.reg_max * 4, 20, 20
        x = [P3, P4, P5]
        #x_rd = [P3, P4, P5] #range-doppler
        for i in range(self.nl): # Traverse P3, P4, P5, and for each of them do reg(cv2) and cls(cv3)
            #x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)  # 0-batchsize; 1-self.reg_max * 4
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i]), self.cv4[i](x[i]), self.cv5[i](x[i])), 1)  # 0-batchsize; 1-self.reg_max * 4
            #x_rd[i] = torch.cat((self.cv2[i](x_rd[i]), self.cv3[i](x_rd[i]), self.cv4[i](x_rd[i])), 1)  # 0-batchsize; 1-self.reg_max * 4

        if self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        
        # [B, self.no, 6400+1600+400]
        # 8400 = 80*80 + 40*40 + 20*20
        # num_classes + self.reg_max * 4 , 8400 =>  cls [num_classes, 8400]; 
        #                                           box [self.reg_max * 4, 8400]
        #                                           obj [1, 8400]
        box, cls, obj,points        = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * self.box_attri, self.num_classes, self.obj_ness, 2), 1)
        #box, cls        = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.num_classes), 1)
        # origin_cls      = [xi.split((self.reg_max * 4, self.num_classes), 1)[1] for xi in x]
        dbox            = self.dfl(box)

        return dbox, cls, x, self.anchors.to(dbox.device), self.strides.to(dbox.device)