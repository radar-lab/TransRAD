"""
Created on Thu Apr  4 16:49:47 2024

@author: leicheng
"""
import torch
import torch.nn as nn
from conformer import Block, FCUDown, FCUUp, FCUDown_RMT, FCUUp_RMT, Med_ConvBlock
from RMT import PatchEmbed, PatchMerging, BasicLayer
from timm.models.layers import trunc_normal_, to_2tuple
import math

class PatchEmbed_for_Tran(nn.Module):
    """ Image or Feature map to Patch Embedding adapt from uniformer, without img_size
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, padding=0):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=padding)

    def forward(self, x):
        #B, C, H, W = x.shape
        x = self.proj(x) #[B, 64, 320, 320] --> [B, 768, 80, 80]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) #(B, HW, C) ; can replace x_t = self.trans_patch_conv(x).flatten(2).transpose(1, 2)
        #x = self.norm(x) #(B, HW, C) 
        #x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  #B, C, H, W
        # x = [H/2, W/2]
        return x
    
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding adapt from pvt_v2
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # uses a stride smaller than the patch size to ensure patches overlap
        assert max(patch_size) > stride, "Set larger patch_size than stride"
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W    

class Conv2Trans2Conv(nn.Module):
    """
    Converting and reverse converting between feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, inplanes, outplanes, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1):

        super(Conv2Trans2Conv, self).__init__()
        expansion = 4


        if num_med_block > 0:
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
            self.med_block = nn.ModuleList(self.med_block)

        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)

        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)

        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def forward(self, x_c, x_t, ret_final=False):

        _, _, H, W = x_c.size() # [B,128,160,160] [B,256,80,80] [B,512,40,40]

        x_c2t = self.squeeze_block(x_c, x_t) #[B HW+1 C] [B,1601,768]

        x_t = self.trans_block(x_c2t + x_t) #[B,1601,768]

        if self.num_med_block > 0:
            for m in self.med_block:
                x_c = m(x_c)

        x_t2c = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride) # [B,128,160,160]
        if ret_final:
            x_c = x_c + x_t2c
        else:
            x_c = x_t2c

        return x_c, x_t
    
    
class Conv2RMT2Conv(nn.Module):
    """
    Converting and reverse converting between feature maps for CNN block and patch embeddings for RMT transformer encoder block
    """

    def __init__(self, inplanes, outplanes, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1, rmt_depth=4, init_value=2, heads_range=4, 
                 norm_layer=nn.LayerNorm, chunkwise_recurrent=True, downsample=None, use_checkpoint=False,
                 layerscale=False, layer_init_values=1e-5):

        super(Conv2RMT2Conv, self).__init__()
        expansion = 4


        if num_med_block > 0:
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
            self.med_block = nn.ModuleList(self.med_block)

        self.squeeze_block = FCUDown_RMT(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)

        self.expand_block = FCUUp_RMT(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)

        self.trans_block = BasicLayer(
                            embed_dim=embed_dim,
                            out_dim=embed_dim,
                            depth=rmt_depth,
                            num_heads=num_heads,
                            init_value=init_value,
                            heads_range=heads_range,
                            ffn_dim=int(mlp_ratio * embed_dim),
                            drop_path=drop_path_rate,
                            norm_layer=norm_layer,
                            chunkwise_recurrent=chunkwise_recurrent,
                            downsample= downsample,
                            use_checkpoint=use_checkpoint,
                            layerscale=layerscale,
                            layer_init_values=layer_init_values
                        ) #[B, H, W, C]

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def forward(self, x_c, x_t, ret_final=True):

        _, _, H, W = x_c.size() #[B,128,160,160] [B,256,80,80] [B,512,40,40]

        x_c2t = self.squeeze_block(x_c, x_t) #[B, H, W, C] #[B,160,160,192]  [B,80,80,192] [B,40,40,192]

        x_t = self.trans_block(x_c2t + x_t) #[B, H, W, C] #[B,160,160,192] [B,80,80,192] [B,40,40,192]

        if self.num_med_block > 0:
            for m in self.med_block:
                x_c = m(x_c)

        x_t2c = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride) #[B,128,160,160] [B,256,80,80] [B,512,40,40]
        
        if ret_final:
            x_c = x_c + x_t2c
        else:
            x_c = x_t2c

        return x_c, x_t    
    

def autopad(k, p=None, d=1):  
    # kernel, padding, dilation
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class SiLU(nn.Module):  
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
    
class Conv(nn.Module):
    default_act = SiLU() 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act    = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))    

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c      = int(c2 * e) 
        self.cv1    = Conv(c1, 2 * self.c, 1, 1)
        self.cv2    = Conv((2 + n) * self.c, c2, 1)
        self.m      = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_          = c1 // 2
        self.cv1    = Conv(c1, c_, 1, 1)
        self.cv2    = Conv(c_ * 4, c2, 1, 1)
        self.m      = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class Backbone(nn.Module):
    def __init__(self, input_channels, base_channels, base_depth, deep_mul, phi, pretrained=False, 
                 embed_dims=[96, 192, 384, 384], rmt_depths=[1, 1, 1, 1], tr_depth=5, drop_path_rate=0.5, patch_size=16, tr_channels=[16,32,64,128,256,512,1024], channel_ratio=4
                 , num_heads=[4, 6, 8, 6], mlp_ratios=[3, 4, 3, 3], qkv_bias=False, qk_scale=None, drop_rate=0.3, attn_drop_rate=0.3, num_med_block=0,
                 init_values=[2, 2, 2, 2],heads_ranges=[4, 4, 6, 6],chunkwise_recurrents=[True, True, True, True],
                 norm_layer=nn.LayerNorm, layerscales=[False, False, False, False], use_checkpoints=[False, False, False, False], 
                 layer_init_values=1e-6):
        super().__init__()
        
        tr_channels[-1] = 1024 * deep_mul
        ###########  Transformer
        self.num_features = self.embed_dim = embed_dims[-1]  # num_features for consistency with other models
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[-1]))
        #assert tr_depth % 3 == 0
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, tr_depth)]  # stochastic depth decay rule
        # 0 stage
        trans_dw_stride_0 = patch_size // 2  #4
        # patch size is 16x16, img size is 224, and split into 14x14 patches
        #For tokenization, we compress the feature maps generated by the stem module into 14×14 patch embeddings without overlap, 
        #by a linear projection layer, which is a 4×4 convolution with stride 4.
        #self.trans_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        self.conv_patch_2tran = PatchEmbed_for_Tran(patch_size=trans_dw_stride_0, in_chans=tr_channels[0], embed_dim=embed_dims[-1])
        self.trans_1 = Block(dim=embed_dims[-1], num_heads=num_heads[-1], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             )
        
        
        ###########  Conv2Trans2Conv
        # 1 stage
        base_channel = tr_channels[1]
        stage_1_channel = int(base_channel * channel_ratio)
        trans_dw_stride = trans_dw_stride_0 // 2
        self.c2t2c_block = Conv2Trans2Conv(stage_1_channel, stage_1_channel, 1, dw_stride=trans_dw_stride, embed_dim=embed_dims[-1],
                                num_heads=num_heads[-1], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[0],
                                num_med_block=num_med_block)
        #trans_dw_stride =1
        self.c2R2c_block = Conv2RMT2Conv(stage_1_channel, stage_1_channel, 1, dw_stride=1, embed_dim=embed_dims[1],
                                num_heads=num_heads[-1], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[0],
                                num_med_block=num_med_block, rmt_depth=rmt_depths[0], init_value=init_values[0], 
                                heads_range=heads_ranges[0], layer_init_values=layer_init_values)
        self.conv_patch_2tran_1 = PatchEmbed_for_Tran(patch_size=trans_dw_stride, in_chans=tr_channels[1], embed_dim=embed_dims[-1]) #in_chans=3*tr_channels[1]
        # 2 stage
        base_channel = tr_channels[2]
        stage_2_channel = int(base_channel * channel_ratio)
        trans_dw_stride = trans_dw_stride_0 // 4
        self.c2t2c_block_2 = Conv2Trans2Conv(stage_2_channel, stage_2_channel, 1, dw_stride=trans_dw_stride, embed_dim=embed_dims[-1],
                                num_heads=num_heads[-1], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[0],
                                num_med_block=num_med_block)
        
        self.c2R2c_block_2 = Conv2RMT2Conv(stage_2_channel, stage_2_channel, 1, dw_stride=trans_dw_stride, embed_dim=embed_dims[1],
                                num_heads=num_heads[-1], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[0],
                                num_med_block=num_med_block, rmt_depth=rmt_depths[0], init_value=init_values[0], 
                                heads_range=heads_ranges[0], layer_init_values=layer_init_values)
        # 3 stage
        base_channel = tr_channels[3]
        stage_3_channel = int(base_channel * channel_ratio)
        trans_dw_stride = trans_dw_stride_0 // 8
        self.c2t2c_block_3 = Conv2Trans2Conv(stage_3_channel, stage_3_channel, 1, dw_stride=trans_dw_stride, embed_dim=embed_dims[-1],
                                num_heads=num_heads[-1], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[0],
                                num_med_block=num_med_block)
        
        self.c2R2c_block_3 = Conv2RMT2Conv(stage_3_channel, stage_3_channel, 1, dw_stride=trans_dw_stride, embed_dim=embed_dims[1],
                                num_heads=num_heads[-1], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[0],
                                num_med_block=num_med_block, rmt_depth=rmt_depths[0], init_value=init_values[0], 
                                heads_range=heads_ranges[0], layer_init_values=layer_init_values)
        # 4 stage
        base_channel = tr_channels[4]
        stage_4_channel = int(base_channel * channel_ratio)
        trans_dw_stride = trans_dw_stride_0 // 8
        self.c2t2c_block_4 = Conv2Trans2Conv(stage_4_channel, stage_4_channel, 1, dw_stride=trans_dw_stride, embed_dim=embed_dims[-1],
                                num_heads=num_heads[-1], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[0],
                                num_med_block=num_med_block)
        
        self.c2R2c_block_4 = Conv2RMT2Conv(stage_4_channel, stage_4_channel, 1, dw_stride=trans_dw_stride, embed_dim=embed_dims[1],
                                num_heads=num_heads[-1], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[0],
                                num_med_block=num_med_block, rmt_depth=rmt_depths[0], init_value=init_values[0], 
                                heads_range=heads_ranges[0], layer_init_values=layer_init_values)
        
        ###########  Retentive Transformer
        self.num_layers = len(rmt_depths)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(rmt_depths))]  # stochastic depth decay rule
        # downsample H/2 and W/2
        self.rmt_downsample = PatchMerging(dim=embed_dims[1], out_dim=embed_dims[1])
        # split image into non-overlapping patches: (b h w c)
        self.patch_embed = PatchEmbed(in_chans=input_channels, embed_dim=embed_dims[1]) #embed_dims[-1]
        # build layers
        self.rmt_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):  #[B, H, W, C]
            layer = BasicLayer(
                embed_dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer+1] if (i_layer < self.num_layers - 1) else None,
                depth=rmt_depths[i_layer],
                num_heads=num_heads[i_layer],
                init_value=init_values[i_layer],
                heads_range=heads_ranges[i_layer],
                ffn_dim=int(mlp_ratios[i_layer]*embed_dims[i_layer]),
                drop_path=dpr[sum(rmt_depths[:i_layer]):sum(rmt_depths[:i_layer + 1])],
                norm_layer=norm_layer,
                chunkwise_recurrent=chunkwise_recurrents[i_layer],
                downsample=None, #=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoints[i_layer],
                layerscale=layerscales[i_layer],
                layer_init_values=layer_init_values
            )
            self.rmt_layers.append(layer)     
        
        
        ###########  YOLO Convolution
        #-----------------------------------------------#
        #   3, 640, 640
        #-----------------------------------------------#
        # 3, 640, 640 => 32, 640, 640 => 64, 320, 320
        self.stem = Conv(input_channels, base_channels, 3, 2)
        
        # 64, 320, 320 => 128, 160, 160 => 128, 160, 160
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            C2f(base_channels * 2, base_channels * 2, base_depth, True),
        )
        # 128, 160, 160 => 256, 80, 80 => 256, 80, 80
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
        )
        # 256, 80, 80 => 512, 40, 40 => 512, 40, 40
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),
        )
        # 512, 40, 40 => 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 20, 20
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
            C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5)
        )
        
        if pretrained:
            url = {
                "n" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n_backbone_weights.pth',
                "s" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s_backbone_weights.pth',
                "m" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m_backbone_weights.pth',
                "l" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l_backbone_weights.pth',
                "x" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x_backbone_weights.pth',
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from " + url.split('/')[-1])

    def process_each_view(self, x):
        # gen cls_token
        B = x.shape[0]  # x--> [B,256,256,256]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        #  stem stage for RMT: split image into non-overlapping patches: [B, C, H, W]-->[B, H, W, C]
        x_r = self.patch_embed(x) #[B,256,256,256] --> [B,64,64,192]
        # # get all mid_feats of rmt
        # rmt_outs = []
        # for i in range(self.num_layers):
        #     x_r = self.rmt_layers(x_r) #[B, H, W, C]
        #     rmt_outs.append(x_r.flatten(1, 2) )  # Flattens [B, H, W, C] to [B, HW, C];
        
        # stem stage:  # stem stage [B, 256,256,256] -> [B, 32, 128, 128]
        x = self.stem(x)
        
        # 1st stage for Transformer
        #x_t = self.trans_patch_conv(x).flatten(2).transpose(1, 2)#.flatten(2) convert [B, C, H, W] to [B, C, HW]; [B, HW, C]
        x_t = self.conv_patch_2tran(x) #[B, 32, 128, 128] -> [B, 256, 576]
        x_t = torch.cat([cls_tokens, x_t], dim=1) #[B, 256, 576] -> [B, 257, 576]
        x_t1 = self.trans_1(x_t) #[B, 257, 576]
        
        #  1st stage for CNN
        x = self.dark2(x) # 32, 128, 128  => 64, 64, 64
        
        #  1st stage for RMT
        rmt_layer = self.rmt_layers[1]
        x_r = rmt_layer(x_r)  #[B,64,64,192]
        
        # Fuse x and x_t
        x_t2c, x_t = self.c2t2c_block(x, x_t1) # x_t2c=[64, 64, 64]; x_t=[B,257, 576]
        
        # Fuse x and x_r
        x, x_r = self.c2R2c_block(x, x_r)    #x=[B,64, 64, 64], x_r=[B,64,64,192]    
        x = x + x_t2c
        return x, x_t, x_r, x_t1, cls_tokens

    def forward_3_views(self, x):  #x: [B, View, C, H, W]
        # select each view from multi-view input
        x_ra = x[:, 0, :, :, :]  
        x_rd = x[:, 1, :, :, :]
        x_ad = x[:, 2, :, :, :]
        
        # process each view
        x_ra, x_t_ra, x_r_ra, x_t1_ra, cls_tokens_ra = self.process_each_view(x_ra)
        x_rd, x_t_rd, x_r_rd, x_t1_rd, cls_tokens_rd = self.process_each_view(x_rd)
        x_ad, x_t_ad, x_r_ad, x_t1_ad, cls_tokens_ad = self.process_each_view(x_ad)
        
        # fuse 3 views as stem
        #x_rad = torch.cat((x_ra, x_rd, x_ad), dim=1) # [B, C, H, W]  [B,128,160,160]->[B,3*128,160,160]#多个雷达帧
        x_rad = torch.cat((x_ra, x_rd, x_ad), dim=3) # [B, C, H, W]  [B,128,160,160]->[B,128,160,3*160]
        B, C, H, W = x_rad.size()
        stride = W // H  #3
        kernel_size = W - (H - 1) * stride #3
        x = nn.Conv2d(C, C, kernel_size=(1, kernel_size), stride=(1, stride))(x_rad)# [B,128,160,480]->[B,128,160,160]
        B, C, H, W = x.size()# [B,128,160,160]

        # 1st stage for Trans
        # gen cls_token
        cls_tokens = cls_tokens_ra + cls_tokens_rd + cls_tokens_ad  #[B,1,768]      
        x_t = self.conv_patch_2tran_1(x)   #[B,1601,768]
        x_t = torch.cat([cls_tokens, x_t], dim=1)#[B,1601,768]
        x_t_v0 = self.trans_1(x_t) #[B,1601,768] multi-view-attn
        # Fuse x and x_t0
        x, x_t_v1 = self.c2t2c_block(x, x_t_v0, ret_final=True) #  x=[B,128,160,160]; x_t=[B,1601,768]
        
        # if detect on RA map, for other map, please change the below 3 vars
        x_t1 = x_t1_ra
        x_t  = x_t_ra + x_t_v1
        x_r  = x_r_ra       ##x_r=[B,160,160,192]
        
        #-----------------------------------------------#
        #-----------------------------------------------#
        x = self.dark3(x) #[B,128,160,160]-->[B,256,80,80]
        # Fuse x and x_t
        x_t2c, x_t = self.c2t2c_block_2(x, x_t_v0 + x_t)  #x_t_v0 + x_t1 + x_t   [B,256,80,80] [B,1601,768]
        # Fuse x and x_r
        x_r = self.rmt_downsample(x_r) #[B,160,160,192] -> [B,80,80,192]
        x, x_r = self.c2R2c_block_2(x, x_r) #[B,256,80,80]  [B,80,80,192]
        x = x + x_t2c  #[B,256,80,80] 
        
        feat1 = x

        #-----------------------------------------------#
        x = self.dark4(x)
        # Fuse x and x_t
        x_t2c, x_t = self.c2t2c_block_3(x, x_t_v0 + x_t)   
        # Fuse x and x_r
        x_r = self.rmt_downsample(x_r) #[B,80,80,192] -> [B,40,40,192]
        x, x_r = self.c2R2c_block_3(x, x_r)  #[B,512,40,40]  [B,40,40,192]
        x = x + x_t2c #[B,512,40,40]
          
        feat2 = x

        #-----------------------------------------------#
        x = self.dark5(x) #[B,512,20,20]
        # Fuse x and x_t
        B, C, H, W = x.size()# [B,512,20,20]
        adaptive_pool = nn.AdaptiveAvgPool1d(H*W)    
        cls_tokens_5 = x_t[:, :1]
        x_t = adaptive_pool(((x_t_v0 + x_t)[:, :1]).transpose(1, 2)).transpose(1, 2) #[B,400,768]
        x_t = torch.cat([cls_tokens_5, x_t], dim=1)#[B,401,768]
        x_t2c, x_t = self.c2t2c_block_4(x, x_t)    
        # Fuse x and x_r
        x_r = self.rmt_downsample(x_r) #[B,40,40,192] -> [B,20,20,192]
        x, x_r = self.c2R2c_block_4(x, x_r) 
        x = x + x_t2c  #[B,512,20,20]
        
        feat3 = x
        return feat1, feat2, feat3


    def forward_1_view_RMT(self, x):  #x: [B, View, C, H, W]
      
        # stem: process each view
        x, x_t_ra, x_r_ra, x_t1_ra, cls_tokens_ra = self.process_each_view(x)

        
        # multi-view
        # #x_rad = torch.cat((x_ra, x_ra1, x_ra2), dim=1) # [B, C, H, W]  [B,128,160,160]->[B,3*128,160,160]#多个雷达帧
        # B, C, H, W = x_ra.size()
        # stride = W // H  #3
        # kernel_size = W - (H - 1) * stride #3
        # x = nn.Conv2d(C, C, kernel_size=(1, kernel_size), stride=(1, stride))(x_ra)# [B,128,160,480]->[B,128,160,160]
        # B, C, H, W = x.size()# [B,128,160,160]

        # # 1st stage for Trans
        # # gen cls_token
        # cls_tokens = cls_tokens_ra   #[B,1,768]      
        # x_t = self.conv_patch_2tran_1(x)   #[B,1601,768]
        # x_t = torch.cat([cls_tokens, x_t], dim=1)#[B,1601,768]
        # x_t_v0 = self.trans_1(x_t) #[B,1601,768] multi-view-attn
        # # Fuse x and x_t0
        # x, x_t_v1 = self.c2t2c_block(x, x_t_v0, ret_final=True) #  x=[B,128,160,160]; x_t=[B,1601,768]
        
        # if detect on RA map, for other map, please change the below 3 vars
        # x_t1 = x_t1_ra
        # x_t  = x_t_ra + x_t_v1
        x_r  = x_r_ra       ##x_r=[B,160,160,192]

        #-----------------------------------------------#
        x = self.dark3(x) #[B,128,160,160]-->[B,256,80,80]
        # Fuse x and x_t
        #x_t2c, x_t = self.c2t2c_block_2(x, x_t_v0 + x_t)  #x_t_v0 + x_t1 + x_t   [B,256,80,80] [B,1601,768]
        # Fuse x and x_r
        x_r = self.rmt_downsample(x_r) #[B,160,160,192] -> [B,80,80,192]
        x, x_r = self.c2R2c_block_2(x, x_r) #[B,256,80,80]  [B,80,80,192]
        #x = x + x_t2c  #[B,256,80,80] 
        
        feat1 = x    #[B,128,32,32]

        #-----------------------------------------------#
        x = self.dark4(x)
        # Fuse x and x_t
        #x_t2c, x_t = self.c2t2c_block_3(x, x_t_v0 + x_t)   
        # Fuse x and x_r
        x_r = self.rmt_downsample(x_r) #[B,80,80,192] -> [B,40,40,192]
        x, x_r = self.c2R2c_block_3(x, x_r)  #[B,512,40,40]  [B,40,40,192]
        #x = x + x_t2c #[B,512,40,40]
          
        feat2 = x    #[B,256,16,16]

        #-----------------------------------------------#
        x = self.dark5(x) #[B,512,20,20]
        # Fuse x and x_t
        B, C, H, W = x.size()# [B,512,20,20]
        adaptive_pool = nn.AdaptiveAvgPool1d(H*W)    
        #cls_tokens_5 = x_t[:, :1]
        #x_t = adaptive_pool(((x_t_v0 + x_t)[:, :1]).transpose(1, 2)).transpose(1, 2) #[B,400,768]
        #x_t = torch.cat([cls_tokens_5, x_t], dim=1)#[B,401,768]
        #x_t2c, x_t = self.c2t2c_block_4(x, x_t)    
        # Fuse x and x_r
        x_r = self.rmt_downsample(x_r) #[B,40,40,192] -> [B,20,20,192]
        x, x_r = self.c2R2c_block_4(x, x_r) 
        #x = x + x_t2c  #[B,512,20,20]
        
        feat3 = x    #[B,512,8,8]
        return feat1, feat2, feat3

    def forward_1_view(self, x):  #x: [B, View, C, H, W]
      
        # stem: process each view
        x, x_t_ra, x_r_ra, x_t1_ra, cls_tokens_ra = self.process_each_view(x)

        
        # multi-view
        # #x_rad = torch.cat((x_ra, x_ra1, x_ra2), dim=1) # [B, C, H, W]  [B,128,160,160]->[B,3*128,160,160]#多个雷达帧
        # B, C, H, W = x_ra.size()
        # stride = W // H  #3
        # kernel_size = W - (H - 1) * stride #3
        # x = nn.Conv2d(C, C, kernel_size=(1, kernel_size), stride=(1, stride))(x_ra)# [B,128,160,480]->[B,128,160,160]
        # B, C, H, W = x.size()# [B,128,160,160]

        # 1st stage for Trans
        # gen cls_token
        cls_tokens = cls_tokens_ra   #[B,1,768]      
        x_t = self.conv_patch_2tran_1(x)   #[B,1601,768]
        x_t = torch.cat([cls_tokens, x_t], dim=1)#[B,1601,768]
        x_t_v0 = self.trans_1(x_t) #[B,1601,768] multi-view-attn
        # Fuse x and x_t0
        x, x_t_v1 = self.c2t2c_block(x, x_t_v0, ret_final=True) #  x=[B,128,160,160]; x_t=[B,1601,768]
        
        # if detect on RA map, for other map, please change the below 3 vars
        x_t1 = x_t1_ra
        x_t  = x_t_ra + x_t_v1
        x_r  = x_r_ra       ##x_r=[B,160,160,192]
        

        #-----------------------------------------------#
        x = self.dark3(x) #[B,128,160,160]-->[B,256,80,80]
        # Fuse x and x_t
        x_t2c, x_t = self.c2t2c_block_2(x, x_t_v0 + x_t)  #x_t_v0 + x_t1 + x_t   [B,256,80,80] [B,1601,768]
        # Fuse x and x_r
        x_r = self.rmt_downsample(x_r) #[B,160,160,192] -> [B,80,80,192]
        x, x_r = self.c2R2c_block_2(x, x_r) #[B,256,80,80]  [B,80,80,192]
        x = x + x_t2c  #[B,256,80,80] 
        
        feat1 = x    #[B,128,32,32]

        #-----------------------------------------------#
        x = self.dark4(x)
        # Fuse x and x_t
        x_t2c, x_t = self.c2t2c_block_3(x, x_t_v0 + x_t)   
        # Fuse x and x_r
        x_r = self.rmt_downsample(x_r) #[B,80,80,192] -> [B,40,40,192]
        x, x_r = self.c2R2c_block_3(x, x_r)  #[B,512,40,40]  [B,40,40,192]
        x = x + x_t2c #[B,512,40,40]
          
        feat2 = x    #[B,256,16,16]

        #-----------------------------------------------#
        x = self.dark5(x) #[B,512,20,20]
        # Fuse x and x_t

        B, C, H, W = x.size()# [B,512,20,20]
        adaptive_pool = nn.AdaptiveAvgPool1d(H*W)    
        cls_tokens_5 = x_t[:, :1]
        x_t = adaptive_pool(((x_t_v0 + x_t)[:, :1]).transpose(1, 2)).transpose(1, 2) #[B,400,768]
        x_t = torch.cat([cls_tokens_5, x_t], dim=1)#[B,401,768]
        x_t2c, x_t = self.c2t2c_block_4(x, x_t)    
        # Fuse x and x_r
        x_r = self.rmt_downsample(x_r) #[B,40,40,192] -> [B,20,20,192]
        x, x_r = self.c2R2c_block_4(x, x_r) 
        x = x + x_t2c  #[B,512,20,20]
        
        feat3 = x    #[B,512,8,8]
        return feat1, feat2, feat3
    
    def forward_yolo(self, x):
        x = self.stem(x)
        x = self.dark2(x)

        x = self.dark3(x)
        feat1 = x

        x = self.dark4(x)
        feat2 = x

        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3    

    def forward(self, x):  #x: [B, View, C, H, W]
        ### For one view input
        #feat1, feat2, feat3 = self.forward_1_view(x)
        feat1, feat2, feat3 = self.forward_1_view_RMT(x)
        ### For multi-view input
        #feat1, feat2, feat3 = self.forward_3_views(x)
        ### For YOLO model
        #feat1, feat2, feat3 = self.forward_yolo(x)        
        return feat1, feat2, feat3


