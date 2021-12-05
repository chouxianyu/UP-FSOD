# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.modeling.make_layers import make_fc
import pdb

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config, in_channels):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=config.MODEL.RESNETS.RES5_DILATION
        )

        self.pooler = pooler
        self.head = head
        self.out_channels = head.out_channels

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor")
class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.avgpooler = nn.AdaptiveAvgPool2d((resolution, resolution))
        self.fc6c = make_fc(input_size, 512, use_gn)
        self.fc7c = make_fc(512, 512, use_gn)
        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

        self.netup = torch.nn.Sequential(
            torch.nn.Conv2d(256, 24, 3, padding=1) # 输入为C=256通道，输出D=24通道
            )
        ## transformation: 将UP转为CP(conditional prototype)
        self.alpha = torch.nn.Parameter(torch.rand(24, 1))
        self.beta = torch.nn.Parameter(torch.rand(24, 1))

        self.num_cluster = 24 # D=24个prototype
        self.upfc = make_fc(24*256, 512, use_gn)
        self.encode = torch.nn.Conv1d(256, 256, kernel_size=1)

        self.transform = torch.nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv1d(256, 256, kernel_size=1),
            nn.ReLU(inplace=False),
            )

    def UP(self, scene, center):
        # 该函数类似于GeneralizedRCNN中的函数UP()


        ## P
        x = scene # (N, C, w, h)
        N, C, W, H = x.shape[0:]
        x = F.normalize(x, p=2, dim=1)

        ## transformation: 将UP转为CP(conditional prototype)
        centroid = self.alpha * center + self.beta

        ## 3×3卷积 + channel-wise softmax
        ## IMPORTANT: 共有D个CP，所以对P做3×3卷积转为D个channel再做channel-wise softmax，作用是感知P的每个location对每个CP的attention/weight是多少
        soft_assign = self.netup(x) # (N, D, w, h)
        soft_assign = F.softmax(soft_assign, dim=1) # (N, D, w, h)
        soft_assign = soft_assign.view(soft_assign.shape[0], soft_assign.shape[1], -1) # (N, D, w*h)

        
        ## residual
        x_flatten = x.view(N, C, -1) # P: (N, C, w, h) => (N, C, w*h)
        # 把P拷贝D份
        x1 = x_flatten.expand(self.num_cluster, -1, -1, -1).permute(1, 0, 2, 3) # P: (N, C, w*h) => (D, N, C, w*h) => (N, D, C, w*h)
        # 把CP拷贝w*h份
        x2 = centroid.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0) # CP: (D, C) => (w*h, D, C) => (D, C, w*h)
        # IMPORTANT：共有D个CP，对于每个CP，P的每个location都要减去它
        residual = x1 - x2 # (N, D, C, w*h) - (D, C, w*h) => (N, D, C, w*h)
        residual = residual * soft_assign.unsqueeze(2) # 加权(每个location对每个CP的weight): (N, D, C, w*h) * (N, D, 1, w*h) => (N, D, C, w*h)
        up = residual.sum(dim=-1) # sum（这张图片所有location对每个CP的weight）(N, D, C, w*h) => (N, D, C)


        up = F.normalize(up, p=2, dim=2)
        up = up.view(x.size(0), -1) # (N, D, C) => (N, D*C)
        up = F.normalize(up, p=2, dim=1)


        ## FC
        up = self.upfc(up) # (N, D*C) => (N, 2*C)

        return up, centroid

    def forward(self, x, proposals=None, center=None):
        ## IMPORTANT: ROIBoxHead会调用该函数
        if proposals is not None:
            ## RoIALign，得到P
            xs = self.pooler(x, proposals) # P: (N, C, w, h) w==h==s


            ##################################################################################
            ## 使用UP(会转成conditional prototype)对P进行增强
            up, centroid = self.UP(xs, center) # (N, 2*C)   (D, C)

            ## 对P进行FC，得到用于回归和分类的feature
            x = xs.view(xs.size(0), -1) # P: (N, C, w, h) => (N, C*w*h)
            
            xc = F.relu(self.fc6c(x)) # P: (N, C*w*h) => (N, 2*C)
            xcm = self.fc7c(xc) # P: (N, 2*C) => (N, 2*C)
            ## 将P和enhanced P在channel维度上拼接
            xc = torch.cat((xcm, up), dim=1) # (N, 2*C) => (N, 4*C)


            ## 对P进行FC: (N, C*w*h) => (N, representation_size)
            xr = F.relu(self.fc6r(x)) # P: (N, C*w*h) => (N, representation_size)
            xr = F.relu(self.fc7r(xr)) # P: (N, representation_size) => (N, representation_size)





            ##################################################################################
            #### enhance
            ## 对P进行1×1卷积，映射到1个embedding space
            aug = self.encode(xs.view(xs.shape[0], xs.shape[1], -1)) # (N, C, w, h) => (N, C, w*h) => (N, C, w*h)
            ## 对CP进行FC，映射到1个embedding space
            new_center = self.encode(centroid.unsqueeze(0).repeat(xs.shape[0], 1, 1).permute(0,2,1)).permute(0,2,1) # CP: (D, C) => (1, D, C) => (N, D, C) => (N, C, D) => 1×1卷积 => (N, D, C)
            ## 将aug与new_center相乘，然后softamx，得到align
            align = F.softmax(torch.matmul(new_center, aug), dim=1) # (N, C, w*h) × (N, D, C) => (N, D, w*h)
            ## 将align和new_center相乘，得到aug_feature
            aug_feature = torch.matmul(new_center.permute(0,2,1), align) # (N, C, D) × (N, D, w*h) => (N, C, w*h)
            ## 将aug和aug_feature拼接起来
            aug = torch.cat((aug, aug_feature), dim=1) # (N, 2*C, w*h)
            ## 对aug进行transformation，即(1×1卷积+ReLU) * 2；然后再和P相加；再ReLU
            aug = F.relu(self.transform(aug) + xs.view(xs.shape[0], xs.shape[1], -1)) # (N, C, w*h)
            ## reshape: 新的enhanced P
            x = aug.view(xs.size(0), -1) # (N, C, w*h) => (N, C*w*h)


            ### 下面3步和上面对P进行FC的操作类似，得到用于分类的feature
            xc1 = F.relu(self.fc6c(x)) # (N, C*w*h) => (N, 2*C)  注: fc6c上面也用过
            xc1 = self.fc7c(xc1) # (N, 2*C) => (N, 2*C)  注: fc7c上面也用过
            ## 将P和新的enhanced P在channel维度上拼接
            xcm1 = torch.cat((xc1, xcm), dim=1) # (N, 2*C) => (N, 4*C)
            
            ## 三者分别是classifier1用的feature、classifier2用的feature、regressor用的feature
            return xc, xcm1, xr
        else:
            features = []
            for feature in x:
                feature = self.avgpooler(feature)

                up, centroid = self.UP(feature, center)

                feature = feature.view(feature.size(0), -1)
                feature = F.relu(self.fc6c(feature))
                feature = self.fc7c(feature)
                feature = torch.cat((feature, up), dim=1)
                features.append(feature)
            return features


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPNXconv1fcFeatureExtractor")
class FPNXconv1fcFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(FPNXconv1fcFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        conv_head_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM
        num_stacked_convs = cfg.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS
        dilation = cfg.MODEL.ROI_BOX_HEAD.DILATION

        xconvs = []
        for ix in range(num_stacked_convs):
            xconvs.append(
                nn.Conv2d(
                    in_channels,
                    conv_head_dim,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    bias=False if use_gn else True
                )
            )
            in_channels = conv_head_dim
            if use_gn:
                xconvs.append(group_norm(in_channels))
            xconvs.append(nn.ReLU(inplace=True))

        self.add_module("xconvs", nn.Sequential(*xconvs))
        for modules in [self.xconvs,]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if not use_gn:
                        torch.nn.init.constant_(l.bias, 0)

        input_size = conv_head_dim * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.fc6 = make_fc(input_size, representation_size, use_gn=False)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.xconvs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        return x


def make_roi_box_feature_extractor(cfg, in_channels):
    func = registry.ROI_BOX_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
