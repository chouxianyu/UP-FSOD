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
            torch.nn.Conv2d(256, 24, 3, padding=1)
            )
        #self.centroids = torch.nn.Parameter(torch.rand(24, 256))
        self.alpha = torch.nn.Parameter(torch.rand(24, 1))
        self.beta = torch.nn.Parameter(torch.rand(24, 1))
        self.num_cluster = 24
        self.upfc = make_fc(24*256, 512, use_gn)
        self.encode = torch.nn.Conv1d(256, 256, kernel_size=1)

        self.transform = torch.nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv1d(256, 256, kernel_size=1),
            nn.ReLU(inplace=False),
            )

    def UP(self, scene, center):
        x = scene
        N, C, W, H = x.shape[0:]

        x = F.normalize(x, p=2, dim=1)
        soft_assign = self.netup(x)

        soft_assign = F.softmax(soft_assign, dim=1)
        soft_assign = soft_assign.view(soft_assign.shape[0], soft_assign.shape[1], -1)

        x_flatten = x.view(N, C, -1)

        #centroid = self.centroids
        centroid = self.alpha * center + self.beta

        x1 = x_flatten.expand(self.num_cluster, -1, -1, -1).permute(1, 0, 2, 3)
        x2 = centroid.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)

        residual = x1 - x2
        residual = residual * soft_assign.unsqueeze(2)
        up = residual.sum(dim=-1)

        up = F.normalize(up, p=2, dim=2)
        up = up.view(x.size(0), -1)
        up = F.normalize(up, p=2, dim=1)

        up = self.upfc(up)

        return up, centroid

    def forward(self, x, proposals=None, center=None):
        if proposals is not None:
            xs = self.pooler(x, proposals)

            up, centroid = self.UP(xs, center)

            x = xs.view(xs.size(0), -1)
            xc = F.relu(self.fc6c(x))
            xcm = self.fc7c(xc)
            xc = torch.cat((xcm, up), dim=1)

            xr = F.relu(self.fc6r(x))
            xr = F.relu(self.fc7r(xr))

            aug = self.encode(xs.view(xs.shape[0], xs.shape[1], -1))
            new_center = self.encode(centroid.unsqueeze(0).repeat(xs.shape[0], 1, 1).permute(0,2,1)).permute(0,2,1)
            align = F.softmax(torch.matmul(new_center, aug), dim=1)

            aug_feature = torch.matmul(new_center.permute(0,2,1), align)
            aug = torch.cat((aug, aug_feature), dim=1)
            aug = F.relu(self.transform(aug) + xs.view(xs.shape[0], xs.shape[1], -1))
            x = aug.view(xs.size(0), -1)
            xc1 = F.relu(self.fc6c(x))
            xc1 = self.fc7c(xc1)
            xcm1 = torch.cat((xc1, xcm), dim=1)
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
