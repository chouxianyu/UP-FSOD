# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
from torch.nn import functional as F
from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
import pdb

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        self.netup = torch.nn.Sequential(
                torch.nn.Conv2d(256, 24, 3, padding=1)
                )
        self.centroids = torch.nn.Parameter(torch.rand(24, 256))
        self.num_cluster = 24
        self.upfc = torch.nn.Linear(24*256, 256)

        self.transform = torch.nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(inplace=False),
            )

    def UP(self, scene):
        x = scene
        N, C, W, H = x.shape[0:]

        x = F.normalize(x, p=2, dim=1)
        soft_assign = self.netup(x)

        soft_assign = F.softmax(soft_assign, dim=1)
        soft_assign = soft_assign.view(soft_assign.shape[0], soft_assign.shape[1], -1)

        x_flatten = x.view(N, C, -1)

        centroid = self.centroids

        x1 = x_flatten.expand(self.num_cluster, -1, -1, -1).permute(1, 0, 2, 3)
        x2 = centroid.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)

        residual = x1 - x2
        residual = residual * soft_assign.unsqueeze(2)
        up = residual.sum(dim=-1)

        up = F.normalize(up, p=2, dim=2)
        up = up.view(x.size(0), -1)
        up = F.normalize(up, p=2, dim=1)

        up = self.upfc(up).unsqueeze(2).unsqueeze(3).repeat(1,1,W,H)

        return up, centroid

    def forward(self, images, targets=None, closeups=None, closeup_targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        src_features = self.backbone(images.tensors)

        up, centroid = self.UP(src_features[4])
        new_features = torch.cat((src_features[4], up), dim=1)
        new_features = self.transform(new_features)

        up_features = []
        for i in range(len(src_features)-1):
            up_features.append(src_features[i])
        up_features.append(new_features)
        features = tuple(up_features)

        if closeups is not None:
            #Feature layer selection
            closeup_rpn_idx = [0, 1, 2, 3, 4, 4]
            closeup_roi_idx = [0, 0, 0, 1, 2, 3]
            select = [0, 1, 2, 3, 4, 5]
            #manual selection
            closeup_rpn_features = []
            closeup_roi_features = []
            for i in range(len(closeups)):
                if i not in select:
                    continue
                feature_per_level = self.backbone(closeups[i])
                closeup_rpn_features.append(feature_per_level[closeup_rpn_idx[i]])
                closeup_roi_features.append(feature_per_level[closeup_roi_idx[i]])
            if 5 in select:
                closeup_rpn_features[-1] = closeup_rpn_features[-1][:, :, 2: -2, 2: -2]#get 13x 13, use 9 x 9, trivial
        else:
            closeup_rpn_features = None
            closeup_roi_features = None

        proposals, proposal_losses = self.rpn(images, features, targets, closeup_rpn_features)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, centroid, proposals, targets, closeup_roi_features, closeup_targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
