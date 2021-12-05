# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F
from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
import pdb

class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def kl_categorical(self, p_logit, q_logit):
        p = F.softmax(p_logit, dim=-1)
        _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                                  - F.log_softmax(q_logit, dim=-1)), 1)
        return torch.mean(_kl)

    def forward(self, features, proposals, centroid=None, targets=None, closeup_features=None, closeup_targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)


        ### xc, aug, xr分别是classifier1用的feature、classifier2用的feature、regressor用的feature
        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        xc, aug, xr = self.feature_extractor(features, proposals, centroid) # 调用FPN2MLPFeatureExtractor
        
        
        ## classifier1和regressor
        class_logits, box_regression = self.predictor(xc, xr)
        ## classifier2
        aug_logits, _ = self.predictor(aug, xr)

        ## KL loss
        kl_loss = self.kl_categorical(class_logits, aug_logits)
        if closeup_features is not None:
            closeup_xc = self.feature_extractor(closeup_features, center=centroid)
            closeup_logits = self.predictor(closeup_xc)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return xc, result, {}

        closeup_labels = closeup_targets.to(dtype=torch.int64)
        loss_classifier, loss_box_reg, loss_closeup = self.loss_evaluator(
            [class_logits], [box_regression], closeup_logits, closeup_labels
        )
        return (
            xc,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg, loss_closeup=loss_closeup, KL_Loss=kl_loss)
        )


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
