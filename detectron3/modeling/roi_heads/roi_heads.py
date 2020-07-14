# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.roi_heads import build_box_head
from detectron2.modeling.roi_heads import FastRCNNOutputLayers
from detectron2.modeling.roi_heads import build_keypoint_head
from detectron2.modeling.roi_heads import build_mask_head
from detectron2.modeling.roi_heads import Res5ROIHeads

from .roi_predictors import build_roi_predictor

__all__=['Res5ROIHeads2']

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads2(Res5ROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg)

        # fmt: off
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales = (1.0 / input_shape[self.in_features[0]].stride,)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.mask_on = cfg.MODEL.MASK_ON
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        assert len(self.in_features) == 1

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        # @Will Lee, 2020/7/14, 通过注册和build机制将predictor解耦出去，做成cfg配置的形式
        # self.box_predictor = FastRCNNOutputLayers(
        #     cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        # )
        # 这里直接使用的是 box_feature.mean(dim=[2,3])作为predictor的输入
        self.box_predictor = build_roi_predictor(cfg, ShapeSpec(channels=out_channels, height=1, width=1))

        if self.mask_on:
            self.mask_head = build_mask_head(
                cfg,
                ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
            )
