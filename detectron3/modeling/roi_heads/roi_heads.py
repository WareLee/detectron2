# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from detectron2.layers import ShapeSpec

from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import build_mask_head, Res5ROIHeads
from detectron2.modeling import ROI_HEADS_REGISTRY

from .roi_predictors import build_roi_predictor

__all__ = ['Res5ROIHeads2']

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
