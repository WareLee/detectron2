# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
@Will Lee, 该部分代码是对detectron2中相关代码的重整理和改写，预期达到3个目的：
0.将源detectron2中所有head, 都统一使用Predictor概念替换，在实例化相关Predictor时统一使用build_roi_predictor()
    而非源码中 build_box_head() 、 build_mask_head() ...
1.将Predictor从ROIHeads中解耦出来，独立成一个可通过cfg配置的部分
    ROIHeads 承担的角色:
    a. (in training only) match proposals with ground truth and sample them
    b. crop the regions and extract per-region features using proposals
    c. make per-region predictions with different heads --- 使用build_roi_predictor()构建相关Predictor
    Predictor承担的角色（依赖ROIHeads,由ROIHeads实例化和使用）:
    a. further fit box_features(Input of forward()) by predictor_bone
    b. calculate losses
    c. inference
2.将predictor_bone从ROIHeads中剔除，移动到Predictor中(detectrion2源码中除了box_predictor外，其它诸如mask_predictor等都已经是这么干的了)

"""
import inspect
import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, Instances

from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import build_mask_head, Res5ROIHeads, StandardROIHeads, select_foreground_proposals
from detectron2.modeling import ROI_HEADS_REGISTRY

from .roi_predictors import build_roi_predictor

__all__ = ['Res5ROIHeads2', 'StandardROIHeads2']

logger = logging.getLogger(__name__)


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads2(StandardROIHeads):
    """@Will Lee , detectrion2源码中StandardROIHeads由于历史的原因，把box_head和box_predictor拆分成了两部分，并且box_head耦合在了StandardROIHeads中。
    这里将box_head从StandardROIHeads中剔除，并放入predictor中

    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
            self,
            *,
            box_in_features: List[str],
            box_pooler: ROIPooler,
            box_predictor: nn.Module,
            mask_in_features: Optional[List[str]] = None,
            mask_pooler: Optional[ROIPooler] = None,
            mask_head: Optional[nn.Module] = None,
            keypoint_in_features: Optional[List[str]] = None,
            keypoint_pooler: Optional[ROIPooler] = None,
            keypoint_head: Optional[nn.Module] = None,
            train_on_pred_boxes: bool = False,
            **kwargs
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask head.
                None if not using mask head.
            mask_pooler (ROIPooler): pooler to extra region features for mask head
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_predictor = box_predictor

        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head
        self.keypoint_on = keypoint_in_features is not None
        if self.keypoint_on:
            self.keypoint_in_features = keypoint_in_features
            self.keypoint_pooler = keypoint_pooler
            self.keypoint_head = keypoint_head

        self.train_on_pred_boxes = train_on_pred_boxes

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        box_predictor = build_roi_predictor(cfg, ShapeSpec(channels=in_channels, height=pooler_resolution,
                                                           width=pooler_resolution))
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_predictor": box_predictor,
        }

    def _forward_box(
            self, features: Dict[str, torch.Tensor], proposals: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        # box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads2(Res5ROIHeads):
    # TODO 将res5改成可替换的SENet,且要将其转移到predictor中，而非耦合在Res5ROIHeads中
    """@Will Lee ,detectron2源码中box_predictor固定采用的是FastRCNNOutputLayers，而且是耦合在Res5ROIHeads代码中的，
    这里将所有的predictor从XXXROIHead中解耦出来，以通过cfg配置的形式构建
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

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

        self.box_predictor = build_roi_predictor(cfg, ShapeSpec(channels=out_channels, height=pooler_resolution,
                                                                width=pooler_resolution))

        if self.mask_on:
            self.mask_head = build_mask_head(
                cfg,
                ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
            )

    def forward(self, images, features, proposals, targets=None):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        # @Will Lee 2020/7/22 Res5ROIHeads中直接使用box_feature.mean(dim=[2,3])作为predictor的输入,这里做了变动
        # predictions = self.box_predictor(box_features.mean(dim=[2, 3]))
        predictions = self.box_predictor(box_features)

        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}
