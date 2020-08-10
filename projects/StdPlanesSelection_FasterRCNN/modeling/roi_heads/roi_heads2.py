# roi_heads2.py专门用来融合全局上下文
from typing import List, Tuple, Dict, Union
import numpy as np
import torch
from torch import nn
from detectron2.layers import cat, ShapeSpec
from detectron2.structures import Instances, Boxes
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads.roi_heads import add_ground_truth_to_proposals, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.modeling.sampling import subsample_labels
from detectron2.config import configurable
from detectron2.modeling.poolers import ROIPooler
from detectron3.modeling.roi_heads import StandardROIHeads2, build_roi_predictor

__all__ = ['TripleStandardROIHeads2']


class FuseContext(nn.Module):
    def __init__(self, input_shape: ShapeSpec, output_size: int = 7):
        super(FuseContext, self).__init__()
        self.astro1 = nn.Conv2d(input_shape.channels, input_shape.channels // 2, kernel_size=(3, 3), stride=2,
                                padding=2,
                                dilation=2)
        self.bn1 = nn.BatchNorm2d(input_shape.channels // 2)
        self.relu1 = nn.ReLU(inplace=True)

        self.astro2 = nn.Conv2d(input_shape.channels // 2, input_shape.channels // 4, kernel_size=(3, 3), stride=2,
                                padding=2,
                                dilation=2)
        self.bn2 = nn.BatchNorm2d(input_shape.channels // 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size)

        nn.init.normal_(self.astro1.weight, std=0.01)
        nn.init.constant_(self.astro1.bias, 0.)
        nn.init.normal_(self.astro2.weight, std=0.01)
        nn.init.constant_(self.astro2.bias, 0.)

    # TODO 后期可以尝试推出list[p2,p3,p4,p5]版本的融合上下文层
    def forward(self, x):
        """
        for example:
        p2 torch.Size([16, 256, 200, 240])
        --> AstroConv(3x3,out=128,s=2,p=2,d=2)
        --> 128x100x120 -->AstroConv(3x3,out=64,s=2,p=2,d=2)
        -->64x50x60-->adaptiveAvgPool2d-->64x7x7

        :param x: Tensor ,shape (N,C,H,W)
        :return: Tensor,shape(N,C//4,ouput_size,ouput_size)
        """
        x = self.astro1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.astro2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        y = self.avg_pool(x)
        return y


@ROI_HEADS_REGISTRY.register()
class TripleStandardROIHeads2(StandardROIHeads2):
    @configurable
    def __init__(self, *, std_num_classes, fuse_context_lys=None, **kwargs):
        self.std_num_classes = std_num_classes
        self.fuse_context_lys = fuse_context_lys
        super(TripleStandardROIHeads2, self).__init__(**kwargs)

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
        if self.fuse_context_lys is not None:
            p2 = features['p2']
            fused = self.fuse_context_lys(p2)  # [N,C,H,W]
            num_boxes = [len(x.proposal_boxes) for x in proposals]
            # TODO 融合全局信息

        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        if self.training:
            predictions = self.box_predictor(box_features, gt_standards=cat([p.gt_standards for p in proposals], dim=0))
        else:
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

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {}
        std_num_classes = cfg.MODEL.ROI_HEADS.STD_NUM_CLS
        is_fuse_context = cfg.MODEL.ROI_HEADS.IS_FUSE_CONTEXT
        fuse_context_lys = None
        if is_fuse_context:
            fuse_output_size = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
            fuse_input_shape = input_shape['p2']
            fuse_context_lys = FuseContext(fuse_input_shape, fuse_output_size)

        ret['std_num_classes'] = std_num_classes
        ret['fuse_context_lys'] = fuse_context_lys
        dic = super().from_config(cfg, input_shape)
        ret.update(dic)
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        is_fuse_context = cfg.MODEL.ROI_HEADS.IS_FUSE_CONTEXT
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
        if is_fuse_context:
            shape = ShapeSpec(channels=(in_channels // 4 + in_channels), height=pooler_resolution,
                              width=pooler_resolution)
        else:
            shape = ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        box_predictor = build_roi_predictor(cfg, shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_predictor": box_predictor,
        }

    @torch.no_grad()
    def label_and_sample_proposals(
            self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes, gt_standards = self._sample_proposals2(
                matched_idxs, matched_labels, targets_per_image.gt_classes, targets_per_image.gt_standards
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.gt_standards = gt_standards

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                # gt_boxes = Boxes(
                #     targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                # )
                gt_boxes = Boxes(
                    torch.zeros(len(sampled_idxs), 4, device=gt_standards.device)
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def _sample_proposals2(
            self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor,
            gt_standards: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
            # gt_standards不考虑背景这一类别
            gt_standards = gt_standards[matched_idxs]
            gt_standards[matched_labels == 0] = self.std_num_classes
            gt_standards[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
            gt_standards = torch.zeros_like(matched_idxs) + self.std_num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs], gt_standards[sampled_idxs]
