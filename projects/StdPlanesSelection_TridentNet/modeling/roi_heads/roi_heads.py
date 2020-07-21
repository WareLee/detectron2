# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from typing import Dict, List, Optional, Tuple, Union
from detectron2.layers import batched_nms
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.structures import Instances
from detectron3.modeling.roi_heads.roi_heads import Res5ROIHeads2
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
import numpy as np
from detectron2.utils.events import get_event_storage


def merge_branch_instances(instances, num_branch, nms_thresh, topk_per_image):
    """
    Merge detection results from different branches of TridentNet.
    Return detection results by applying non-maximum suppression (NMS) on bounding boxes
    and keep the unsuppressed boxes and other instances (e.g mask) if any.

    Args:
        instances (list[Instances]): A list of N * num_branch instances that store detection
            results. Contain N images and each image has num_branch instances.
        num_branch (int): Number of branches used for merging detection results for each image.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        results: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections after merging results from multiple
            branches.
    """
    if num_branch == 1:
        return instances

    batch_size = len(instances) // num_branch
    results = []
    for i in range(batch_size):
        instance = Instances.cat([instances[i + batch_size * j] for j in range(num_branch)])

        # Apply per-class NMS
        keep = batched_nms(
            instance.pred_boxes.tensor, instance.scores, instance.pred_classes, nms_thresh
        )
        keep = keep[:topk_per_image]
        result = instance[keep]

        results.append(result)

    return results


@ROI_HEADS_REGISTRY.register()
class TridentRes5ROITripleHeads(Res5ROIHeads2):
    """
    The TridentNet ROIHeads in a typical "C4" R-CNN model.
    See :class:`Res5ROIHeads`.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        self.num_branch = cfg.MODEL.TRIDENT.NUM_BRANCH
        self.trident_fast = cfg.MODEL.TRIDENT.TEST_BRANCH_IDX != -1

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`Res5ROIHeads.forward`.
        """
        num_branch = self.num_branch if self.training or not self.trident_fast else 1
        all_targets = targets * num_branch if targets is not None else None
        pred_instances, losses = super().forward(images, features, proposals, all_targets)
        del images, all_targets, targets

        if self.training:
            return pred_instances, losses
        else:
            pred_instances = merge_branch_instances(
                pred_instances,
                num_branch,
                self.box_predictor.test_nms_thresh,
                self.box_predictor.test_topk_per_image,
            )

            return pred_instances, {}

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
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )
            # @Will Lee 应对增加的新字段
            gt_standards = targets_per_image.gt_standards
            gt_standards = gt_standards[matched_idxs]
            # print(gt_standards.size())
            gt_standards = gt_standards[sampled_idxs]
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
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
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
