"""
将 ROIPredictor 从ROIHeads中解耦出来，通过cfg配置指定要使用的ROIPredictor。

ROIPredictor:
    predictions = forward（box_feature） --> 返回结果组织 scores, proposal_deltas
    losses(predictions,proposals) --> 头部损失计算，

    inference（predictions,proposals） -->

    >>> predictions = box_predictor(box_features)
    >>> if training:
    >>>     losses = box_predictor.losses(predictions, proposals)
    >>> else:
    >>>     pred_instances, _ = box_predictor.inference(predictions, proposals)

"""
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.structures import Boxes
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.modeling.roi_heads import FastRCNNOutputLayers
from fvcore.nn import giou_loss, smooth_l1_loss
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron3.modeling.roi_heads import ROI_PREDICTORS_REGISTRY

# ROI_PREDICTORS_REGISTRY = Registry("ROI_PREDICTORS")
__all__ = ['TripleBranchOutputLayer']


class TripleBranchOutput(object):
    # 仅仅用于组织 Predictor 的损失和记录准确率到日志
    def __init__(self,
                 pred_class_logits,
                 pred_standard_cls_arc_logits,
                 pred_standard_cls_softmax_logits,
                 pred_proposal_deltas,
                 proposals,
                 box2box_transform,
                 arc_softmax_loss_weights=(0.8, 0.2),
                 smooth_l1_beta=0.0,
                 box_reg_loss_type="smooth_l1",
                 box_reg_loss_weight=1.0):
        assert len(arc_softmax_loss_weights) == 2
        self.arc_softmax_loss_weights = arc_softmax_loss_weights
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]

        # pred results
        self.pred_class_logits = pred_class_logits
        self.pred_standard_cls_arc_logits = pred_standard_cls_arc_logits  # 可能为None
        self.pred_standard_cls_softmax_logits = pred_standard_cls_softmax_logits  # 可能为None
        self.pred_proposal_deltas = pred_proposal_deltas

        # loss type
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type
        self.box_reg_loss_weight = box_reg_loss_weight

        # proposals : list[Instances] -> Boxes
        # gt_proposal_deltas/gt_boxes， gt_classes，gt_standards
        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            # cat(..., dim=0) concatenates over all images in the batch
            self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
            assert (
                not self.proposals.tensor.requires_grad
            ), "Proposals should not require gradients!"

            # The following fields should exist only when training.
            if proposals[0].has("gt_boxes"):
                self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
                # self.gt_proposal_deltas = self.box2box_transform.get_deltas(self.proposals.tensor, self.gt_boxes.tensor)
                assert proposals[0].has("gt_classes")
                self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
            if proposals[0].has("gt_standards"):
                self.gt_standards = cat([p.gt_standards for p in proposals], dim=0)
        else:
            self.proposals = Boxes(torch.zeros(0, 4, device=self.pred_proposal_deltas.device))
        self._no_instances = len(proposals) == 0  # no instances found

    def standard_cls_loss(self):
        # 标准分类分支不考虑bg
        loss_dict = {}
        bg_class_ind = self.pred_class_logits.shape[1] - 1
        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        if self._no_instances or torch.nonzero(fg_inds).numel() == 0:
            if self.pred_standard_cls_arc_logits is not None:
                loss_dict['standard_cls_arc_loss'] = 0.0 * self.pred_standard_cls_arc_logits.sum()
            if self.pred_standard_cls_softmax_logits is not None:
                loss_dict['standard_cls_softmax_loss'] = 0.0 * self.pred_standard_cls_softmax_logits.sum()
        else:
            fg_gt_standards = self.gt_standards[fg_inds]
            if self.pred_standard_cls_arc_logits is not None:
                fg_pred_std_cls_arc_logits = self.pred_standard_cls_arc_logits[fg_inds]
                standard_cls_arc_loss = F.cross_entropy(fg_pred_std_cls_arc_logits, fg_gt_standards,
                                                        reduction='mean')
                loss_dict['standard_cls_arc_loss'] = standard_cls_arc_loss
                self._log_accuracy(fg_pred_std_cls_arc_logits, fg_gt_standards, suffix='std_cls_arc_accuracy')
            if self.pred_standard_cls_softmax_logits is not None:
                fg_pred_std_cls_softmax_logits = self.pred_standard_cls_softmax_logits[fg_inds]
                standard_cls_softmax_loss = F.cross_entropy(fg_pred_std_cls_softmax_logits, fg_gt_standards,
                                                            reduction='mean')
                loss_dict['standard_cls_softmax_loss'] = standard_cls_softmax_loss
                self._log_accuracy(fg_pred_std_cls_softmax_logits, fg_gt_standards, suffix='std_cls_soft_accuracy')

        return loss_dict

    def losses(self):
        loss_dict = {}
        # 大分类
        loss_dict['cls_loss'] = self.category_cls_loss()

        # 标准分类
        std_cls_loss_dict = self.standard_cls_loss()
        loss_dict.update(std_cls_loss_dict)

        # box reg
        loss_dict['box_reg_loss'] = self.box_reg_loss()
        return loss_dict

    def category_cls_loss(self):
        # 大分类损失
        if self._no_instances:
            cls_loss = 0.0 * self.pred_class_logits.sum()
        else:
            cls_loss = F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction='mean')
            bg_class_id = self.pred_class_logits.shape[1] - 1
            self._log_accuracy(self.pred_class_logits, self.gt_classes, suffix='cls_accuracy',
                               bg_class_id=bg_class_id)
            # if self.pred_class_logits.numel() > 0:
            #
            # else:
            #     cls_loss = 0.0 * self.pred_class_logits.sum()
        return cls_loss

    def _log_accuracy(self, logits, gt_classes, suffix='cls_accuracy', bg_class_id=None):
        # 仅当bg_class_id不为None时才计算fg_cls_accuracy
        num_instances = gt_classes.numel()
        pred_classes = logits.argmax(dim=1)
        num_accurate = (pred_classes == gt_classes).nonzero().numel()
        storage = get_event_storage()
        if num_instances > 0:
            storage.put_scalar("fast_rcnn/" + suffix, num_accurate / num_instances)
            if bg_class_id is not None:
                fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_id)
                num_fg = fg_inds.nonzero().numel()
                if num_fg > 0:
                    fg_gt_classes = gt_classes[fg_inds]
                    fg_pred_classes = pred_classes[fg_inds]
                    fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()
                    storage.put_scalar('fast_rcnn/fg_' + suffix, fg_num_accurate / num_fg)

    def box_reg_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_proposal_deltas.sum()

        box_dim = self.gt_boxes.tensor.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind))[0]
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        if self.box_reg_loss_type == "smooth_l1":
            gt_proposal_deltas = self.box2box_transform.get_deltas(
                self.proposals.tensor, self.gt_boxes.tensor
            )
            loss_box_reg = smooth_l1_loss(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                gt_proposal_deltas[fg_inds],
                self.smooth_l1_beta,
                reduction="sum",
            )
        elif self.box_reg_loss_type == "giou":
            loss_box_reg = giou_loss(
                self._predict_boxes()[fg_inds[:, None], gt_class_cols],
                self.gt_boxes.tensor[fg_inds],
                reduction="sum",
            )
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg * self.box_reg_loss_weight / self.gt_classes.numel()
        return loss_box_reg

    def _predict_boxes(self):
        """
        Returns:
            Tensor: A Tensors of predicted class-specific or class-agnostic boxes
                for all images in a batch. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        return self.box2box_transform.apply_deltas(self.pred_proposal_deltas, self.proposals.tensor)


@ROI_PREDICTORS_REGISTRY.register()
class TripleBranchOutputLayer(FastRCNNOutputLayers):
    @configurable
    def __init__(self, input_shape, std_category_num, std_cls_loss_type, arc_softmax_loss_weights, **kwargs):
        super(TripleBranchOutputLayer, self).__init__(input_shape, **kwargs)
        self.std_category_num = std_category_num
        self.std_cls_loss_type = std_cls_loss_type
        self.arc_softmax_loss_weights = arc_softmax_loss_weights

        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)

        # 新增加的第三个分类分支： 预测是否标准
        # @Will Lee, 标准预测部分不考虑bg,因为大分类分支已经做了这个工作
        self.standard_cls_score = Linear(input_size, self.std_category_num)
        nn.init.normal_(self.standard_cls_score.weight, std=0.01)
        nn.init.constant_(self.standard_cls_score.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        std_category_num = cfg.MODEL.ROI_HEADS.STD_CATEGORY_NUM
        std_cls_loss_type = cfg.MODEL.ROI_HEADS.STD_CLS_LOSS_TYPE
        arc_softmax_loss_weights = cfg.MODEL.ROI_HEADS.STD_CLS_ARC_SOFTMAX_LOSS_WEIGHTS

        kwargs = FastRCNNOutputLayers.from_config(cfg, input_shape)
        return {
            'input_shape': input_shape,
            'std_category_num': std_category_num,
            'std_cls_loss_type': std_cls_loss_type,
            'arc_softmax_loss_weights': arc_softmax_loss_weights,
            **kwargs
        }

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        cls_logits = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        # TODO 补充
        if self.std_cls_loss_type == 'softmax':
            standard_cls_logits = self.standard_cls_score(x)
            predictions = (cls_logits, standard_cls_logits, proposal_deltas)
        elif self.std_cls_loss_type == 'arc':
            pass
        elif self.std_cls_loss_type == 'arc+softmax':
            pass
        else:
            raise NotImplementedError('标准分类只支持三种损失模式{arc, softmax, arc+softmax}')

        # cls_logits , standard_cls_logits,proposal_deltas
        # cls_logits , standard_cls_arc_logits,standard_cls_softmax_logits,proposal_deltas
        return predictions

    def losses(self, predictions, proposals):
        # TODO 1.
        # proposals: List(Instances)
        if self.std_cls_loss_type == 'softmax':
            pred_class_logits, pred_standard_cls_softmax_logits, pred_proposal_deltas = predictions
            pred_standard_cls_arc_logits = None
        elif self.std_cls_loss_type == 'arc':
            pass
        elif self.std_cls_loss_type == 'arc+softmax':
            pass
        else:
            raise NotImplementedError('标准分类只支持三种损失模式{arc, softmax, arc+softmax}')

        return TripleBranchOutput(pred_class_logits,
                                  pred_standard_cls_arc_logits,
                                  pred_standard_cls_softmax_logits,
                                  pred_proposal_deltas,
                                  proposals,
                                  self.box2box_transform,
                                  arc_softmax_loss_weights=self.arc_softmax_loss_weights,
                                  smooth_l1_beta=self.smooth_l1_beta,
                                  box_reg_loss_type=self.box_reg_loss_type,
                                  box_reg_loss_weight=self.box_reg_loss_weight).losses()

    def inference(self, predictions, proposals):
        """
        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        # TODO 2.
        image_shapes = [x.image_size for x in proposals]

        if self.std_cls_loss_type == 'softmax':
            pred_class_logits, pred_standard_cls_softmax_logits, pred_proposal_deltas = predictions
            cls_scores = self.predict_probs(pred_class_logits, proposals)
            std_cls_softmax_score = self.predict_probs(pred_standard_cls_softmax_logits, proposals)
            boxes = self.predict_boxes(pred_proposal_deltas, proposals)
            # instances: (list[Instances])
            # kept_indices: (list[Tensor])
            instances, kept_indices = fast_rcnn_inference(boxes, cls_scores, image_shapes, self.test_score_thresh,
                                                          self.test_nms_thresh, self.test_topk_per_image)
            for i, indices in enumerate(kept_indices):
                instances[i].pred_standards = torch.argmax(std_cls_softmax_score[i][indices], dim=-1)

            return instances, kept_indices
        elif self.std_cls_loss_type == 'arc':
            pass
        elif self.std_cls_loss_type == 'arc+softmax':
            pass
        else:
            raise NotImplementedError('标准分类只支持三种损失模式{arc, softmax, arc+softmax}')

    def predict_boxes(self, proposal_deltas, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(self, logits, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(logits, dim=-1)
        return probs.split(num_inst_per_image, dim=0)
