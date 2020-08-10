# StandardROIHeads中分离了 box_head 和box_predictor， 注释中申明的是历史原因，这就是说，把box_head体和box_predictor合并在一起共同构建predictor，才比较符合抽象要求
# Predictor应该只接收box_features,predictor_bone,calc_loss,inference
# ROIHead,负责构建box_features(generate_proposals,ROIpooling), build_predictor
import math
import torch
from torch import nn
from torch.nn import functional as F
from fvcore.nn import giou_loss, smooth_l1_loss
from detectron2.utils.events import get_event_storage
from detectron2.structures import Boxes
from detectron2.layers import ShapeSpec, Linear, cat, nonzero_tuple
from detectron2.config import configurable
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron3.modeling.roi_heads import ROI_PREDICTORS_REGISTRY, StandardFastRCNNOutputLayers

__all__ = ['MlabelStandardFastRCNNOutputLayer']


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # return x * y.expand_as(x)
        return x * y


class ArcLayer(nn.Module):
    def __init__(self, in_features, out_classes, s=30.0, m=0.50, easy_margin=False):
        super(ArcLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_classes, in_features))
        # nn.init.xavier_uniform_(self.weight)
        # nn.init.normal_(self.weight, std=0.01)
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.s = s

    def forward(self, embeddings, labels=None):
        # print(embeddings[0][0:10])
        cosine = F.linear(F.normalize(embeddings, dim=1), F.normalize(self.weight, dim=1))
        # print(cosine[0:3])
        sine = ((1.0 - cosine.pow(2)).clamp(0, 1)).sqrt()
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        if labels is not None:
            one_hot = torch.zeros(cosine.size(), device=cosine.device)
            one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        else:
            output = phi
        # print(output[0:4])
        output = output * self.s

        return output


class ArcSoftLayer(nn.Module):
    def __init__(self, in_features, out_classes, mode='softmax', s=30.0, m=0.50, easy_margin=False):
        super(ArcSoftLayer, self).__init__()
        self.mode = mode
        assert mode in ('arc', 'softmax', 'arc+softmax', 'cross_entropy')
        if 'arc' in mode:
            self.arc_ly = ArcLayer(in_features, out_classes, s, m, easy_margin)
        if 'softmax' in mode or 'cross_entropy' in mode:
            self.soft_ly = Linear(in_features, out_classes)
            nn.init.normal_(self.soft_ly.weight, std=0.01)
            nn.init.constant_(self.soft_ly.bias, 0)

    def forward(self, embeddings, labels=None):
        if 'arc+softmax' == self.mode:
            y1 = self.soft_ly(embeddings)
            y2 = self.arc_ly(embeddings, labels)
        elif 'arc' == self.mode:
            y1 = self.arc_ly(embeddings, labels)
            y2 = y1 * 0.0
        elif 'softmax' == self.mode or 'cross_entropy' == self.mode:
            y1 = self.soft_ly(embeddings)
            y2 = y1 * 0.0
        else:
            raise NotImplementedError('目前仅支持softmax(cross_entropy)、arc、arc+softmax三种模式，暂不支持{}'.format(self.mode, ))

        return y1, y2


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=7, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha: 阿尔法α,类别权重.
                    当α是列表时,为各类别权重；
                    当α为常数时,类别权重为[α, 1-α, 1-α, ....],
                    常用于目标检测算法中抑制背景类,
                    retainnet中设置为0.25
        :param gamma: 伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes: 类别数量
        :param size_average: 损失计算方式,默认取均值
        """
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            # α可以以list方式输入,
            # size:[num_classes] 用于对不同类别精细地赋予权重
            # print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            # print("Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用.".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[-1] += alpha
            self.alpha[0:-1] += (1 - alpha)  # α 最终为 [1-α, 1-α, 1-α, 1-α, ...,α] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds: 预测类别. size:[B,N,C] or [B,C]    分
                别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然也可以使用log_softmax,然后进行exp操作)
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax)

        # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class MlabelStandardFastRCNNOutput(object):
    #  仅仅用于组织 Predictor 的【损失】和记录准确率到【日志】
    def __init__(self, pred_category_score1,
                 pred_category_score2,
                 pred_proposal_deltas,
                 pred_standard_score1,
                 pred_standard_score2,
                 proposals,
                 box2box_transform,
                 std_cls_loss_type='softmax',
                 category_loss_type='cross_entropy',
                 smooth_l1_beta=0.0,
                 box_reg_loss_type="smooth_l1",
                 box_reg_loss_weight=1.0):
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]

        self.std_cls_loss_type = std_cls_loss_type
        # pred results
        self.pred_category_score1 = pred_category_score1
        self.pred_category_score2 = pred_category_score2
        self.pred_proposal_deltas = pred_proposal_deltas
        self.pred_standard_score1 = pred_standard_score1
        self.pred_standard_score2 = pred_standard_score2

        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type
        self.box_reg_loss_weight = box_reg_loss_weight
        self.category_loss_type = category_loss_type

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

    def losses(self):
        if self.category_loss_type in ('arc', 'softmax', 'cross_entropy'):
            category_cls_loss = self.category_cls_loss() * 4.0
        else:
            # softmax_pred   , arc_pred
            category_cls_loss = self.category_cls_loss() * 0.2 + self.category_cls_loss2() * 0.8
            category_cls_loss = category_cls_loss * 4.0

        if self.std_cls_loss_type in ('arc', 'softmax'):
            standard_cls_loss = self.standard_cls_loss()
        else:
            # arc + softmax
            standard_cls_loss = self.standard_cls_loss() * 0.2 + self.standard_cls_loss2() * 0.8
        box_reg_loss = self.box_reg_loss()
        return {'category_cls_loss': category_cls_loss, 'standard_cls_loss': standard_cls_loss,
                'box_reg_loss': box_reg_loss}

    def standard_cls_loss(self):
        # 标准与否的细分类,不考虑bg
        bg_class_id = self.pred_standard_score1.shape[1] - 1
        fg_inds = (self.gt_standards >= 0) & (self.gt_standards < bg_class_id)
        if self._no_instances or torch.nonzero(fg_inds).numel() == 0:
            std_cls_loss = 0.0 * self.pred_standard_score1.sum()
        else:
            fg_gt_standards = self.gt_standards[fg_inds]
            fg_pred_standard_score = self.pred_standard_score1[fg_inds]
            std_cls_loss = F.cross_entropy(fg_pred_standard_score, fg_gt_standards, reduction='mean')
            self._log_accuracy(fg_pred_standard_score, fg_gt_standards, suffix='std_cls_accuracy')
        return std_cls_loss

    def standard_cls_loss2(self):
        # 标准与否的细分类,不考虑bg
        bg_class_id = self.pred_standard_score2.shape[1] - 1
        fg_inds = (self.gt_standards >= 0) & (self.gt_standards < bg_class_id)
        if self._no_instances or torch.nonzero(fg_inds).numel() == 0:
            std_cls_loss = 0.0 * self.pred_standard_score2.sum()
        else:
            fg_gt_standards = self.gt_standards[fg_inds]
            fg_pred_standard_score = self.pred_standard_score2[fg_inds]
            std_cls_loss = F.cross_entropy(fg_pred_standard_score, fg_gt_standards, reduction='mean')
            self._log_accuracy(fg_pred_standard_score, fg_gt_standards, suffix='arc_std_accuracy')
        return std_cls_loss

    def category_cls_loss(self):
        # 大分类损失
        if self._no_instances:
            cls_loss = 0.0 * self.pred_category_score1.sum()
        else:
            bg_class_id = self.pred_category_score1.shape[1] - 1
            cls_loss = FocalLoss(num_classes=bg_class_id + 1)(self.pred_category_score1, self.gt_classes)
            # if self.category_loss_type == 'cross_entropy':
            #     cls_loss = F.cross_entropy(self.pred_category_score, self.gt_classes, reduction='mean')
            # elif self.category_loss_type == 'focal_loss':
            #     cls_loss = FocalLoss(num_classes=bg_class_id + 1)(self.pred_category_score, self.gt_classes)
            # else:
            #     raise NotImplementedError('大分类只支持 cross_entropy 和 focal_loss 两种损失类型')
            self._log_accuracy(self.pred_category_score1, self.gt_classes, suffix='cls_accuracy',
                               bg_class_id=bg_class_id)
        return cls_loss

    def category_cls_loss2(self):
        # 大分类损失
        if self._no_instances:
            cls_loss = 0.0 * self.pred_category_score2.sum()
        else:
            bg_class_id = self.pred_category_score2.shape[1] - 1
            cls_loss = FocalLoss(num_classes=bg_class_id + 1)(self.pred_category_score2, self.gt_classes)
            # if self.category_loss_type == 'cross_entropy':
            #     cls_loss = F.cross_entropy(self.pred_category_score, self.gt_classes, reduction='mean')
            # elif self.category_loss_type == 'focal_loss':
            #     cls_loss = FocalLoss(num_classes=bg_class_id + 1)(self.pred_category_score, self.gt_classes)
            # else:
            #     raise NotImplementedError('大分类只支持 cross_entropy 和 focal_loss 两种损失类型')
            self._log_accuracy(self.pred_category_score2, self.gt_classes, suffix='arc_cls_accuracy',
                               bg_class_id=bg_class_id)
        return cls_loss

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

        bg_class_ind = self.pred_category_score1.shape[1] - 1

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

    def _predict_boxes(self):
        """
        Returns:
            Tensor: A Tensors of predicted class-specific or class-agnostic boxes
                for all images in a batch. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        return self.box2box_transform.apply_deltas(self.pred_proposal_deltas, self.proposals.tensor)


@ROI_PREDICTORS_REGISTRY.register()
class MlabelStandardFastRCNNOutputLayer(nn.Module):
    """Three branches :
    (1) proposal-to-detection box regression deltas
    (2) big classification scores
    (3) standard classification scores
    """

    @configurable
    def __init__(
            self,
            input_shape,
            *,
            standard_cls_bone,
            std_num_classes,
            std_cls_emb_dim,
            box2box_transform,
            num_classes,
            arc_args={},
            test_score_thresh=0.0,
            test_nms_thresh=0.5,
            test_topk_per_image=100,
            category_loss_type='cross_entropy',
            std_cls_loss_type='softmax',
            cls_agnostic_bbox_reg=False,
            smooth_l1_beta=0.0,
            box_reg_loss_type="smooth_l1",
            box_reg_loss_weight=1.0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
            box_reg_loss_weight (float): Weight for box regression loss
        """
        super(MlabelStandardFastRCNNOutputLayer, self).__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        # 大类别分类
        self.category_flatten = Flatten()
        self.category_score = ArcSoftLayer(7 * 7 * 256, num_classes + 1, mode=category_loss_type,
                                           s=arc_args['s'],
                                           m=arc_args['m'],
                                           easy_margin=arc_args['easy_margin'])
        # self.category_score = nn.Sequential(Flatten(), Linear(input_size, num_classes + 1))
        # box回归
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = nn.Sequential(Flatten(), Linear(input_size, num_bbox_reg_classes * box_dim))
        # 细分类
        self.standard_cls_bone = standard_cls_bone
        self.std_cls_score = ArcSoftLayer(std_cls_emb_dim, std_num_classes + 1, mode=std_cls_loss_type, s=arc_args['s'],
                                          m=arc_args['m'],
                                          easy_margin=arc_args['easy_margin'])
        # if std_cls_loss_type == 'softmax':
        #     self.std_cls_score = Linear(std_cls_emb_dim, std_num_classes + 1)
        #     nn.init.normal_(self.std_cls_score.weight, std=0.01)
        #     nn.init.constant_(self.std_cls_score.bias, 0)
        # elif std_cls_loss_type == 'arc':
        #     self.std_cls_score = ArcLayer(std_cls_emb_dim, std_num_classes + 1, s=arc_args['s'], m=arc_args['m'],
        #                                   easy_margin=arc_args['easy_margin'])
        # else:
        #     raise NotImplementedError('目前仅支持softmax、arc两种模式，暂不支持{}'.format(std_cls_loss_type, ))
        for pairs in [self.standard_cls_bone.named_parameters(), self.category_score.named_parameters(),
                      self.bbox_pred.named_parameters()]:
            for name, params in pairs:
                if 'weight' in name:
                    nn.init.normal_(params, std=0.01)
                elif 'bias' in name:
                    nn.init.constant_(params, 0.)
        self.std_cls_loss_type = std_cls_loss_type
        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        self.box_reg_loss_weight = box_reg_loss_weight
        self.std_cls_loss_type = std_cls_loss_type
        self.category_loss_type = category_loss_type

    @classmethod
    def from_config(cls, cfg, input_shape):
        std_cls_loss_type = cfg.MODEL.ROI_HEADS.STD_CLS_LOSS_TYPE
        category_loss_type = cfg.MODEL.ROI_HEADS.CATEGORY_LOSS_TYPE
        arc_args = {}
        # if 'arc' in std_cls_loss_type:
        arc_args['s'] = cfg.MODEL.ROI_HEADS.STD_ARC_LOSS_S
        arc_args['m'] = cfg.MODEL.ROI_HEADS.STD_ARC_LOSS_M
        arc_args['easy_margin'] = cfg.MODEL.ROI_HEADS.STD_ARC_LOSS_EASY_MARGIN

        standard_cls_branch_name = cfg.MODEL.ROI_HEADS.STD_CLS_BRANCH_NAME
        std_num_classes = cfg.MODEL.ROI_HEADS.STD_NUM_CLS
        std_cls_emb_dim = cfg.MODEL.ROI_HEADS.STD_CLS_EMB_DIM
        reduction = cfg.MODEL.ROI_HEADS.REDUCTION
        standard_cls_bone = cls.build_std_bone(standard_cls_branch_name, input_shape, emb_dim=std_cls_emb_dim,
                                               reduction=reduction)
        dic = {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg": cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta": cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh": cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "box_reg_loss_weight": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT,
            # fmt: on
        }
        dic['standard_cls_bone'] = standard_cls_bone
        dic['std_cls_loss_type'] = std_cls_loss_type
        dic['category_loss_type'] = category_loss_type
        dic['std_num_classes'] = std_num_classes
        dic['std_cls_emb_dim'] = std_cls_emb_dim
        dic['arc_args'] = arc_args
        return dic

    @classmethod
    def build_std_bone(cls, standard_cls_branch_name, input_shape, emb_dim=512, reduction=8):
        # 2fc 7*7*256 --> flatten --> 512
        # DWSE 7*7*256--> 7*7*512 --> 4*4*512 --> flatten --> 512
        if standard_cls_branch_name == '2fc':
            std_cls_branch = nn.Sequential(
                Flatten(),
                Linear(input_shape.height * input_shape.width * input_shape.channels, emb_dim),
                nn.ReLU(inplace=True),
                # Linear(emb_dim, std_num_classes),
            )
        elif standard_cls_branch_name == 'DWSE':
            std_cls_branch = nn.Sequential(
                # depthwise conv 膨胀卷积操作（@WillLee期待它可以表现的像MaxPool，关注最重要的信息，减弱其他信息）
                nn.Conv2d(input_shape.channels, input_shape.channels, kernel_size=(3, 3), padding=1,
                          groups=input_shape.channels),
                nn.BatchNorm2d(input_shape.channels),
                nn.ReLU(inplace=True),
                # 通道变换 : 3x3-->1x1升通道
                nn.Conv2d(input_shape.channels, input_shape.channels, (3, 3), padding=1, stride=1),
                nn.BatchNorm2d(input_shape.channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(input_shape.channels, input_shape.channels * 2, (3, 3), padding=1, stride=1),
                nn.BatchNorm2d(input_shape.channels * 2),
                nn.ReLU(inplace=True),
                # SE 模块
                SELayer(input_shape.channels * 2, reduction=reduction),
                nn.ZeroPad2d(padding=(0, 1, 0, 1)),  # ?x7x7 --> ?x8x8
                nn.MaxPool2d((2, 2), stride=2),
                # embedding
                Flatten(),
                Linear(4 * 4 * input_shape.channels * 2, emb_dim),
                nn.ReLU(inplace=True),
                # Linear(emb_dim, std_num_classes)
            )
        elif standard_cls_branch_name == '133ConvSE':
            std_cls_branch = nn.Sequential(
                # 升通道
                nn.Conv2d(input_shape.channels, input_shape.channels, (1, 1)),
                nn.BatchNorm2d(input_shape.channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(input_shape.channels, input_shape.channels, (3, 3), padding=1, stride=1),
                nn.BatchNorm2d(input_shape.channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(input_shape.channels, input_shape.channels * 2, (3, 3), padding=1, stride=1),
                nn.BatchNorm2d(input_shape.channels * 2),
                nn.ReLU(inplace=True),

                # SE 模块
                SELayer(input_shape.channels * 2, reduction=reduction),
                # MaxPool降低分辨率
                nn.ZeroPad2d(padding=(0, 1, 0, 1)),  # ?x7x7 --> ?x8x8
                nn.MaxPool2d((2, 2), stride=2),
                # embedding
                Flatten(),
                nn.Linear(4 * 4 * input_shape.channels * 2, emb_dim),
                nn.ReLU(inplace=True)
            )
        else:
            raise NotImplementedError('目前标准分类分支网络构建，仅支持2fc、DWSE、133ConvSE三种')

        return std_cls_branch

    def forward(self, x, gt_standards=None):
        category_score1, category_score2 = self.category_score(self.category_flatten(x), labels=gt_standards)
        proposal_deltas = self.bbox_pred(x)
        embedding = self.standard_cls_bone(x)
        standard_score1, standard_score2 = self.std_cls_score(embedding, labels=gt_standards)
        # if self.std_cls_loss_type == 'arc':
        #     standard_score = self.std_cls_score(embedding, labels=gt_standards)
        # else:
        #     standard_score = self.std_cls_score(embedding)
        return category_score1, category_score2, proposal_deltas, standard_score1, standard_score2

    def losses(self, predictions, proposals):
        category_score1, category_score2, proposal_deltas, standard_score1, standard_score2 = predictions
        return MlabelStandardFastRCNNOutput(category_score1, category_score2, proposal_deltas, standard_score1,
                                            standard_score2,
                                            proposals, self.box2box_transform, self.std_cls_loss_type,
                                            self.category_loss_type, self.smooth_l1_beta,
                                            self.box_reg_loss_type,
                                            self.box_reg_loss_weight).losses()
        # return MlabelStandardFastRCNNOutput(category_score, proposal_deltas, standard_score, proposals,
        #                                     self.box2box_transform, self.category_loss_type, self.smooth_l1_beta,
        #                                     self.box_reg_loss_type,
        #                                     self.box_reg_loss_weight).losses()

    def inference(self, predictions, proposals):
        """
        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        image_shapes = [x.image_size for x in proposals]
        category_logits, _, proposal_deltas, standard_logits, _ = predictions
        category_scores = self.predict_probs(category_logits, proposals)
        standard_scores = self.predict_probs(standard_logits, proposals)
        boxes = self.predict_boxes(proposal_deltas, proposals)
        # instances: (list[Instances])
        # kept_indices: (list[Tensor])
        instances, kept_indices = fast_rcnn_inference(boxes, category_scores, image_shapes, self.test_score_thresh,
                                                      self.test_nms_thresh, self.test_topk_per_image)
        for i, indices in enumerate(kept_indices):
            instances[i].pred_standards = torch.argmax(standard_scores[i][indices], dim=-1)

        return instances, kept_indices

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
