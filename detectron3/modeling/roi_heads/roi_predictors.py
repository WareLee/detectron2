"""
将 ROIPredictor 从ROIHeads中解耦出来，通过cfg配置指定要使用的ROIPredictor。

ROIPredictor:
    predictions = forward（box_feature） --> 返回结果组织 scores, proposal_deltas
    losses(predictions,proposals) --> 头部损失计算，

    inference（predictions,proposals） -->
"""
from detectron2.utils.registry import Registry
from detectron2.modeling.roi_heads import FastRCNNOutputLayers

__all__ = ['build_roi_predictor','StandardFastRCNNOutputLayers']
ROI_PREDICTORS_REGISTRY = Registry("ROI_PREDICTORS")
ROI_PREDICTORS_REGISTRY.__doc__ = """ """


@ROI_PREDICTORS_REGISTRY.register()
class StandardFastRCNNOutputLayers(FastRCNNOutputLayers):
    pass


def build_roi_predictor(cfg, input_shape):
    name = cfg.MODEL.ROI_HEADS.PREDICTOR
    return ROI_PREDICTORS_REGISTRY.get(name)(cfg, input_shape)
