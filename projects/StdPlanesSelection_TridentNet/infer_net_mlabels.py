import argparse
from typing import Union, List
import numpy as np
import os
import cv2
import torch
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron3.data.transforms import augmentation_impl as custom_aug
from detectron2.engine import default_setup
from tridentnet import add_tridentnet_config
import modeling.roi_heads.roi_predictors
import modeling.roi_heads


class MLabelsInfer():
    def __init__(self, cfg, category2id, standard2id, aug=None, output_dir=None):
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        self.output_dir = output_dir

        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()

        self.id2catrgory = {v: k for k, v in category2id.items()}
        self.id2standard = {v: k for k, v in standard2id.items()}
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        if aug is None:
            aug = custom_aug.ResizeWithPad((self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MAX_SIZE_TEST))
        self.aug = aug
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, images: Union[List[str], List[np.ndarray]]):
        """
        Args: images, List[str], each str means one image path
                      List[np.ndarray], each np.ndarray means one image of shape (H,W,C)(in BGR order)
        """
        assert isinstance(images, list)
        inputs = []

        if isinstance(images[0], str):
            for i, path in enumerate(images):
                assert os.path.exists(path)
                images[i] = cv2.imread(path)

        for i in range(len(images)):
            if self.input_format == 'RGB':
                img = images[i][:, :, ::-1]
            else:
                img = images[i]
            height, width = img.shape[:2]
            img = self.aug.get_transform(img).apply_image(img)
            img = torch.as_tensor(img.astype(np.float32).transpose(2, 0, 1))
            inputs.append({'image': img, 'height': height, 'width': width})

        with torch.no_grad():
            predictions = self.model(inputs)

        # 保存可视化结果
        if self.output_dir is not None:
            for image, pred_dic in zip(images, predictions):
                instances = pred_dic['instances']
                pred_boxes = instances.pred_boxes.tensor.cpu().numpy()
                scores = instances.scores.cpu().numpy()
                pred_classes = instances.pred_classes.cpu().numpy()
                pred_standards = instances.pred_standards.cpu().numpy()
                for box, score, pred_class, pred_standard in zip(pred_boxes, scores, pred_classes, pred_standards):
                    # if float(score) < 0.25:
                    #     continue
                    box = [int(it) for it in box]
                    cv2.rectangle(image, tuple(box[0:2]), tuple(box[2:4]), (0, 255, 0), 2)
                    ret = '{}_{} {}'.format(self.id2catrgory[int(pred_class)], self.id2standard[int(pred_standard)],
                                            round(float(score), 3))
                    cv2.putText(image, ret, (box[0],box[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                (0, 255, 0), 2)
                new_name = '{}.jpg'.format(len(os.listdir(self.output_dir)), )
                cv2.imwrite(os.path.join(self.output_dir, new_name), image)

        return predictions


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_tridentnet_config(cfg)
    cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    # cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cfg = setup(args)
    cfg.INPUT.MAX_SIZE_TRAIN = 960
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    cfg.INPUT.MAX_SIZE_TEST = 960
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'choice'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
    # -----------------------
    cfg.MODEL.ROI_HEADS.NAME = 'TridentRes5ROITripleHeads'
    cfg.MODEL.ROI_HEADS.PREDICTOR = 'TripleBranchOutputLayer'
    cfg.MODEL.ROI_HEADS.STD_CATEGORY_NUM = 2
    cfg.MODEL.ROI_HEADS.STD_CLS_LOSS_TYPE = 'softmax'  # arc,softmax,arc+softmax
    cfg.MODEL.ROI_HEADS.STD_CLS_ARC_SOFTMAX_LOSS_WEIGHTS = (0.8, 0.2)
    cfg.MODEL.ROI_HEADS.FINE_BONE_NAME = 'PoolFc'  # 细分类分支网络 PoolFc or SE33PoolFc
    cfg.MODEL.ROI_HEADS.FINE_BONE_EMB_DIM = 512
    # -----------------------
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[64, 128, 256, 512, 704]]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[208, 240, 448, 560, 672]]
    cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.8]
    # cfg.MODEL.PIXEL_MEAN = [0.15676956, 0.16244236, 0.16733762]
    # cfg.MODEL.PIXEL_STD = [0.17134029, 0.17603996, 0.18256618]
    cfg.MODEL.PIXEL_MEAN = [44.52619347, 43.15665151, 42.63251777]
    cfg.MODEL.PIXEL_STD = [47.42477607, 46.13824758, 45.65620214]
    # -----------------------
    cfg.OUTPUT_DIR = './logs'
    cfg.MODEL.WEIGHTS = '/home/ultrasonic/detectron22/projects/StdPlanesSelection_TridentNet/logs/model_0008924.pth'
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    ITERS_IN_ONE_EPOCH = 38267 // cfg.SOLVER.IMS_PER_BATCH
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH
    cfg.SOLVER.STEPS = (ITERS_IN_ONE_EPOCH * 5, ITERS_IN_ONE_EPOCH * 12)
    cfg.SOLVER.MAX_ITER = ITERS_IN_ONE_EPOCH * 24
    cfg.SOLVER.CHECKPOINT_PERIOD = int(ITERS_IN_ONE_EPOCH * 0.5)  # 多少次迭代保存一次checkpoint
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
    cfg.DATASETS.TRAIN = ('kuangtu6_train',)
    cfg.DATASETS.TEST = ('kuangtu6_test',)
    print(cfg)
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='6')
    parser.add_argument('--config_file', type=str, default='./configs/tridentnet_fast_R_50_C4_3x.yaml')
    args = parser.parse_args()
    cfg = main(args)

    # category2id = {'丘脑': 0, '腹部': 1, '股骨': 2, '脊柱': 3, '小脑水平横切面': 4, '四腔心切面': 5}
    category2id = {'HC': 0, 'AC': 1, 'FL': 2, 'JZ': 3, 'XN': 4, 'SQX': 5}
    # standard2id = {'非标准': 0, '标准': 1}
    standard2id = {'nstd': 0, 'std': 1}
    output_dir = r'./datasets/results'
    inference = MLabelsInfer(cfg, category2id=category2id, standard2id=standard2id, output_dir=output_dir)
    while True:
        print('Enter image paths: ')
        paths = input().strip().split(',')
        paths = [os.path.join('./datasets/images', it) for it in paths]
        predictions = inference(paths)
        print(predictions)  # 1.png,2.png,3.jpg,4.png
