from detectron2.data.transforms import augmentation_impl as aug
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron3.data.transforms import augmentation_impl as custom_aug
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
import argparse
import os
import modeling.roi_heads.roi_predictors
import modeling.roi_heads
from data.dataset_mapper import MLablesDatasetMapper
from data.datasets.mlabel_csv import load_csv, category2id, standard2id
from tridentnet import add_tridentnet_config


def register_dataset():
    train_csv = r'./datasets/huang_6planes/train.csv'
    test_csv = r'./datasets/huang_6planes/test.csv'
    trainset_name = 'kuangtu6_train'
    testset_name = 'kuangtu6_test'
    DatasetCatalog.register(trainset_name, lambda: load_csv(train_csv))
    MetadataCatalog.get(trainset_name).thing_classes = list(category2id.keys())
    MetadataCatalog.get(trainset_name).standard_classes = list(standard2id.keys())

    DatasetCatalog.register(testset_name, lambda: load_csv(test_csv))
    MetadataCatalog.get(trainset_name).thing_classes = list(category2id.keys())
    MetadataCatalog.get(trainset_name).standard_classes = list(standard2id.keys())


class MlabelsTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        # _C.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
        assert len(cfg.INPUT.MIN_SIZE_TRAIN) == 1
        assert cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING == "choice"

        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        if not isinstance(min_size, int):
            min_size = min(min_size)
        tfm_gens = []
        tfm_gens.append(custom_aug.ResizeWithPad((min_size, max_size)))
        tfm_gens.append(aug.RandomFlip(prob=0.4, horizontal=True, vertical=False))
        tfm_gens.append(aug.RandomFlip(prob=0.4, horizontal=False, vertical=True))
        tfm_gens.append(aug.RandomApply(aug.RandomRotation([-20, 20], expand=False), prob=0.5))
        tfm_gens.append(aug.RandomApply(aug.RandomContrast(0.8, 1.2), prob=0.3))
        tfm_gens.append(aug.RandomApply(aug.RandomBrightness(0.9, 1.1), prob=0.2))

        mapper = MLablesDatasetMapper(cfg, True, augs=tfm_gens)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST

        if not isinstance(min_size, int):
            min_size = min(min_size)
        tfm_gens = []
        tfm_gens.append(custom_aug.ResizeWithPad((min_size, max_size)))
        mapper = MLablesDatasetMapper(cfg, False, augs=tfm_gens)

        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)


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
    register_dataset()
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
    # -----------------------
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[64, 128, 256, 512,704]]
    cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.8]
    # cfg.MODEL.PIXEL_MEAN = [0.15676956, 0.16244236, 0.16733762]
    # cfg.MODEL.PIXEL_STD = [0.17134029, 0.17603996, 0.18256618]
    cfg.MODEL.PIXEL_MEAN = [44.52619347,43.15665151,42.63251777]
    cfg.MODEL.PIXEL_STD = [47.42477607,46.13824758,45.65620214]
    # means [0.17461253 0.16924178 0.16718635]
    # stdevs [0.18597952 0.18093431 0.17904393]
    # -----------------------
    cfg.OUTPUT_DIR = './logs'
    # cfg.MODEL.WEIGHTS = '/home/ultrasonic/detectron22/projects/StdPlanesSelection_TridentNet/logs/model_0005099.pth'
    cfg.MODEL.WEIGHTS = ''
    cfg.SOLVER.IMS_PER_BATCH = 6
    cfg.SOLVER.BASE_LR = 0.001
    ITERS_IN_ONE_EPOCH = 38267 // cfg.SOLVER.IMS_PER_BATCH
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH
    cfg.SOLVER.STEPS = (ITERS_IN_ONE_EPOCH * 5, ITERS_IN_ONE_EPOCH * 12)
    cfg.SOLVER.MAX_ITER = ITERS_IN_ONE_EPOCH * 24
    cfg.SOLVER.CHECKPOINT_PERIOD = int(ITERS_IN_ONE_EPOCH * 0.2)  # 多少次迭代保存一次checkpoint
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
    cfg.DATASETS.TRAIN = ('kuangtu6_train',)
    cfg.DATASETS.TEST = ('kuangtu6_test',)
    print(cfg)
    trainer = MlabelsTrainer(cfg)
    # trainer.resume_or_load(resume=False)
    if args.eval_only == 1:
        evaluators = []
        trainer.test(cfg, trainer.model, evaluators=evaluators)
        return
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--eval_only', type=int, default=0)
    parser.add_argument('--config_file', type=str, default='./configs/tridentnet_fast_R_50_C4_3x.yaml')
    args = parser.parse_args()
    main(args)

    # python train_net_mlabels.py --gpu 2 --eval_only 0 > ./logs/triple_softmax.log 2>&1 &
