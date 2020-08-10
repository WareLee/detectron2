from detectron2.data.transforms import augmentation_impl as aug
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron3.data.transforms import augmentation_impl as custom_aug
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
import argparse
import os
from modeling.roi_heads import *
from data.dataset_mapper import MLablesDatasetMapper
from data.datasets.mlabel_csv import load_csv, category2id, standard2id


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
    cfg = get_cfg()
    # 新增加的关键字，新建出来 ----------------
    cfg.MODEL.ROI_HEADS.PREDICTOR = 'MlabelStandardFastRCNNOutputLayer'
    cfg.MODEL.ROI_HEADS.STD_NUM_CLS = 2
    cfg.MODEL.ROI_HEADS.STD_CLS_LOSS_TYPE = 'softmax'
    cfg.MODEL.ROI_HEADS.STD_ARC_LOSS_S = 20
    cfg.MODEL.ROI_HEADS.STD_ARC_LOSS_M = 0.2
    cfg.MODEL.ROI_HEADS.STD_ARC_LOSS_EASY_MARGIN = False
    cfg.MODEL.ROI_HEADS.STD_CLS_BRANCH_NAME = '2fc'
    cfg.MODEL.ROI_HEADS.STD_CLS_EMB_DIM = 512
    cfg.MODEL.ROI_HEADS.REDUCTION = 8
    cfg.MODEL.ROI_HEADS.CATEGORY_LOSS_TYPE=''
    # -----------------------------------------

    cfg.merge_from_file(args.config_file)
    # default_setup(cfg, args)  # 仅关注 args.eval_only 和 args.config_file

    # 分大类
    cfg.MODEL.ROI_HEADS.CATEGORY_LOSS_TYPE = args.cate_loss

    # fine_bone and std_loss
    cfg.MODEL.ROI_HEADS.STD_CLS_BRANCH_NAME = args.fine_bone
    cfg.MODEL.ROI_HEADS.STD_CLS_LOSS_TYPE = args.std_loss

    # cfg.OUTPUT_DIR
    # cfg.MODEL.WEIGHTS = '/data/will/StdPlanesSelection_FasterRCNN/logs/DWSE/softmax/model_0010754.pth'
    sub = 'cate_' + cfg.MODEL.ROI_HEADS.CATEGORY_LOSS_TYPE+'_28'
    cfg.OUTPUT_DIR = os.path.join('/data/will/StdPlanesSelection_FasterRCNN/logs', args.fine_bone, args.std_loss, sub)
    # cfg.OUTPUT_DIR = os.path.join('/data/will/StdPlanesSelection_FasterRCNN/logs', args.fine_bone, args.std_loss)
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    # cfg.solver
    ITERS_IN_ONE_EPOCH = 38267 // cfg.SOLVER.IMS_PER_BATCH
    cfg.SOLVER.STEPS = (ITERS_IN_ONE_EPOCH * 10, ITERS_IN_ONE_EPOCH * 18)
    cfg.SOLVER.MAX_ITER = ITERS_IN_ONE_EPOCH * 24  # 训练24个epoch
    cfg.SOLVER.CHECKPOINT_PERIOD = int(ITERS_IN_ONE_EPOCH * 0.5)

    return cfg


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    register_dataset()
    cfg = setup(args)
    print(cfg)
    trainer = MlabelsTrainer(cfg)
    trainer.resume_or_load(resume=True)
    if args.eval_only == 1:
        # TODO 有待完善
        evaluators = []
        trainer.test(cfg, trainer.model, evaluators=evaluators)
    else:
        trainer.train()


if __name__ == '__main__':
    # TODO 确定哪些关键字参数通过命令行带入
    # fine_bone , std_loss
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='3')

    parser.add_argument('--config_file', type=str, default='./faster_rcnn_R_50_FPN_3x_mlabel.yaml')
    parser.add_argument('--eval_only', type=int, default=0)

    parser.add_argument('--fine_bone', type=str, default='2fc')  # 2fc or DWSE or 133ConvSE
    parser.add_argument('--std_loss', type=str, default='softmax')  # softmax or arc
    parser.add_argument('--cate_loss', type=str, default='cross_entropy')  # cross_entropy or arc

    args = parser.parse_args()
    main(args)

    # python train_net_mlabels.py --gpu 2 --fine_bone 2fc --std_loss softmax > ./logs/2fc/softmax/log.txt 2>&1 &

    # 多标签分类
    # python train_net_mlabels.py --gpu 1 --fine_bone DWSE --std_loss softmax > /data/will/StdPlanesSelection_FasterRCNN/logs/DWSE/softmax/log.txt 2>&1 &

    # python train_net_mlabels.py --gpu 6 --fine_bone 133ConvSE --std_loss softmax --cate_loss softmax
    # python train_net_mlabels.py --gpu 6 --fine_bone 133ConvSE --std_loss softmax --cate_loss arc
    # python train_net_mlabels.py --gpu 6 --fine_bone 133ConvSE --std_loss softmax --cate_loss arc+softmax
