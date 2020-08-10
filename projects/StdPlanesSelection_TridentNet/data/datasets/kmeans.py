import numpy as np
import os
import math


class YOLO_Kmeans:
    """注：Kmeans的过程中是没有resize操作的，它统计的是原始尺寸图像中目标的大小"""

    def __init__(self, cluster_number, filenames, target_size=None, mode='resize', for_network='yolo_v3'):
        """

        :param cluster_number:
        :param filenames:
        :param target_size: 仅当target_size不为None时考虑图片的缩放问题,图片被用于模型训练时采用的输入大小wh
        :param mode: 当target_size不为None时，该参数有效；
            目前只支持 ‘resize’（直接缩放） 和 ‘resize_pad’（保持原图有效区域比例的缩放，然后填充） 两种缩放模式
        :param for_network: 用于那种网络模型， 目前仅支持 yolo_v3 和 faster_rcnn 两种

        """
        self.cluster_number = cluster_number
        self.target_size = target_size
        self.mode = mode
        self.for_network = for_network
        if isinstance(filenames, list):
            self.filenames = filenames
        else:
            self.filenames = [filenames, ]

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:

            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):
        f = open("yolo_anchors.txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):

        if self.target_size is not None:
            print('图片的缩放将被纳入考虑 ... ... ')

        dataSet = []
        for fname in self.filenames:
            assert os.path.exists(fname)
            with open(fname, mode='r', encoding='utf-8') as f:
                for line in f.readlines()[1:]:
                    # file_name,width,height,x1,y1,x2,y2,BigClass,StdClass
                    imgname, w, h, *insts = line.strip().split(',')
                    w, h = int(w), int(h)
                    if self.target_size is not None:
                        # sh,sw分别表示在缩放之后有效区域的大小
                        if self.mode == 'resize':
                            sw, sh = self.target_size
                        elif self.mode == 'resize_pad':
                            if h / w < self.target_size[1] / self.target_size[0]:
                                sw = self.target_size[0]
                                sh = int(sw * h / w)
                            elif h / w > self.target_size[1] / self.target_size[0]:
                                sh = self.target_size[1]
                                sw = int(w * sh / h)
                            else:
                                sw, sh = self.target_size
                        else:
                            raise Exception('resize mode you asked is not supported yet : {}'.format(self.mode, ))

                    for i in range(0, len(insts), 6):

                        width = int(insts[i + 2]) - int(insts[i])
                        height = int(insts[i + 3]) - int(insts[i + 1])
                        if self.target_size is not None:
                            # width = int(self.target_size[0] / sw * width)
                            # height = int(self.target_size[1] / sh * height)
                            width = abs(int(sw / w * width))
                            height = abs(int(sh / h * height))

                            # @WillLee 如果用于Faster rcnn 应如是计算
                            if self.for_network == 'faster_rcnn':
                                width = int(math.sqrt(width * height))
                                height = width
                        dataSet.append([width, height])
        result = np.array(dataSet)
        return result

    def txt2clusters(self):
        all_boxes = self.txt2boxes()
        for i in range(6):
            result = self.kmeans(all_boxes, k=self.cluster_number)
            result = result[np.lexsort(result.T[0, None])]
            self.result2txt(result)
            print("K anchors:\n {}".format(result))
            print("Accuracy: {:.2f}%".format(
                self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    # cluster_number = 9
    cluster_number = 5
    # cluster_number = 6
    # filenames = [r"D:\cur_work\shuiwei\datasets\dataset\dataset_1\annotations_ruler_only.csv",
    #              r"D:\cur_work\shuiwei\datasets\dataset\dataset_2\annotations_ruler_only.csv",
    #              r"D:\cur_work\shuiwei\datasets\dataset\dataset_3\annotations_ruler_only.csv",
    #              r"D:\cur_work\shuiwei\datasets\dataset\dataset_4\annotations_ruler_only.csv",
    #              r"D:\cur_work\shuiwei\datasets\dataset\dataset_5\annotations_ruler_only.csv",
    #              r"D:\cur_work\shuiwei\datasets\dataset\dataset_6\annotations_ruler_only.csv",
    #              r"D:\cur_work\shuiwei\datasets\dataset\dataset_7\annotations_ruler_only.csv",
    #              r"D:\cur_work\shuiwei\datasets\dataset\dataset_8\annotations_ruler_only.csv",
    #              r"D:\cur_work\shuiwei\datasets\dataset\dataset_9\annotations_ruler_only.csv",
    #              r"D:\cur_work\shuiwei\datasets\dataset\dataset_10\annotations_ruler_only.csv",
    #              r"D:\cur_work\shuiwei\datasets\dataset\dataset_11\annotations_ruler_only.csv",
    #              ]
    filenames = [r'/home/ultrasonic/detectron22/projects/StdPlanesSelection_TridentNet/datasets/huang_6planes/annotations_v4.csv', ]
    kmeans = YOLO_Kmeans(cluster_number, filenames, target_size=(960, 800), mode='resize_pad',
                         for_network='faster_rcnn')
    kmeans.txt2clusters()

    """
    # huang_6planes
    K anchors: [[209 209] [331 331] [446 446] [550 550] [664 664]]  Accuracy: 87.07%
    K anchors: [[208 208] [240 240] [448 448] [560 560] [672 672]]  使为16的倍数

    """
