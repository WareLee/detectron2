"""
@Author WillLee 2020/7/17

功能类似于 load_coco_json(),读取单标签
"""
import os
from detectron2.structures.boxes import BoxMode


category2id = {'丘脑_标准': 0, '丘脑_非标准': 1, '腹部_标准': 2, '腹部_非标准': 3, '股骨_标准': 4, '股骨_非标准': 5, '脊柱_标准': 6, '脊柱_非标准': 7,
               '小脑水平横切面_标准': 8, '小脑水平横切面_非标准': 9, '四腔心切面_标准': 10, '四腔心切面_非标准': 11}
image_root = r'/home/ultrasonic/hnumedical/SourceCode/StdPlane-2/CenterNet-master/data/Muti_Planes_Det/images'


# TODO 应该用户自定义
def line2dict_func(line, image_id):
    """
    例: xxx.jpg,w,h,238,159,720,538,腹部,非标准,238,159,720,538,腹部,非标准,...
    Args:
        line: str
        image_id: str or int ,该图的唯一标识

    Return : dict, with fields:
                -file_name: str
                -width,height: int
                -image_id: str or int , 该图的唯一标识
                -annotations: list[dict], where dict with fields:
                        -bbox: list[float]
                        -category_id: int
    """
    file_name, width, height, *items = line.strip().split(',')
    file_name = os.path.join(image_root, file_name)
    width = int(width)
    height = int(height)
    annotations = []
    for i in range(0, len(items), 6):
        bbox = [float(it) for it in items[i:i + 4]]
        # bbox超界矫正
        bbox = [max(0, it) for it in bbox]  # 去除负数
        bbox[0] = min(width - 1, bbox[0])  # 宽度超界矫正
        bbox[2] = min(width - 1, bbox[2])
        bbox[1] = min(height - 1, bbox[1])  # 高度超界矫正
        bbox[3] = min(height - 1, bbox[3])
        # 去除不合规的框（XYXY）
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            continue

        category_id = category2id['{}_{}'.format(items[i + 4], items[i + 5])]
        dic = {'bbox': bbox, 'category_id': category_id, 'iscrowd': 0,
               'bbox_mode': BoxMode.XYXY_ABS}
        annotations.append(dic)
    if len(annotations)==0:
        return None

    return {'file_name': file_name, 'width': width, 'height': height, 'image_id': image_id,
            'annotations': annotations}


# TODO 只需要用户自定义line2dict_func函数，且line2dict_func只接受两个参数
def load_csv(csv_path, line2dict_func=line2dict_func, comment=True):
    assert os.path.exists(csv_path)
    dicts = []
    with open(csv_path, encoding='utf-8') as f:
        lines = f.readlines()
        if comment:
            lines = lines[1:]
    for image_id, line in enumerate(lines):
        per_img_dic = line2dict_func(line, image_id)
        if per_img_dic is not None:
            dicts.append(per_img_dic)
    return dicts


if __name__ == '__main__':
    csv_path = r'xx.csv'
    category2id = {'丘脑_标准': 0, '丘脑_非标准': 1, '腹部_标准': 2, '腹部_非标准': 3, '股骨_标准': 4, '股骨_非标准': 5, '脊柱_标准': 6, '脊柱_非标准': 7,
                   '小脑水平横切面_标准': 8, '小脑水平横切面_非标准': 9, '四腔心切面_标准': 10, '四腔心切面_非标准': 11}
    list_dic = load_csv(csv_path, lambda line, image_id: line2dict_func(line, image_id, category2id),
                        comment=True)
