"""
@Author WillLee 2020/7/17

功能类似于 load_coco_json(),读取多标签
"""
import os
from detectron2.structures.boxes import BoxMode

category2id = {'丘脑': 0, '腹部': 1, '股骨': 2, '脊柱': 3, '小脑水平横切面': 4, '四腔心切面': 5}
standard2id = {'非标准': 0, '标准': 1}
image_root = r'/home/ultrasonic/hnumedical/SourceCode/StdPlane-2/CenterNet-master/data/Muti_Planes_Det/images'


def line2dict_func(line, image_id):
    """
    例: xxx.jpg,238,159,720,538,腹部,非标准,238,159,720,538,腹部,非标准,...
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
                        -standard_id: int
                        -iscrowd: int,0 or 1
    """
    file_name, width, height, *items = line.strip().split(',')
    file_name = os.path.join(image_root, file_name)
    width = int(width)
    height = int(height)
    annotations = []
    for i in range(0, len(items), 6):
        bbox = [float(it) for it in items[i:i + 4]]
        category_id = category2id[items[i + 4]]
        standard_id = standard2id[items[i + 5]]
        dic = {'bbox': bbox, 'category_id': category_id, 'standard_id': standard_id, 'iscrowd': 0,
               'bbox_mode': BoxMode.XYXY_ABS}
        annotations.append(dic)

    return {'file_name': file_name, 'width': width, 'height': height, 'image_id': image_id,
            'annotations': annotations}


def load_csv(csv_path, line2dict_func=line2dict_func, comment=True):
    assert os.path.exists(csv_path)
    dicts = []
    with open(csv_path, encoding='utf-8') as f:
        lines = f.readlines()
        if comment:
            lines = lines[1:]
    for image_id, line in enumerate(lines):
        per_img_dic = line2dict_func(line, image_id)
        dicts.append(per_img_dic)
    return dicts
