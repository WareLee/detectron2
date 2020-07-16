from typing import Tuple
from PIL import Image
from detectron2.data.transforms.augmentation import Augmentation
from .transform import ResizeWithPadTransform


class ResizeWithPad(Augmentation):
    """@WillLee 固定图片输出大小的保持宽高比缩放"""
    def __init__(self, output_shape: tuple, interp=Image.BILINEAR, pad_value=0):
        """
        Args:
        output_shape(tuple): (h,w)
        interp:
        """
        self.output_h = output_shape[0]
        self.output_w = output_shape[1]
        self.interp = interp
        self.pad_value = pad_value

    def get_transform(self, img):
        h, w = img.shape[:2]
        scale = min(self.output_h / h, self.output_w / w)
        newh = min(self.output_h, int(h * scale + 0.5))
        neww = min(self.output_w, int(w * scale + 0.5))
        return ResizeWithPadTransform(h, w, newh, neww, self.output_h, self.output_w, interp=self.interp,
                                      pad_value=self.pad_value)
