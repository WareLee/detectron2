import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from detectron2.data.transforms import Transform

class ResizeWithPadTransform(Transform):
    """@Will Lee这是一个Transform,是为了配合ResizeWithPad这一TransformGen而设置"""

    def __init__(self, h, w, new_h, new_w, output_h, output_w, interp=None, pad_value=0):
        super(ResizeWithPadTransform, self).__init__()
        assert output_h >= new_h and output_w >= new_w, '输出宽高应大于缩放后的宽高 ({},{}) vs ({},{}) '.format(output_w, output_h,
                                                                                                   new_w, new_h)
        self.h = h
        self.w = w
        self.new_h = new_h
        self.new_w = new_w
        self.output_h = output_h
        self.output_w = output_w
        self.pad_value = pad_value
        self.interp = interp

    def inverse(self):
        raise Exception('ResizeWithPadTransform的inverse()方法尚未实现')

    def apply_segmentation(self, segmentation):
        raise Exception('ResizeWithPadTransform的apply_segmentation()方法尚未实现')

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_image(self, img, interp=None):
        assert img.shape[:2] == (self.h, self.w)
        assert len(img.shape) <= 4

        ret = np.ones(shape=(self.output_h, self.output_w, img.shape[-1]), dtype=img.dtype) * self.pad_value

        if img.dtype == np.uint8:
            ret = np.asarray(ret, img.dtype)

            pil_image = Image.fromarray(img)
            interp_method = interp if interp is not None else self.interp
            pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)

            ret[:self.new_h, :self.new_w, :] = np.asarray(pil_image)
        else:
            # PIL only supports uint8
            img = torch.from_numpy(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {Image.BILINEAR: "bilinear", Image.BICUBIC: "bicubic"}
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[self.interp]
            img = F.interpolate(img, (self.new_h, self.new_w), mode=mode, align_corners=False)
            shape[:2] = (self.new_h, self.new_w)
            resized_img = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)
            ret[:self.new_h, :self.new_w, :] = resized_img

        return ret
