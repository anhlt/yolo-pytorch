from __future__ import division
import random
import numpy as np
from PIL import Image
from . import functional as F
from collections import Iterable


__all__ = ["Compose", "RandomHorizontalFlip", "Resize", "XyToCenter"]


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Resize(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (
            isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, bboxes):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        resized_img = F.resize(img, self.size, self.interpolation)
        ratio_x = resized_img.size[0] / img.size[0]
        ratio_y = resized_img.size[1] / img.size[1]
        return resized_img, bboxes * [ratio_x, ratio_y, ratio_x, ratio_y]

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class XyToCenter(object):
    """Convert Boxes data from (x_min, y_min, x_max, y_max) , to (x_center, y_center, width, heigth)
    """

    def __call__(self, img, bboxes):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        print(bboxes)
        image_width, image_height = img.size

        bbox_width = bboxes[:, 2] - bboxes[:, 0]
        bbox_height = bboxes[:, 3] - bboxes[:, 1]
        x_center = bboxes[:, 0] + 0.5 * bbox_width
        y_center = bboxes[:, 1] + 0.5 * bbox_height

        converted_bboxes = np.vstack((x_center, y_center, bbox_width, bbox_height)).T

        return img, converted_bboxes / [image_width, image_height, image_width, image_height]

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        img_center = np.array(img.size) / 2
        img_center = np.hstack((img_center, img_center)).astype(np.int32)
        if random.random() < self.p:
            img = F.hflip(img)
            bboxes[:, [0, 2]] += 2 * (img_center[[0, 2]] - bboxes[:, [0, 2]])

            box_w = abs(bboxes[:, 0] - bboxes[:, 2])

            bboxes[:, 0] -= box_w
            bboxes[:, 2] += box_w

        return img, bboxes

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
