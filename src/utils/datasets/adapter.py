import numpy as np
import torch
from collections import defaultdict
from ..process_boxes import preprocess_true_boxes
from ...config import VOC_ANCHORS
import logging

logger = logging.getLogger("base_logger")


def convert_data(blobs):
    logger.debug(blobs)
    blobs = [i for i in blobs if i is not None]
    current_batch_size = len(blobs)
    if not current_batch_size:
        return None

    logger.debug([blob['tensor'].shape for blob in blobs])

    max_height = np.max([blob['tensor'].shape[1] for blob in blobs])
    max_width = np.max([blob['tensor'].shape[2] for blob in blobs])
    max_boxes = max([blob['boxes'].shape[0] for blob in blobs])
    batch_tensor = torch.Tensor(
        current_batch_size, 3, max_height, max_width).fill_(0.)
    total_boxes = 0
    batch_boxes = np.empty((0, 5))
    img_name = []
    im_info = np.array([[batch_tensor.shape[2], batch_tensor.shape[3]]])
    batch_boxes = np.zeros((current_batch_size, max_boxes, 5))

    for i, blob in enumerate(blobs):
        shape = blob['tensor'].shape
        gt_classes = blob['gt_classes']
        current_boxes = blob['boxes']

        current_boxes = np.hstack((current_boxes, gt_classes[:, np.newaxis]))

        detectors_mask, matching_true_boxes = preprocess_true_boxes(current_boxes, VOC_ANCHORS, (608, 608), len(blob['classes']))

        print(detectors_mask.shape)
        print(matching_true_boxes.shape)

        batch_tensor[i, :, :shape[1], :shape[2]] = blob['tensor']
        total_boxes = current_boxes.shape[0]

        batch_boxes[i, :total_boxes, :5] = current_boxes

        img_name.append(blob['im_name'])

    return batch_tensor, im_info, batch_boxes, img_name


def convert_data_with_out_img(blobs):

    dd = defaultdict(list)
    for i, blob in enumerate(blobs):
        for key, val in blob.iteritems():  # .items() in Python 3.
            dd[key].append(val)
    return dict(dd)
