import numpy as np
import torch
from collections import defaultdict
from ..process_boxes import preprocess_true_boxes
from ...config import VOC_ANCHORS
import logging
from typing import List, Tuple

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
    classes = blobs[0]['classes']
    batch_tensor = torch.Tensor(
        current_batch_size, 3, max_height, max_width).fill_(0.)
    total_boxes = 0
    batch_boxes = np.empty((0, 5))
    img_name = []
    im_info = np.array([[batch_tensor.shape[2], batch_tensor.shape[3]]])
    batch_boxes = torch.zeros((current_batch_size, max_boxes, 5))
    detectors_mask = torch.zeros((current_batch_size, VOC_ANCHORS.shape[0], 1, batch_tensor.shape[2] // 32, batch_tensor.shape[3] // 32))
    matching_true_boxes = torch.zeros((current_batch_size, VOC_ANCHORS.shape[0], 4 + len(classes), batch_tensor.shape[2] // 32, batch_tensor.shape[3] // 32))

    for i, blob in enumerate(blobs):
        shape = blob['tensor'].shape
        gt_classes = blob['gt_classes']
        current_boxes = blob['boxes']

        current_boxes = np.hstack((current_boxes, gt_classes[:, np.newaxis]))
        current_detectors_mask, current_matching_true_boxes = preprocess_true_boxes(current_boxes, VOC_ANCHORS, (batch_tensor.shape[2], batch_tensor.shape[3]), len(blob['classes']))
        batch_tensor[i, :, :shape[1], :shape[2]] = blob['tensor']
        total_boxes = current_boxes.shape[0]
        detectors_mask[i] = torch.Tensor(current_detectors_mask)
        matching_true_boxes[i] = torch.Tensor(current_matching_true_boxes)
        batch_boxes[i, :total_boxes, :5] = torch.Tensor(current_boxes)

        img_name.append(blob['im_name'])

    return batch_tensor, batch_boxes, detectors_mask, matching_true_boxes, im_info, img_name


def convert_data_v3(blobs, ratios: List[int]):
    logger.debug(blobs)
    blobs = [i for i in blobs if i is not None]
    current_batch_size = len(blobs)
    if not current_batch_size:
        return None

    logger.debug([blob['tensor'].shape for blob in blobs])

    max_height = np.max([blob['tensor'].shape[1] for blob in blobs])
    max_width = np.max([blob['tensor'].shape[2] for blob in blobs])
    max_boxes = max([blob['boxes'].shape[0] for blob in blobs])
    classes = blobs[0]['classes']
    batch_tensor = torch.Tensor(
        current_batch_size, 3, max_height, max_width).fill_(0.)
    total_boxes = 0
    batch_boxes = np.empty((0, 5))
    img_name = []
    im_info = np.array([[batch_tensor.shape[2], batch_tensor.shape[3]]])
    batch_boxes = torch.zeros((current_batch_size, max_boxes, 5))

    mask_truebox = {}

    for ratio in ratios:
        mask_truebox['detector_mask_%d' % ratio] = torch.zeros((current_batch_size, VOC_ANCHORS.shape[0], 1, batch_tensor.shape[2] // ratio, batch_tensor.shape[3] // ratio))
        mask_truebox['matching_true_boxes_%d' % ratio] = torch.zeros((current_batch_size, VOC_ANCHORS.shape[0], 4 + len(classes), batch_tensor.shape[2] // ratio, batch_tensor.shape[3] // ratio))

    for i, blob in enumerate(blobs):
        shape = blob['tensor'].shape
        gt_classes = blob['gt_classes']
        current_boxes = blob['boxes']
        current_boxes = np.hstack((current_boxes, gt_classes[:, np.newaxis]))

        for ratio in ratios:
            current_detectors_mask, current_matching_true_boxes = preprocess_true_boxes(current_boxes, VOC_ANCHORS, (batch_tensor.shape[2], batch_tensor.shape[3]), len(blob['classes']), ratio)
            mask_truebox['detector_mask_%d' % ratio][i] = torch.Tensor(current_detectors_mask)
            mask_truebox['matching_true_boxes_%d' % ratio][i] = torch.Tensor(current_matching_true_boxes)

        batch_tensor[i, :, :shape[1], :shape[2]] = blob['tensor']
        total_boxes = current_boxes.shape[0]
        batch_boxes[i, :total_boxes, :5] = torch.Tensor(current_boxes)

        img_name.append(blob['im_name'])

    return batch_tensor, batch_boxes, mask_truebox, im_info, img_name


def convert_data_with_out_img(blobs):

    dd = defaultdict(list)
    for i, blob in enumerate(blobs):
        for key, val in blob.iteritems():  # .items() in Python 3.
            dd[key].append(val)
    return dict(dd)
