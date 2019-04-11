import numpy as np
import torch


def preprocess_true_boxes(true_boxes, anchors, image_size, num_classes):
    """Find detector in YOLO where ground truth box should appear.
    Parameters
    ----------
    true_boxes : array
        List of ground truth boxes in form of relative x, y, w, h, class.
        Relative coordinates are in the range [0, 1] indicating a percentage
        of the original image dimensions.
    anchors : array
        List of anchors in form of w, h.
        Anchors are assumed to be in the range [0, conv_size] where conv_size
        is the spatial dimension of the final convolutional features.
    image_size : array-like
        List of image dimensions in form of h, w in pixels.
    Returns
    -------
    detectors_mask : array
        0/1 mask for detectors in [num_anchors, 1 ,conv_height, conv_width]
        that should be compared with a matching ground truth box.
    matching_true_boxes: array
        Same shape as detectors_mask with the corresponding ground truth box
        adjusted for comparison with predicted parameters at training time.
    """
    height, width = image_size
    num_anchors = len(anchors)
    # Downsampling factor of 5x 2-stride max_pools == 32.
    # TODO: Remove hardcoding of downscaling calculations.
    assert height % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    assert width % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    conv_height = height // 32
    conv_width = width // 32
    num_box_params = true_boxes.shape[1] - 1 + num_classes
    detectors_mask = np.zeros(
        (num_anchors, 1, conv_height, conv_width), dtype=np.float32)
    matching_true_boxes = np.zeros(
        (num_anchors, num_box_params, conv_height, conv_width),
        dtype=np.float32)

    for box in true_boxes:
        # scale box to convolutional feature spatial dimensions
        box_class = box[4:5].astype(np.int32)
        box_class_one_hot = np.zeros(num_classes)
        box_class_one_hot[box_class] = 1

        box = box[0:4] * np.array(
            [conv_width, conv_height, conv_width, conv_height])
        i = min(conv_height - 1, max(np.floor(box[1]).astype('int'), 0))
        j = min(conv_width - 1, max(np.floor(box[0]).astype('int'), 0))
        best_iou = 0
        best_anchor = 0

        for k, anchor in enumerate(anchors):
            # Find IOU between box shifted to origin and anchor box.
            box_maxes = box[2:4] / 2.
            box_mins = -box_maxes
            anchor_maxes = (anchor / 2.)
            anchor_mins = -anchor_maxes

            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[0] * intersect_wh[1]
            box_area = box[2] * box[3]
            anchor_area = anchor[0] * anchor[1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            if iou > best_iou:
                best_iou = iou
                best_anchor = k

        if best_iou > 0:
            detectors_mask[best_anchor, :, i, j] = 1
            adjusted_box = np.array(
                [
                    box[0] - j,
                    box[1] - i,
                    np.log(box[2] / anchors[best_anchor][0]),
                    np.log(box[3] / anchors[best_anchor][1])
                ],
                dtype=np.float32)
            adjusted_box = np.concatenate((adjusted_box, box_class_one_hot))
            matching_true_boxes[best_anchor, :, i, j] = adjusted_box
    return detectors_mask, matching_true_boxes


def boxes_to_cornels(box_xy, box_wh):
    """Convert boxes from x_center,y_center, width, height to x_min, y_min, x_max, ymax

    Parameters
    ----------
    box_xy : torch.Tensor
        Predicted xy value
        shape [batch_size, num_anchors, 2, conv_height, conv_width]
    box_wh : torch.Tensor
        Predicted wh value shape [batch_size, num_anchors, 2, conv_height, conv_width]
    Returns
    -------
    torch.Tensor
        Boxes in x_min, y_min, x_max, y_max for mat
        shape [batch_size, num_anchors, 4, conv_height, conv_width]

    """

    box_mins = box_xy - box_wh / 2.
    box_maxes = box_xy + box_wh / 2.

    return torch.cat(
        (
            box_mins,
            box_maxes
        ),
        dim=2
    )


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
    """filter boxes that has score smaller than threshold
    Parameters
    ----------
    box_confidence : torch.Tensor
        Object confidence
        shape [batch_size, num_anchors, 1, conv_height, conv_width]
    boxes : torch.Tensor
        boxes in format x_min, y_min, x_max, y_max
        shape [batch_size, num_anchors, 4, conv_height, conv_width]

    box_class_probs : TYPE
        Class probalility
        shape [batch_size, num_anchors, num_classes, conv_height, conv_width]
    threshold : float, optional
        Score threshold

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Filtered boxes, scores, classes
    """

    batch_size, num_anchors, _, conv_height, conv_width = box_confidence.shape

    box_scores = box_confidence * box_class_probs

    box_classes = torch.argmax(box_scores, dim=2, keepdim=True)

    box_class_scores, _ = torch.max(box_scores, dim=2, keepdim=True)

    prediction_mask = box_class_scores > threshold

    classes = box_classes[prediction_mask]
    scores = box_class_scores[prediction_mask]

    boxes = boxes.permute(0, 1, 3, 4, 2)
    prediction_mask = prediction_mask.permute(0, 1, 3, 4, 2)
    boxes = boxes[prediction_mask.expand_as(boxes)].view(-1, 4)

    return boxes, scores, classes
