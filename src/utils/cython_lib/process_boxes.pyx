# cython: profile=True
cimport cython
import numpy as np
cimport numpy as np


DTYPE = np.float
ctypedef np.float_t DTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def preprocess_true_boxes(np.ndarray[DTYPE_t, ndim=2] true_boxes,
                          np.ndarray[DTYPE_t, ndim=2] anchors,
                          tuple image_size):


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
        0/1 mask for detectors in [conv_height, conv_width, num_anchors, 1]
        that should be compared with a matching ground truth box.
    matching_true_boxes: array
        Same shape as detectors_mask with the corresponding ground truth box
        adjusted for comparison with predicted parameters at training time.
    """
    cdef unsigned int height = image_size[0]
    cdef unsigned int width = image_size[1]

    cdef unsigned int num_anchors = anchors.shape[0]


    # Downsampling factor of 5x 2-stride max_pools == 32.
    # TODO: Remove hardcoding of downscaling calculations.
    assert height % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    assert width % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    cdef unsigned int conv_height = height // 32
    cdef unsigned int conv_width = width // 32


    cdef unsigned int num_box_params = true_boxes.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=4] detectors_mask = np.zeros(
        (num_anchors, 1, conv_height, conv_width), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] matching_true_boxes = np.zeros(
        (num_anchors, num_box_params, conv_height, conv_width),
        dtype=DTYPE)

    cdef np.float_t box_class
    # cdef np.ndarray[DTYPE_t, ndim=1] unscaled_box, box_maxes, box_mins, anchor_maxes, anchor_mins
    # cdef np.float_t best_iou, best_anchor


    for box in true_boxes:
        # scale box to convolutional feature spatial dimensions
        box_class = box[4:5]
        unscaled_box = box[0:4] * np.array(
            [conv_width, conv_height, conv_width, conv_height])
        i = np.floor(unscaled_box[1]).astype('int')
        j = min(np.floor(unscaled_box[0]).astype('int'), 1)
        best_iou = 0.
        best_anchor = 0.

        for k, anchor in enumerate(anchors):
        # for k in range(num_anchors):
            # Find IOU between box shifted to origin and anchor box.
            box_maxes = unscaled_box[2:4] / 2.
            box_mins = -box_maxes
            anchor_maxes = (anchor / 2.)
            anchor_mins = -anchor_maxes

            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[0] * intersect_wh[1]
            box_area = unscaled_box[2] * unscaled_box[3]
            anchor_area = anchor[0] * anchor[1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            if iou > best_iou:
                best_iou = iou
                best_anchor = k

        if best_iou > 0:
            detectors_mask[best_anchor, : ,i, j] = 1
            adjusted_box = np.array(
                [
                    unscaled_box[0] - j, unscaled_box[1] - i,
                    np.log(unscaled_box[2] / anchors[best_anchor][0]),
                    np.log(unscaled_box[3] / anchors[best_anchor][1]), box_class
                ],
                dtype=np.float32)
            matching_true_boxes[best_anchor, : ,i, j] = adjusted_box
    return detectors_mask, matching_true_boxes
