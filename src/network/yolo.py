from .base import YoloBody, YoloHead
from torch import nn
import torch
from ..config import IOU_THRESHOLD
from ..utils.process_boxes import yolo_filter_boxes, boxes_to_cornels


class Yolo(nn.Module):
    """docstring for Yolo"""

    def __init__(self, anchors, classes, iou_threshold=IOU_THRESHOLD):
        super(Yolo, self).__init__()
        self.num_anchors = anchors.shape[0]
        self.classes = classes
        self.num_classes = len(classes)
        self.anchors = torch.from_numpy(anchors)
        self.yolo_body = YoloBody(num_anchors=self.num_anchors, num_classes=len(classes))
        self.yolo_head = YoloHead(self.anchors, len(classes))

        self.object_scale = 5
        self.no_object_scale = 1
        self.class_scale = 1
        self.coordinates_scale = 1
        self.iou_threshold = iou_threshold

    def forward(self, x):
        conv_features = self.yolo_body(x)
        # x = self.yolo_head(x)
        # x = self.yolo_head(conv_features)
        return conv_features

    def eval(self, yolo_output, image_shape, max_boxes=10, score_threshold=0.6, iou_threshold=0.5):
        pred_confidence, pred_xy, pred_wh, pred_class_prob = self.yolo_head(yolo_output)
        boxes = boxes_to_cornels(pred_xy, pred_wh)
        boxes, scores, classes = yolo_filter_boxes(pred_confidence, boxes, pred_class_prob, score_threshold)
        return boxes, scores, classes

    def loss(self, yolo_output: torch.Tensor, true_boxes: torch.Tensor, detectors_mask: torch.Tensor, matching_true_boxes: torch.Tensor):
        """Summary

        Parameters
        ----------
        yolo_output : torch.Tensor
            conv_features shape : (batch_size, num_anchors * (5 + num_classes), height, width)
        true_boxes : torch.Tensor
            Ground truth boxes tensor with shape [batch, num_true_boxes, 5]
            containing box x_center, y_center, width, height, and class.
        detectors_mask : torch.Tensor
            0/1 mask for detectors in [batch_sizes, num_anchors, 1, conv_height, conv_width]
            that should be compared with a matching ground truth box.
        matching_true_boxes : torch.Tensor
            Same shape as detectors_mask with the corresponding ground truth box
            adjusted for comparison with predicted parameters at training time.
            [batch_sizes, num_anchors, 4 + num_class, conv_height, conv_width]

        """
        # classification loss

        pred_confidence, pred_xy, pred_wh, pred_class_prob = self.yolo_head(yolo_output)
        # pred_confidence : torch.Size([batch_size, num_anchors, 1, conv_height, conv_width])
        # pred_xy : torch.Size([batch_size, num_anchors, 2, conv_height, conv_width])
        # pred_wh : torch.Size([batch_size, num_anchors, 2, conv_height, conv_width])
        # pred_class_prob : torch.Size([batch_size, num_anchors, 3, conv_height, conv_width])

        batch_size, conv_number_channels, feats_height, feats_width = yolo_output.shape
        feats = yolo_output.view((-1, self.num_anchors, self.num_classes + 5, feats_height, feats_width))

        box_xy = torch.sigmoid(feats[:, :, :2, ...])
        box_wh = feats[:, :, 2:4, ...]
        pred_boxes = torch.cat((box_xy, box_wh), 2)

        # batch, num_anchors, num_true_boxes, box_params, conv_height, conv_width

        pred_xy = pred_xy.unsqueeze(2)  # batch_size, num_anchors, 1 , 2, conv_height, conv_width
        pred_wh = pred_wh.unsqueeze(2)  # batch_size, num_anchors, 1 , 2, conv_height, conv_width

        pred_wh_mid_point = pred_wh / 2
        pred_min = pred_xy - pred_wh_mid_point
        pred_max = pred_xy + pred_wh_mid_point

        true_boxes_shape = true_boxes.shape

        true_boxes = true_boxes.view(true_boxes_shape[0], 1, true_boxes_shape[1], true_boxes_shape[2], 1, 1)

        true_xy = true_boxes[:, :, :, 0:2, ...]
        true_wh = true_boxes[:, :, :, 2:4, ...]

        true_wh_mid_point = true_wh / 2
        true_max = true_xy + true_wh_mid_point
        true_min = true_xy - true_wh_mid_point

        intersect_min = torch.min(true_min, pred_min)
        intersect_max = torch.max(true_max, pred_max)
        intersect_wh = (intersect_max - intersect_min).clamp(min=0)
        intersect_areas = intersect_wh[:, :, :, 0, ...] * intersect_wh[:, :, :, 1, ...]

        pred_areas = pred_wh[:, :, :, 0, ...] * pred_wh[:, :, :, 1, ...]
        true_areas = true_wh[:, :, :, 0, ...] * true_wh[:, :, :, 1, ...]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = intersect_areas / union_areas  # torch.Size([batch_size, num_anchors, num_true_boxes, conv_height, conv_width])
        best_iou, best_iou_index = torch.max(iou_scores, dim=2, keepdim=True)
        # torch.Size([batch_size, num_anchors, 1, conv_height, conv_width])

        object_mask = (best_iou > self.iou_threshold).type(torch.FloatTensor)         # torch.Size([batch_size, num_anchors, 1, conv_height, conv_width])

        no_object_weigth = self.no_object_scale * (1 - object_mask) * (1 - detectors_mask)
        no_object_loss = no_object_weigth * ((- pred_confidence) ** 2)

        object_loss = self.object_scale * detectors_mask * ((best_iou - pred_confidence) ** 2)

        confidence_loss = object_loss + no_object_loss

        matching_classes = matching_true_boxes[:, :, 4:, ...]

        classification_loss = self.class_scale * detectors_mask * (matching_classes - pred_class_prob) ** 2

        matching_boxes = matching_true_boxes[:, :, 0:4, ...]

        coordinates_loss = self.coordinates_scale * detectors_mask * (matching_boxes - pred_boxes) ** 2

        total_loss = 0.5 * (confidence_loss.sum() + classification_loss.sum() + coordinates_loss.sum()) / batch_size

        return total_loss
