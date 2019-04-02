from .base import YoloBody, YoloHead
from torch import nn
import torch


class Yolo(nn.Module):
    """docstring for Yolo"""

    def __init__(self, anchors, classes):
        super(Yolo, self).__init__()
        self.num_anchors = anchors.shape[0]
        self.classes = classes
        self.anchors = torch.from_numpy(anchors)
        self.yolo_body = YoloBody(num_anchors=self.num_anchors, num_classes=len(classes))
        self.yolo_head = YoloHead(self.anchors, len(classes))

        self.object_scale = 5
        self.no_object_scale = 1
        self.class_scale = 1
        self.coordinates_scale = 1

    def forward(self, x):
        conv_features = self.yolo_body(x)
        x = self.yolo_head(x)
        x = self.yolo_head(conv_features)
        return x

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
            0/1 mask for detectors in [num_anchors, 1, conv_height, conv_width]
            that should be compared with a matching ground truth box.
        matching_true_boxes : torch.Tensor
            Same shape as detectors_mask with the corresponding ground truth box
            adjusted for comparison with predicted parameters at training time.

        """
        # classification loss

        pred_confidence, pred_xy, pred_wh, pred_class_prob = self.yolo_head(yolo_output)
        # pred_conficence : torch.Size([batch_size, num_anchors, 1, conv_height, conv_width])
        # pred_wh : torch.Size([batch_size, num_anchors, 2, conv_height, conv_width])
        # pred_xy : torch.Size([batch_size, num_anchors, 2, conv_height, conv_width])
        # pred_class_prob : torch.Size([batch_size, num_anchors, 3, conv_height, conv_width])

        batch_size, conv_number_channels, feats_height, feats_width = yolo_output.shape
        feats = yolo_output.view((-1, self.num_anchors, self.num_classes + 5, feats_height, feats_width))

        # batch, num_anchors, num_true_boxes, box_params, conv_height, conv_width

        pred_xy = pred_xy.unsqueeze(2)  # batch_size, num_anchors, 1 , 2, conv_height, conv_width
        pred_wh = pred_wh.unsqueeze(2)  # batch_size, num_anchors, 1 , 2, conv_height, conv_width

        pred_wh_mid_point = pred_wh / 2
        pred_min = pred_xy - pred_wh_mid_point
        pred_max = pred_xy + pred_wh_mid_point

        true_boxes_shape = true_boxes.shape

        true_boxes = true_boxes.view(true_boxes_shape[0], 1, true_boxes_shape[0], true_boxes_shape[1], 1, 1)

        true_xy = true_boxes[:, :, :, 0:2, ...]
        true_wh = true_boxes[:, :, :, 2:4, ...]


        true_wh_mid_point = true_wh / 2
        true_max = true_xy + true_wh_mid_point
        true_min = true_xy - true_wh_mid_point