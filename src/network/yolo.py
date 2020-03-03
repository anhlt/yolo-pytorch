"""Summary
"""
from .base import YoloBody, YoloHead
from torch import nn
import torch
from ..config import IOU_THRESHOLD
from ..utils.process_boxes import yolo_filter_boxes, boxes_to_cornels
from torchvision.ops import nms
import torchvision.transforms as transforms
from numpy import ndarray
from PIL import Image
from typing import Tuple, List, Callable
import numpy as np


class Yolo(nn.Module):
    """docstring for Yolo

    """

    def __init__(self, anchors: np.ndarray, classes: List[str], iou_threshold: float = IOU_THRESHOLD):
        """Summary

        Parameters
        ----------
        anchors : np.ndarray
            Description
        classes : List[str]
            Description
        iou_threshold : float, optional
            Description
        """
        super(Yolo, self).__init__()
        self.num_anchors: int = anchors.shape[0]
        self.classes: List[str] = classes
        self.num_classes: int = len(classes)
        self.anchors: torch.Tensor = torch.from_numpy(anchors)

        self.yolo_body: nn.Module = nn.DataParallel(YoloBody(
            num_anchors=self.num_anchors, num_classes=len(classes)))

        self.yolo_head: nn.Module = nn.DataParallel(
            YoloHead(self.anchors, len(classes)))

        self.object_scale: int = 5
        self.no_object_scale: int = 1
        self.class_scale: int = 1
        self.coordinates_scale: int = 1
        self.iou_threshold: float = iou_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Summary

        Parameters
        ----------
        x : torch.Tensor
            Description

        Returns
        -------
        torch.Tensor
            Description
        """
        conv_features: torch.Tensor = self.yolo_body(x)
        # x = self.yolo_head(x)
        # x = self.yolo_head(conv_features)
        return conv_features

    def _eval(self, yolo_output, image_shape, max_boxes=10, score_threshold=0.6, iou_threshold=0.5):
        """Summary

        Parameters
        ----------
        yolo_output : TYPE
            Description
        image_shape : TYPE
            Description
        max_boxes : int, optional
            Description
        score_threshold : float, optional
            Description
        iou_threshold : float, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        pred_confidence, pred_xy, pred_wh, pred_class_prob = self.yolo_head(
            yolo_output)
        boxes = boxes_to_cornels(pred_xy, pred_wh)
        boxes, scores, classes = yolo_filter_boxes(
            pred_confidence, boxes, pred_class_prob, score_threshold)

        height = image_shape[0]
        width = image_shape[1]

        boxes = boxes * \
            torch.Tensor([width, height, width, height]).to(yolo_output.device)
        nms_index = nms(boxes, scores, iou_threshold)
        boxes = boxes[nms_index]
        scores = scores[nms_index]
        classes = classes[nms_index]
        return boxes, scores, classes

    def predict(self, image: str, im_to_tensor_func: Callable[[str], Tuple[torch.Tensor, Tuple[float, float]]], score_threshold: float = 0.6, iou_threshold: float = 0.5):
        """Predict

        Parameters
        ----------
        image : str
            Shape: (1, 3, heith, width)
        im_to_tensor_func : Callable[[str], Tuple[torch.Tensor, Tuple[float, float]]]
            Description
        score_threshold : float, optional
            Description
        iou_threshold : float, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        self.eval()
        image_tensor, ratio = im_to_tensor_func(image)
        yolo_output = self(image_tensor)
        image_shape = image_tensor.shape[2:]
        boxes, scores, classes = self._eval(
            yolo_output, image_shape, score_threshold=score_threshold, iou_threshold=iou_threshold)
        boxes = boxes.clamp(0, image_shape[0]).cpu(
        ) / torch.tensor([ratio[0], ratio[1], ratio[0], ratio[1]])
        return boxes, scores.cpu(), classes.cpu()

    def loss(self, yolo_output: torch.Tensor, true_boxes: torch.Tensor, detectors_mask: torch.Tensor, matching_true_boxes: torch.Tensor):
        """Calculate loss

        Parameters
        ----------
        yolo_output : torch.Tensor
            conv_features shape : (batch_size, num_anchors * (5 + num_classes), height, width)
        true_boxes : torch.Tensor
            Ground truth boxes tensor with shape [batch, num_true_boxes, 5]
            containing box x_center, y_center, width, height, and class.
        detectors_mask : torch.Tensor
            0/1 mask for detectors in [batch_size, num_anchors, 1, conv_height, conv_width]
            that should be compared with a matching ground truth box.
        matching_true_boxes : torch.Tensor
            Same shape as detectors_mask with the corresponding ground truth box
            adjusted for comparison with predicted parameters at training time.
            [batch_size, num_anchors, 4 + num_class, conv_height, conv_width]

        Returns
        -------
        TYPE
            Description

        """
        # classification loss

        pred_confidence, pred_xy, pred_wh, pred_class_prob = self.yolo_head(
            yolo_output)
        # pred_confidence : torch.Size([batch_size, num_anchors, 1, conv_height, conv_width])
        # pred_xy : torch.Size([batch_size, num_anchors, 2, conv_height, conv_width])
        # pred_wh : torch.Size([batch_size, num_anchors, 2, conv_height, conv_width])
        # pred_class_prob : torch.Size([batch_size, num_anchors, 3, conv_height, conv_width])

        batch_size, conv_number_channels, feats_height, feats_width = yolo_output.shape
        _, num_true_boxes, _ = true_boxes.shape
        feats = yolo_output.view(
            (-1, self.num_anchors, self.num_classes + 5, feats_height, feats_width))

        box_xy = torch.sigmoid(feats[:, :, :2, ...])
        box_wh = feats[:, :, 2:4, ...]
        pred_boxes = torch.cat((box_xy, box_wh), 2)

        # batch, num_anchors, num_true_boxes, box_params, conv_height, conv_width

        # batch_size, num_anchors, 1 , 2, conv_height, conv_width
        pred_xy = pred_xy.unsqueeze(2).detach()
        # batch_size, num_anchors, 1 , 2, conv_height, conv_width
        pred_wh = pred_wh.unsqueeze(2).detach()
        pred_wh_mid_point = pred_wh / 2
        pred_min = pred_xy - pred_wh_mid_point
        pred_max = pred_xy + pred_wh_mid_point

        true_boxes_shape = true_boxes.shape

        true_boxes = true_boxes.view(
            true_boxes_shape[0], 1, true_boxes_shape[1], true_boxes_shape[2], 1, 1)

        true_xy = true_boxes[:, :, :, 0:2, ...]
        true_wh = true_boxes[:, :, :, 2:4, ...]

        true_wh_mid_point = true_wh / 2
        true_max = true_xy + true_wh_mid_point
        true_min = true_xy - true_wh_mid_point

        intersect_min = torch.max(true_min, pred_min)
        intersect_max = torch.min(true_max, pred_max)
        intersect_wh = (intersect_max - intersect_min).clamp(min=0)

        intersect_areas = intersect_wh[:, :, :,
                                       0, ...] * intersect_wh[:, :, :, 1, ...]

        pred_areas = pred_wh[:, :, :, 0, ...] * pred_wh[:, :, :, 1, ...]
        true_areas = true_wh[:, :, :, 0, ...] * true_wh[:, :, :, 1, ...]

        union_areas = pred_areas + true_areas - intersect_areas
        # torch.Size([batch_size, num_anchors, num_true_boxes, conv_height, conv_width])
        iou_scores = intersect_areas / union_areas
        best_iou, best_iou_index = torch.max(iou_scores, dim=2, keepdim=True)

        # torch.Size([batch_size, num_anchors, 1, conv_height, conv_width])

        # torch.Size([batch_size, num_anchors, 1, conv_height, conv_width])
        object_mask = (best_iou > self.iou_threshold).float().to(
            yolo_output.device)

        no_object_weigth = self.no_object_scale * \
            (torch.ones_like(object_mask) - object_mask) * \
            (torch.ones_like(detectors_mask) - detectors_mask)

        no_object_loss = nn.MSELoss(reduction='sum')(torch.zeros_like(
            pred_confidence), no_object_weigth * pred_confidence)
        object_loss = nn.MSELoss(reduction='sum')(
            self.object_scale * detectors_mask * best_iou, self.object_scale * detectors_mask * pred_confidence)

        matching_classes = matching_true_boxes[:, :, 4:, ...]

        classification_loss = nn.MSELoss(reduction='sum')(
            self.class_scale * detectors_mask * matching_classes, self.class_scale * detectors_mask * pred_class_prob)

        matching_boxes = matching_true_boxes[:, :, 0:4, ...]

        coordinates_loss = nn.MSELoss(reduction='sum')(
            self.coordinates_scale * detectors_mask * matching_boxes, self.class_scale * detectors_mask * pred_boxes)

        total_loss = (object_loss + no_object_loss + classification_loss +
                      coordinates_loss) / (batch_size * num_true_boxes)

        return total_loss
