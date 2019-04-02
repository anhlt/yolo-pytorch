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

    def forward(self, x):
        x = self.yolo_body(x)
        x = self.yolo_head(x)
        return x
