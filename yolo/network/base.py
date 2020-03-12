"""Summary
"""
from torch import nn
from torch.nn import MaxPool2d
import torch
import torch.nn.functional as F
from typing import Tuple


class Conv2d(nn.Module):
    """docstring for Conv2d

    Attributes
    ----------
    bn : TYPE
        Description
    conv : TYPE
        Description
    relu : TYPE
        Description
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 relu=True,
                 same_padding=False,
                 bn=False):
        """Summary

        Parameters
        ----------
        in_channels : TYPE
            Description
        out_channels : TYPE
            Description
        kernel_size : TYPE
            Description
        stride : int, optional
            Description
        relu : bool, optional
            Description
        same_padding : bool, optional
            Description
        bn : bool, optional
            Description
        """
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding=padding, bias=not bn)
        nn.init.xavier_normal_(self.conv.weight)
        self.bn = nn.BatchNorm2d(
            out_channels, eps=0.001,
            affine=True
        ) if bn else None
        self.relu = nn.LeakyReLU(negative_slope=0.1) if relu else None

    def forward(self, x):
        """Summary

        Parameters
        ----------
        x : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BottleneckBlock(nn.Module):
    """docstring for BottleneckBlock

    Attributes
    ----------
    bottleneck_filter : TYPE
        Description
    first_layer : TYPE
        Description
    input_channels : TYPE
        Description
    outer_filter : TYPE
        Description
    second_layer : TYPE
        Description
    third_layer : TYPE
        Description
    """

    def __init__(self, input_channels: int, outer_filter: int, bottleneck_filter: int):
        """Summary

        Parameters
        ----------
        input_channels : int
            Description
        outer_filter : int
            Description
        bottleneck_filter : int
            Description
        """
        super(BottleneckBlock, self).__init__()
        self.input_channels = input_channels
        self.outer_filter = outer_filter
        self.bottleneck_filter = bottleneck_filter

        self.first_layer = Conv2d(input_channels, outer_filter, 3, same_padding=True, bn=True)
        self.second_layer = Conv2d(outer_filter, bottleneck_filter, 1, same_padding=True, bn=True)
        self.third_layer = Conv2d(bottleneck_filter, outer_filter, 3, same_padding=True, bn=True)

    def forward(self, x):
        """Summary

        Parameters
        ----------
        x : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        x = self.first_layer(x)
        x = self.second_layer(x)
        x = self.third_layer(x)
        return x


class DoubleBottleneckBlock(nn.Module):
    """docstring for DoubleBottleneckBlock

    Attributes
    ----------
    bottleneck_filter : TYPE
        Description
    first_layer : TYPE
        Description
    input_channels : TYPE
        Description
    outer_filter : TYPE
        Description
    second_layer : TYPE
        Description
    third_layer : TYPE
        Description
    """

    def __init__(self, input_channels: int, outer_filter: int, bottleneck_filter: int):
        """Summary

        Parameters
        ----------
        input_channels : int
            Description
        outer_filter : int
            Description
        bottleneck_filter : int
            Description
        """
        super(DoubleBottleneckBlock, self).__init__()
        self.input_channels = input_channels
        self.outer_filter = outer_filter
        self.bottleneck_filter = bottleneck_filter

        self.first_layer = BottleneckBlock(input_channels, outer_filter, bottleneck_filter)
        self.second_layer = Conv2d(outer_filter, bottleneck_filter, 1, same_padding=True, bn=True)
        self.third_layer = Conv2d(bottleneck_filter, outer_filter, 3, same_padding=True, bn=True)

    def forward(self, x):
        """Summary

        Parameters
        ----------
        x : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        x = self.first_layer(x)
        x = self.second_layer(x)
        x = self.third_layer(x)
        return x


class DarknetBodyBottom(nn.Module):
    """docstring for DarknetBodyBottom

    Attributes
    ----------
    eighth_layer : TYPE
        Description
    fifth_layer : TYPE
        Description
    first_layer : TYPE
        Description
    forth_layer : TYPE
        Description
    nineth_layer : TYPE
        Description
    second_layer : TYPE
        Description
    seventh_layer : TYPE
        Description
    sixth_layer : TYPE
        Description
    third_layer : TYPE
        Description
    """

    def __init__(self, **kwargs):
        """Summary

        Parameters
        ----------
        **kwargs
            Description
        """
        super(DarknetBodyBottom, self).__init__()
        self.first_layer = Conv2d(3, 32, 3, **kwargs)
        self.second_layer = MaxPool2d(2)
        self.third_layer = Conv2d(32, 64, 3, **kwargs)
        self.forth_layer = MaxPool2d(2)
        self.fifth_layer = BottleneckBlock(64, 128, 64)
        self.sixth_layer = MaxPool2d(2)
        self.seventh_layer = BottleneckBlock(128, 256, 128)
        self.eighth_layer = MaxPool2d(2)
        self.nineth_layer = DoubleBottleneckBlock(256, 512, 256)

    def forward(self, x):
        """Summary

        Parameters
        ----------
        x : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        x = self.first_layer(x)
        x = self.second_layer(x)
        x = self.third_layer(x)
        x = self.forth_layer(x)
        x = self.fifth_layer(x)
        x = self.sixth_layer(x)
        x = self.seventh_layer(x)
        x = self.eighth_layer(x)
        x = self.nineth_layer(x)
        return x


class DarknetBodyHead(nn.Module):
    """docstring for DarknetBodyHead

    Attributes
    ----------
    eleventh_layer : TYPE
        Description
    tenth_layer : TYPE
        Description
    """

    def __init__(self):
        """Summary
        """
        super(DarknetBodyHead, self).__init__()

        self.tenth_layer = MaxPool2d(2)
        self.eleventh_layer = DoubleBottleneckBlock(512, 1024, 512)

    def forward(self, x):
        """Summary

        Parameters
        ----------
        x : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        x = self.tenth_layer(x)
        x = self.eleventh_layer(x)
        return x


class DarknetBody(nn.Module):
    """docstring for DarknetBody

    Attributes
    ----------
    body_bottom : TYPE
        Description
    body_head : TYPE
        Description
    """

    def __init__(self):
        """Summary
        """
        super(DarknetBody, self).__init__()
        kwargs = {
            "same_padding": True,
            "bn": True
        }

        self.body_bottom = DarknetBodyBottom(**kwargs)
        self.body_head = DarknetBodyHead()

    def forward(self, x):
        """Summary

        Parameters
        ----------
        x : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        x = self.body_bottom(x)
        x = self.body_head(x)

        return x


class DarkNet(nn.Module):
    """docstring for DarkNet

    Attributes
    ----------
    active : TYPE
        Description
    body : TYPE
        Description
    global_avg_pool : TYPE
        Description
    head : TYPE
        Description
    """

    def __init__(self):
        """Summary
        """
        super(DarkNet, self).__init__()
        self.body = DarknetBody()
        self.head = nn.Conv2d(1024, 1000, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.active = nn.Softmax(dim=1)

    def forward(self, x):
        """Summary

        Parameters
        ----------
        x : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        x = self.body(x)
        x = self.head(x)
        x = self.global_avg_pool(x)
        x = self.active(x)
        return x


class Reorg(nn.Module):

    """Summary

    Attributes
    ----------
    stride : TYPE
        Description
    """

    def __init__(self, stride=2):
        """Summary

        Parameters
        ----------
        stride : int, optional
            Description
        """
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        """Summary

        Parameters
        ----------
        x : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H // hs, hs, W // ws, ws).transpose(3, 4).contiguous()
        x = x.view(B, C, H // hs * W // ws, hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, H // hs, W // ws).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, H // hs, W // ws)
        return x


class YoloBody(nn.Module):
    """docstring for YoloBody

    Attributes
    ----------
    after_concat : TYPE
        Description
    body_bottom : TYPE
        Description
    body_head : TYPE
        Description
    find_grain : TYPE
        Description
    first_layer : TYPE
        Description
    last_layer : TYPE
        Description
    num_anchors : TYPE
        Description
    num_classes : TYPE
        Description
    re_org : TYPE
        Description
    second_layer : TYPE
        Description
    """

    def __init__(self, num_anchors, num_classes):
        """Summary

        Parameters
        ----------
        num_anchors : TYPE
            Description
        num_classes : TYPE
            Description
        """
        super(YoloBody, self).__init__()
        kwargs = {
            "same_padding": True,
            "bn": True
        }
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.body_bottom = DarknetBodyBottom(**kwargs)
        self.body_head = DarknetBodyHead()

        self.first_layer = Conv2d(1024, 1024, 3, same_padding=True, bn=True)
        self.second_layer = Conv2d(1024, 1024, 3, same_padding=True, bn=True)

        self.find_grain = Conv2d(512, 64, 1, same_padding=True, bn=True)
        self.re_org = Reorg(stride=2)

        self.after_concat = Conv2d(256 + 1024, 1024, 3, same_padding=True, bn=True)

        self.last_layer = nn.Conv2d(1024, num_anchors * (5 + num_classes), 1, 1, 0, bias=False)

    def forward(self, x):
        """Summary

        Parameters
        ----------
        x : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        x1 = self.body_bottom(x)
        x = self.body_head(x1)
        x = self.first_layer(x)
        x = self.second_layer(x)
        x1 = self.find_grain(x1)
        x1 = self.re_org(x1)
        x = torch.cat((x, x1), dim=1)
        x = self.after_concat(x)
        x = self.last_layer(x)
        return x


class YoloHead(nn.Module):
    """docstring for YoloHead
    """

    def __init__(self, anchors: torch.Tensor, num_classes: int):
        """Convert final layer features to bounding box

        Parameters
        ----------
        anchors : torch.Tensor
            List of anchors in form of w, h.
            shape (num_anchors, 2)
        num_classes : int
            number of target class
        """
        super(YoloHead, self).__init__()
        self.anchors :torch.Tensor = anchors
        self.num_classes : int = num_classes

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] : # type: ignore
        """Summary

        Parameters
        ----------
        x : torch.Tensor
            Convolution feature shape (batch, (5 + num_classes) * num_anchors, conv_height, conv_width)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Return tuple of tensor
                - box_confidence: the confidence that box contain an object
                - box_xy: predicted box xy value
                - box_wh: predicted box wh value
                - box_class_probs: predicted classes probabilities
        """

        num_anchors = len(self.anchors)
        conv_height, conv_width = x.shape[2:4]

        x_shifts, y_shifts = torch.meshgrid([torch.arange(0, conv_width), torch.arange(0, conv_height)])
        shifts = torch.stack((y_shifts.flatten(), x_shifts.flatten())).contiguous()

        shifts = shifts.view((1, 1, 2, conv_height, conv_width)).float().to(x.device)

        x = x.view((-1, num_anchors, self.num_classes + 5, conv_height, conv_width))
        anchors_tensor = self.anchors.view((1, num_anchors, 2, 1, 1)).float().to(x.device)
        conv_dims = torch.tensor((conv_height, conv_width)).view(1, 1, 2, 1, 1).float().to(x.device)

        box_confidence = torch.sigmoid(x[:, :, 4:5, ...])
        box_xy = torch.sigmoid(x[:, :, :2, ...])
        box_wh = torch.exp(x[:, :, 2:4, ...])
        box_class_probs = F.softmax(x[:, :, 5:, ...], dim=2)

        box_xy = (box_xy + shifts) / conv_dims
        box_wh = box_wh * anchors_tensor / conv_dims

        return box_confidence, box_xy, box_wh, box_class_probs
