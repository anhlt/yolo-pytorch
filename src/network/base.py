from torch import nn
from torch.nn import MaxPool2d
import torch
import torch.nn.functional as F


class Conv2d(nn.Module):
    """docstring for Conv2d"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 relu=True,
                 same_padding=False,
                 bn=False):

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
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BottleneckBlock(nn.Module):
    """docstring for BottleneckBlock"""

    def __init__(self, input_channels: int, outer_filter: int, bottleneck_filter: int):
        super(BottleneckBlock, self).__init__()
        self.input_channels = input_channels
        self.outer_filter = outer_filter
        self.bottleneck_filter = bottleneck_filter

        self.layers = nn.Sequential(
            Conv2d(input_channels, outer_filter, 3, same_padding=True, bn=True),
            Conv2d(outer_filter, bottleneck_filter, 1, same_padding=True, bn=True),
            Conv2d(bottleneck_filter, outer_filter, 3, same_padding=True, bn=True)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class DoubleBottleneckBlock(nn.Module):
    """docstring for DoubleBottleneckBlock"""

    def __init__(self, input_channels: int, outer_filter: int, bottleneck_filter: int):
        super(DoubleBottleneckBlock, self).__init__()
        self.input_channels = input_channels
        self.outer_filter = outer_filter
        self.bottleneck_filter = bottleneck_filter

        self.layers = nn.Sequential(
            BottleneckBlock(input_channels, outer_filter, bottleneck_filter),
            Conv2d(outer_filter, bottleneck_filter, 1, same_padding=True, bn=True),
            Conv2d(bottleneck_filter, outer_filter, 3, same_padding=True, bn=True),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class V3Block(nn.Module):
    """docstring for V3Block"""

    def __init__(self, input_channels: int, bottleneck_filter: int, **kwargs):
        super(V3Block, self).__init__()
        self.input_channels = input_channels
        self.bottleneck_filter = bottleneck_filter

        self.layers = nn.Sequential(
            Conv2d(input_channels, bottleneck_filter, 1, same_padding=True, bn=True),
            Conv2d(bottleneck_filter, input_channels, 3, same_padding=True, bn=True)
        )

    def forward(self, x):
        x1 = self.layers(x)
        out = x + x1

        return out


class DarknetBodyBottom(nn.Module):
    """docstring for DarknetBodyBottom"""

    def __init__(self, **kwargs):
        super(DarknetBodyBottom, self).__init__()

        self.layers = nn.Sequential(
            Conv2d(3, 32, 3, **kwargs),
            MaxPool2d(2),
            Conv2d(32, 64, 3, **kwargs),
            MaxPool2d(2),
            BottleneckBlock(64, 128, 64),
            MaxPool2d(2),
            BottleneckBlock(128, 256, 128),
            MaxPool2d(2),
            DoubleBottleneckBlock(256, 512, 256)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class DarknetBodyHead(nn.Module):
    """docstring for DarknetBodyHead"""

    def __init__(self):
        super(DarknetBodyHead, self).__init__()

        self.layers = nn.Sequential(
            MaxPool2d(2),
            DoubleBottleneckBlock(512, 1024, 512)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class DarknetBody(nn.Module):
    """docstring for DarknetBody"""

    def __init__(self):
        super(DarknetBody, self).__init__()
        kwargs = {
            "same_padding": True,
            "bn": True
        }
        self.layers = nn.Sequential(
            DarknetBodyBottom(**kwargs),
            DarknetBodyHead())

    def forward(self, x):
        x = self.layers(x)
        return x


class DarkNet(nn.Module):
    """docstring for DarkNet"""

    def __init__(self):
        super(DarkNet, self).__init__()
        self.layers = nn.Sequential(
            DarknetBody(),
            nn.Conv2d(1024, 1000, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
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
    """docstring for YoloBody"""

    def __init__(self, num_anchors, num_classes):
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
    """docstring for YoloHead"""

    def __init__(self, anchors, num_classes):
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
        self.anchors = anchors
        self.num_classes = num_classes

    def forward(self, x):
        """Summary

        Parameters
        ----------
        x : torch.Tensor
            Convolution feature shape (batch, (5 + num_classes) * num_anchors, conv_height, conv_width)
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


class Yolo53Bottom(nn.Module):

    def __init__(self):
        super(Yolo53Bottom, self).__init__()
        kwargs = {
            'same_padding': True,
            'bn': True
        }

        self.layers = nn.Sequential(
            nn.Sequential(
                Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, **kwargs),             # 416 x 416 x   3   ->   416 x 416 x  32
                Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, **kwargs),            # 416 x 416 x  32   ->   208 x 208 x  64
                V3Block(input_channels=64, bottleneck_filter=32, **kwargs),                            # 416 x 416 x  32   ->   208 x 208 x  64
                Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, **kwargs),           # 208 x 208 x  64   ->   104 x 104 x 128
                V3Block(input_channels=128, bottleneck_filter=64, **kwargs),                           # 104 x 104 x 128   ->   104 x 104 x 128
                V3Block(input_channels=128, bottleneck_filter=64, **kwargs),                           # 104 x 104 x 128   ->   104 x 104 x 128
                Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, **kwargs),          # 104 x 104 x 128   ->    52 x  52 x 256
                V3Block(input_channels=256, bottleneck_filter=128, **kwargs),                          # 52 x  52 x 256    ->    52 x  52 x 256
                V3Block(input_channels=256, bottleneck_filter=128, **kwargs),                          # 52 x  52 x 256    ->    52 x  52 x 256
                V3Block(input_channels=256, bottleneck_filter=128, **kwargs),                          # 52 x  52 x 256    ->    52 x  52 x 256
                V3Block(input_channels=256, bottleneck_filter=128, **kwargs),                          # 52 x  52 x 256    ->    52 x  52 x 256
                V3Block(input_channels=256, bottleneck_filter=128, **kwargs),                          # 52 x  52 x 256    ->    52 x  52 x 256
                V3Block(input_channels=256, bottleneck_filter=128, **kwargs),                          # 52 x  52 x 256    ->    52 x  52 x 256
                V3Block(input_channels=256, bottleneck_filter=128, **kwargs),                          # 52 x  52 x 256    ->    52 x  52 x 256
                V3Block(input_channels=256, bottleneck_filter=128, **kwargs),                          # 52 x  52 x 256    ->    52 x  52 x 256
            ),
            nn.Sequential(

                Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, **kwargs),          # 52 x  52 x 256    ->    26 x  26 x 512
                V3Block(input_channels=512, bottleneck_filter=256, **kwargs),                          # 26 x  26 x 512    ->    26 x  26 x 512
                V3Block(input_channels=512, bottleneck_filter=256, **kwargs),                          # 26 x  26 x 512    ->    26 x  26 x 512
                V3Block(input_channels=512, bottleneck_filter=256, **kwargs),                          # 26 x  26 x 512    ->    26 x  26 x 512
                V3Block(input_channels=512, bottleneck_filter=256, **kwargs),                          # 26 x  26 x 512    ->    26 x  26 x 512
                V3Block(input_channels=512, bottleneck_filter=256, **kwargs),                          # 26 x  26 x 512    ->    26 x  26 x 512
                V3Block(input_channels=512, bottleneck_filter=256, **kwargs),                          # 26 x  26 x 512    ->    26 x  26 x 512
                V3Block(input_channels=512, bottleneck_filter=256, **kwargs),                          # 26 x  26 x 512    ->    26 x  26 x 512
                V3Block(input_channels=512, bottleneck_filter=256, **kwargs),                          # 26 x  26 x 512    ->    26 x  26 x 512
            ),

            nn.Sequential(
                Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, **kwargs),         # 26 x  26 x 512    ->    13 x  13 x1024
                V3Block(input_channels=1024, bottleneck_filter=512, **kwargs),                         # 13 x  13 x1024    ->    13 x  13 x1024
                V3Block(input_channels=1024, bottleneck_filter=512, **kwargs),                         # 13 x  13 x1024    ->    13 x  13 x1024
                V3Block(input_channels=1024, bottleneck_filter=512, **kwargs),                         # 13 x  13 x1024    ->    13 x  13 x1024
                V3Block(input_channels=1024, bottleneck_filter=512, **kwargs),                         # 13 x  13 x1024    ->    13 x  13 x1024

                Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, **kwargs),         # 13 x  13 x1024    ->    13 x  13 x 512
                Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, **kwargs),         # 13 x  13 x 512    ->    13 x  13 x1024
                Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, **kwargs),         # 13 x  13 x1024    ->    13 x  13 x 512
                Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, **kwargs),         # 13 x  13 x 512    ->    13 x  13 x1024
                Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, **kwargs),         # 13 x  13 x1024    ->    13 x  13 x 512   (Layer 79)
            )
        )

    def forward(self, x):
        current_result = []
        for layer in self.layers:
            x = layer(x)
            current_result.append(x)
        return tuple(current_result)


class YoloV3Predict(nn.Module):

    def __init__(self, in_channels: int, num_anchors: int, num_classes: int):
        super(YoloV3Predict, self).__init__()
        kwargs = {
            'same_padding': True,
            'bn': True
        }

        self.layers = nn.Sequential(
            Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=3, stride=1, **kwargs),         # 13 x  13 x 512    ->    13 x  13 x1024
            nn.Conv2d(in_channels * 2, num_anchors * (5 + num_classes), 1, 1, 0, bias=False)
        )

    def forward(self, x):
        return self.layers(x)


class YoloV3Head(nn.Module):
    """docstring for Yolo53Head"""

    def __init__(self, num_anchors: int, num_classes: int):
        super(YoloV3Head, self).__init__()
        kwargs = {
            'same_padding': True,
            'bn': True
        }

        self.bottom = Yolo53Bottom()                                                                    # Layer 1 -> Layer 79
        self.predict_1 = YoloV3Predict(512, num_anchors, num_classes)                                   # Layer 80 -> Layer 82

        self.branch_1 = nn.Sequential(
            Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, **kwargs),               # 13 x  13 x 512    ->    13 x  13 x 256 (Layer 84)
            nn.Upsample(scale_factor=2),                                                                # 13 x  13 x 256    ->    26 x  26 x 256 (Layer 85)
        )
        self.branch_2 = nn.Sequential(
            Conv2d(in_channels=768, out_channels=256, kernel_size=1, stride=1, **kwargs),               # 26 x  26 x 768    ->    26 x  26 x 256 (Layer 87)
            Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, **kwargs),               # 26 x  26 x 256    ->    26 x  26 x 512 (Layer 88)
            Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, **kwargs),               # 26 x  26 x 512    ->    26 x  26 x 256 (Layer 89)
            Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, **kwargs),               # 26 x  26 x 256    ->    26 x  26 x 512 (Layer 90)
            Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, **kwargs),               # 26 x  26 x 512    ->    26 x  26 x 256 (Layer 91)
        )

        self.predict_2 = YoloV3Predict(256, num_anchors, num_classes)                                   # Layer 92 -> Layer 94

        # TODO: Insert Detection layer 2

        self.branch_3 = nn.Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, **kwargs),               # 26 x  26 x 256->    26 x  26 x 128 (Layer 96)
            nn.Upsample(scale_factor=2),                                                                # 26 x  26 x 128->    52 x  52 x 128 (Layer 97)
        )

        self.branch_4 = nn.Sequential(
            Conv2d(in_channels=384, out_channels=128, kernel_size=1, stride=1, **kwargs),               # 52 x  52 x 384    ->    52 x  52 x 128 (Layer 99)
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, **kwargs),               # 52 x  52 x 128    ->    52 x  52 x 256 (Layer 100)
            Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, **kwargs),               # 52 x  52 x 256    ->    52 x  52 x 128 (Layer 101)
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, **kwargs),               # 52 x  52 x 128    ->    52 x  52 x 256 (Layer 102)
            Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, **kwargs),               # 52 x  52 x 256    ->    52 x  52 x 128 (Layer 103)
        )

        self.predict_3 = YoloV3Predict(128, num_anchors, num_classes)                                   # Layer 92 -> Layer 94

    def forward(self, x):
        out_36, out_61, out_79 = self.bottom(x)
        predict_1 = self.predict_1(out_79)
        out_85 = self.branch_1(out_79)
        out_86 = torch.cat((out_85, out_61), 1)
        out_91 = self.branch_2(out_86)
        predict_2 = self.predict_2(out_91)
        out_97 = self.branch_3(out_91)
        out_98 = torch.cat((out_97, out_36), 1)
        out_103 = self.branch_4(out_98)
        predict_3 = self.predict_3(out_103)
        return predict_1, predict_2, predict_3
