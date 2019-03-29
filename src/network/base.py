from torch import nn


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
                              kernel_size, stride, padding=padding)
        nn.init.xavier_normal_(self.conv.weight)
        self.bn = nn.BatchNorm2d(
            out_channels, eps=0.001,
            momentum=0,
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

    def __init__(self, outer_filter: int, bottleneck_filter: int):
        super(BottleneckBlock, self).__init__()
        self.outer_filter = outer_filter
        self.bottleneck_filter = bottleneck_filter
