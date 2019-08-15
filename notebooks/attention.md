---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import torch
from torch import nn
```

```python
inp = torch.randn(5, 32, 224, 224)
```

```python
class AttentionConv2d(nn.Module):
    """docstring for Conv2d"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes = (3, 5, 7),
                 stride=1,
                 relu=True,
                 same_padding=False,
                 bn=False):

        super(AttentionConv2d, self).__init__()
        self.max_kernel_size = max(kernel_sizes)
        self.number_of_kernel = len(kernel_sizes)
        self.kernel_sizes = kernel_sizes
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, self.max_kernel_size, self.max_kernel_size)) # when groups=1
        nn.init.xavier_normal_(self.weight)

        self.kernels_params = {}
        for kernel_size in kernel_sizes:
            padding = int((kernel_size - 1) / 2) if same_padding else 0
            self.kernels_params["weight_unpadding_%d" % kernel_size] = (self.max_kernel_size - kernel_size) // 2 
            self.kernels_params["unfold_%d" % kernel_size] = nn.Unfold(kernel_size, padding=padding, stride=stride)
        
        self.bn = nn.BatchNorm2d(
            out_channels, eps=0.001,
            affine=True
        ) if bn else None

        self.relu = nn.LeakyReLU(negative_slope=0.1) if relu else None
        print(self.kernels_params)

    def forward(self, x):
        N, C, H, W = x.shape        
        out = torch.empty((N, self.number_of_kernel * self.out_channels, H, W))
        for key, kernel_size in enumerate(self.kernel_sizes):
            weight_unpadding = self.kernels_params["weight_unpadding_%d" % kernel_size]
            current_weight = self.weight[:, :, weight_unpadding:(self.max_kernel_size - weight_unpadding), weight_unpadding:(self.max_kernel_size - weight_unpadding)]
            unfold_module = self.kernels_params["unfold_%d" % kernel_size]
            x1 = unfold_module(x)
            x1 = x1.transpose(1, 2)
            x1 = x1.matmul(current_weight.contiguous().view(current_weight.size(0), -1).t())
            x1 = x1.transpose(1, 2).view(N, -1, H, W)
            out[:, key * self.out_channels:(key+1) *self.out_channels, : , :] = x1
        return out
```

```python
unfold =  AttentionConv2d(32, 32, same_padding=True, bn=True)
```

```python
conv_out = unfold(inp)
```

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MultiHeadAttention', 'ScaledDotProductAttention']


class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )
```

```python
multi_head_attention = MultiHeadAttention(32, 8)
```

```python
N, C , H , W = conv_out.shape
multi_head_in = conv_out.permute(0, 2, 3, 1).contiguous().view(N * H * W, 3, -1)
```

```python
out = multi_head_attention(multi_head_in, multi_head_in ,multi_head_in)
```

```python
out.view(N, H, W, -1).permute(0, 3, 1, 2).shape
```

```python

```
