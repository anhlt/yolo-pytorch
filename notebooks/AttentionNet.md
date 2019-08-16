```python
%matplotlib inline
%load_ext autoreload
%autoreload 2
%cd /data/
```

```python
from src.network.attention import ConvAttention2DSize3
from src.network.base import Conv2d, BottleneckBlock, DarknetBody, DoubleBottleneckBlock

from torch import nn
from torch.nn import MaxPool2d
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import time
```

```python
class DarknetBodyBottom(nn.Module):
    """docstring for DarknetBodyBottom"""

    def __init__(self, **kwargs):
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
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 , 200)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.second_layer(x)
        x = self.third_layer(x)
        x = self.forth_layer(x)
        x = self.fifth_layer(x)
        x = self.sixth_layer(x)
        x = self.seventh_layer(x)
        x = self.eighth_layer(x)
        x = self.nineth_layer(x)
        x = self.global_avg_pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
```

```python
model = DarknetBodyBottom()
model.cuda()
```

```python
train_data = torchvision.datasets.ImageFolder('/data/data/tiny-imagenet-200/train', transform=transforms.Compose([
                           transforms.ToTensor()
                       ]), target_transform=None)
```

```python
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
```

```python

def train(train_loader, model, criterion, optimizer, epoch, **kwargs):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        
        images = images.cuda()
        target = target.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % kwargs["print_freq"] == 0:
            progress.display(i)


def validate(val_loader, model, criterion, **kwargs):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
 
            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % kwargs['print_freq'] == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

```

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-3)
```

```python
for i in range(100):
    train(train_loader, model, criterion, optimizer, i, **{'print_freq': 400})
```

```python

```
