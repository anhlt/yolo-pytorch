# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
# %cd /data/

# +
import torch
from src.utils.datasets.ggimages import OpenImage
from src.utils.datasets.transform import RandomHorizontalFlip, Resize, Compose, XyToCenter
import torchvision.transforms as transforms
from src.utils.display.images import imshow, result_show
from torch.utils.data import DataLoader
from src.utils.datasets.adapter import convert_data
import numpy as np
from src.network.yolo import Yolo
from src.config import VOC_ANCHORS
from src.utils.process_boxes import preprocess_true_boxes
from src.config import IOU_THRESHOLD, TENSORBOARD_PATH
from tensorboardX import SummaryWriter
from datetime import datetime
import time
from torch.optim import SGD, RMSprop, Adam
from torch.optim.lr_scheduler import StepLR
from src.utils.evaluate.metter import AverageMeter
from torch import nn

general_transform = Compose([
    Resize((416, 416)),
    RandomHorizontalFlip(0.3),
    XyToCenter()
])


transform = transforms.Compose([
    transforms.RandomChoice([
        transforms.ColorJitter(hue=.3, saturation=.2),
        transforms.RandomGrayscale(p=0.3),
    ]),
    transforms.ToTensor()
])

# -

batch_size = 48
ds = OpenImage('/data/data/SmallDataset/', 'train', general_transform=general_transform, transform=transform)
train_data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=convert_data, num_workers=4, drop_last=True)

# +
from src.network.base import YoloV3Head

model = YoloV3Head(3 , 3)
model.cuda()
model.train()
print(1)
# -

loader_iter = iter(train_data_loader)


with torch.no_grad():
    blobs = next(loader_iter)
    batch_tensor, batch_boxes, detectors_mask, matching_true_boxes, im_info, img_name = blobs
    batch_tensor = batch_tensor.to(torch.device('cuda'))
    detectors_mask = detectors_mask.to(torch.device('cuda'))
    matching_true_boxes = matching_true_boxes.to(torch.device('cuda'))
    batch_boxes = batch_boxes.to(torch.device('cuda'))
    out = model(batch_tensor)
    for i in out:
        print(i.shape)


