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
from torch.utils.data import DataLoader
from src.utils.datasets.adapter import convert_data_v3
from src.config import IOU_THRESHOLD, TENSORBOARD_PATH,  VOC_ANCHORS, RATIOS
from tensorboardX import SummaryWriter
from datetime import datetime
import time
from torch.optim import SGD, RMSprop, Adam
from torch.optim.lr_scheduler import StepLR
from src.utils.evaluate.metter import AverageMeter
from torch import nn
from src.network.yolov3 import Yolo
from functools import partial


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

val_general_transform = Compose([
    Resize((448, 448)),
    XyToCenter()
])


val_transform = transforms.Compose([
                transforms.ToTensor()
            ])

# +
batch_size = 48

ds = OpenImage('/data/data/SmallDataset/', 'train', general_transform=general_transform, transform=transform)
ds_val = OpenImage('/data/data/SmallDataset/', 'validation', general_transform=val_general_transform, transform=val_transform)

train_data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=partial(lambda p, x: convert_data_v3(x, p), RATIOS), num_workers=4, drop_last=True)
val_data_loader = DataLoader(ds_val, batch_size=batch_size , shuffle=True, collate_fn=partial(lambda p, x: convert_data_v3(x, p), RATIOS), num_workers=4, drop_last=True)

print(ds.classes)
print(len(ds))
print(ds_val.classes)

# +

model = Yolo(VOC_ANCHORS, ds.classes)
model.cuda()
model.train()
print(1)
# -

loader_iter = iter(train_data_loader)


with torch.no_grad():
    blobs = next(loader_iter)
    batch_tensor, batch_boxes, mask_dict, im_info, img_name = blobs
    batch_tensor = batch_tensor.to(torch.device('cuda'))
    batch_boxes = batch_boxes.to(torch.device('cuda'))

    print(batch_boxes.shape)
    
    detector_mask_32 = mask_dict['detector_mask_32'].to(torch.device('cuda'))
    detector_mask_16 = mask_dict['detector_mask_16'].to(torch.device('cuda'))
    detector_mask_8 = mask_dict['detector_mask_8'].to(torch.device('cuda'))
    matching_true_boxes_32 = mask_dict['matching_true_boxes_32'].to(torch.device('cuda'))
    matching_true_boxes_16 = mask_dict['matching_true_boxes_16'].to(torch.device('cuda'))
    matching_true_boxes_8 = mask_dict['matching_true_boxes_8'].to(torch.device('cuda'))


#     detectors_mask = detectors_mask.to(torch.device('cuda'))
#     matching_true_boxes = matching_true_boxes.to(torch.device('cuda'))
#     batch_boxes = batch_boxes.to(torch.device('cuda'))
    out_32, out_16, out_8 = model(batch_tensor)
    loss_32 = model.loss(out_32, batch_boxes, detector_mask_32, matching_true_boxes_32)
    loss_16 = model.loss(out_16, batch_boxes, detector_mask_16, matching_true_boxes_16)
    loss_8 = model.loss(out_8, batch_boxes, detector_mask_8, matching_true_boxes_8)
    print(loss_32, loss_16, loss_8)



