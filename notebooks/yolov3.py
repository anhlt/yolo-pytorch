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
batch_size = 16

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

def val(data_val_gen ,model):
    val_loss = AverageMeter()
    with torch.no_grad():
        model.eval()
        for step, blobs in enumerate(data_val_gen):
            batch_tensor, batch_boxes, mask_dict, im_info, img_name = blobs
            batch_tensor = batch_tensor.to(torch.device('cuda'))
            detector_mask_32 = mask_dict['detector_mask_32'].to(torch.device('cuda'))
            detector_mask_16 = mask_dict['detector_mask_16'].to(torch.device('cuda'))
            detector_mask_8 = mask_dict['detector_mask_8'].to(torch.device('cuda'))
            matching_true_boxes_32 = mask_dict['matching_true_boxes_32'].to(torch.device('cuda'))
            matching_true_boxes_16 = mask_dict['matching_true_boxes_16'].to(torch.device('cuda'))
            matching_true_boxes_8 = mask_dict['matching_true_boxes_8'].to(torch.device('cuda'))
            batch_boxes = batch_boxes.to(torch.device('cuda'))
            out_32, out_16, out_8 = model(batch_tensor)
            loss_32 = model.loss(out_32, batch_boxes, detector_mask_32, matching_true_boxes_32)
            loss_16 = model.loss(out_16, batch_boxes, detector_mask_16, matching_true_boxes_16)
            loss_8 = model.loss(out_8, batch_boxes, detector_mask_8, matching_true_boxes_8)
            loss = loss_32 + loss_16 + loss_8
            val_loss.update(loss.item())
    return val_loss


def train(data_gen, data_val_gen ,model, metters, optimizer, lr_scheduler, tensorboard_writer, current_epoch=0):

    steps_per_epoch = len(data_gen)
    model.train()
    train_loss = metters
    start_time = time.time()

    for step, blobs in enumerate(data_gen):
        batch_tensor, batch_boxes, mask_dict, im_info, img_name = blobs
        batch_tensor, batch_boxes, mask_dict, im_info, img_name = blobs
        batch_tensor = batch_tensor.to(torch.device('cuda'))
        detector_mask_32 = mask_dict['detector_mask_32'].to(torch.device('cuda'))
        detector_mask_16 = mask_dict['detector_mask_16'].to(torch.device('cuda'))
        detector_mask_8 = mask_dict['detector_mask_8'].to(torch.device('cuda'))
        matching_true_boxes_32 = mask_dict['matching_true_boxes_32'].to(torch.device('cuda'))
        matching_true_boxes_16 = mask_dict['matching_true_boxes_16'].to(torch.device('cuda'))
        matching_true_boxes_8 = mask_dict['matching_true_boxes_8'].to(torch.device('cuda'))
        batch_boxes = batch_boxes.to(torch.device('cuda'))
        
        out_32, out_16, out_8 = model(batch_tensor)
        loss_32 = model.loss(out_32, batch_boxes, detector_mask_32, matching_true_boxes_32)
        loss_16 = model.loss(out_16, batch_boxes, detector_mask_16, matching_true_boxes_16)
        loss_8 = model.loss(out_8, batch_boxes, detector_mask_8, matching_true_boxes_8)
        loss = loss_32 + loss_16 + loss_8
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        train_loss.update(loss.item())
        del batch_tensor

        current_step = current_epoch * steps_per_epoch + step
        if step % 100 == 10:
            print("epochs time %s" % (time.time() - start_time))
            start_time = time.time()
            tensorboard_writer.add_scalar("loss", train_loss.avg, (current_epoch * steps_per_epoch) + step)
            log_text = 'epoch: %d : step %d,  loss: %.4f at %s' % (
                current_epoch + 1, step , train_loss.avg, datetime.now().strftime('%m/%d_%H:%M'))
            print(log_text)


        if step % 500 == 10:
            print("Validate")
            val_loss = val(data_val_gen, model)
            log_text = 'epoch: %d : step %d,  val_loss: %.4f at %s' % (
                current_epoch + 1, step , val_loss.avg, datetime.now().strftime('%m/%d_%H:%M'))
            print(log_text)
            tensorboard_writer.add_scalar("val_loss", val_loss.avg, (current_epoch * steps_per_epoch) + step)

optimizer = RMSprop(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=0.005)
# optimizer = optim.Adam(model.parameters(), lr = 0.0001, weight_decay=0.00005)
exp_lr_scheduler = StepLR(optimizer, step_size=1000, gamma=0.9)

writer = SummaryWriter("%s/%s_rms_0.005_with_aug" % (TENSORBOARD_PATH , datetime.now().strftime('%m/%d_%H:%M')))
train_loss = AverageMeter()
for i in range(20):
    train(train_data_loader, val_data_loader ,model, train_loss, optimizer, exp_lr_scheduler, writer,i)
    torch.save(model.state_dict(), './save_model/model_%s.pth' % i)


