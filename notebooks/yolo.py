# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
# %cd /data

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




general_transform = Compose([
    Resize((448, 448)),
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

ds = OpenImage('/data/data/OpenImage/', 'train', general_transform=general_transform, transform=transform)
ds_val = OpenImage('/data/data/OpenImage/', 'validation', general_transform=val_general_transform, transform=val_transform)


train_data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=convert_data, num_workers=4, drop_last=True)
val_data_loader = DataLoader(ds_val, batch_size=batch_size , shuffle=True, collate_fn=convert_data, num_workers=4, drop_last=True)

print(ds.classes)
print(len(ds))
print(ds_val.classes)
print(len(ds_val))


# +
from src.network.base import DarkNet, DarknetBody, YoloBody

model = Yolo(VOC_ANCHORS, ds.classes)
model.cuda()
model.train()
# -

# optimizer = SGD(model.parameters(), lr = 0.0001, momentum=0.9)
# optimizer = RMSprop(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=0.005)
optimizer = Adam(model.parameters(), lr = 0.0001, weight_decay=0.00005)
exp_lr_scheduler = StepLR(optimizer, step_size=1000, gamma=0.9)


def val(data_val_gen ,model):
    val_loss = AverageMeter()
    with torch.no_grad():
        model.eval()
        for step, blobs in enumerate(data_val_gen):
            batch_tensor, batch_boxes, detectors_mask, matching_true_boxes, im_info, img_name = blobs
            batch_tensor = batch_tensor.to(torch.device('cuda'))
            detectors_mask = detectors_mask.to(torch.device('cuda'))
            matching_true_boxes = matching_true_boxes.to(torch.device('cuda'))
            batch_boxes = batch_boxes.to(torch.device('cuda'))
            output = model(batch_tensor)
            loss = model.loss(output, batch_boxes, detectors_mask, matching_true_boxes)
            val_loss.update(loss.item())
    return val_loss


def train(data_gen, data_val_gen ,model, metters, optimizer, lr_scheduler, tensorboard_writer, current_epoch=0):
    
    steps_per_epoch = len(data_gen) 
    model.train()
    train_loss = metters
    start_time = time.time()

    for step, blobs in enumerate(data_gen):
        batch_tensor, batch_boxes, detectors_mask, matching_true_boxes, im_info, img_name = blobs
        batch_tensor = batch_tensor.to(torch.device('cuda'))
        detectors_mask = detectors_mask.to(torch.device('cuda'))
        matching_true_boxes = matching_true_boxes.to(torch.device('cuda'))
        batch_boxes = batch_boxes.to(torch.device('cuda'))
        output = model(batch_tensor)
        loss = model.loss(output, batch_boxes, detectors_mask, matching_true_boxes)
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


writer = SummaryWriter("%s/%s_rms_0.005_with_aug" % (TENSORBOARD_PATH , datetime.now().strftime('%m/%d_%H:%M')))
train_loss = AverageMeter()
for i in range(20):
    train(train_data_loader, val_data_loader ,model, train_loss, optimizer, exp_lr_scheduler, writer,i)
    torch.save(model.state_dict(), './save_model/model_%s.pth' % i)

torch.save(model.state_dict(), './model.pth')

model.eval()

# +
for step, blobs in enumerate(val_data_loader):
    batch_tensor, batch_boxes, detectors_mask, matching_true_boxes, im_info, img_name = blobs
    break
from src.utils.display.images import imshow, result_show

for k in range(batch_tensor.shape[0]):
    current_im_info = im_info[k]
    tmp = batch_boxes[k] * torch.Tensor([current_im_info[0], current_im_info[1], current_im_info[0], current_im_info[1], 1])
    tmp = tmp.numpy()        
    between = tmp[:, 2:4] / 2        
    xy = tmp[:, :2]
    xy_min = xy - between
    xy_max = xy + between
    print(np.hstack((xy_min, xy_max)))
    imshow(batch_tensor[k], gt_boxes=np.hstack((xy_min, xy_max)))
    break

# +
model.train()
batch_tensor = batch_tensor.to(torch.device('cuda'))
detectors_mask = detectors_mask.to(torch.device('cuda'))
matching_true_boxes = matching_true_boxes.to(torch.device('cuda'))
batch_boxes = batch_boxes.to(torch.device('cuda'))

output = model(batch_tensor)
loss = model.loss(output, batch_boxes, detectors_mask, matching_true_boxes)
print(loss.item())
# -

model.eval()
boxes, scores, classes = model.predict("data/example/Red_rose.jpg")



print(boxes, scores, classes)

for k in range(batch_tensor.shape[0]):
    current_im_info = im_info[k]
    tmp = boxes.cpu()
    tmp = tmp.detach().numpy()       
    imshow(batch_tensor[k].cpu(), gt_boxes=tmp)
    break


