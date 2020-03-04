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
# %cd /data/
# %pwd

from yolo.network.base import DarkNet, DarknetBody, YoloBody
import torch
from yolo.utils.display.images import result_show
from yolo.network.yolo import Yolo
from yolo.config import VOC_ANCHORS
from PIL import Image
from yolo.utils.process_boxes import im_path_to_tensor, image_to_tensor
from functools import partial
from torchvision import transforms

classes = ['__background__', 'Plant', 'Flower', 'Tree']


# +

transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor()])

func = partial(image_to_tensor, transform=transform)

model = Yolo(VOC_ANCHORS, classes)
model.load_state_dict(torch.load(
    './save_model/model_16.pth', map_location=torch.device('cpu')))
# -

with torch.no_grad():
    img_path = './data/example/gs5.jpg'
    img = Image.open(img_path)
    boxes, scores, classes = model.predict(
        img, func,  score_threshold=0.5, iou_threshold=0.2)
    print(boxes, scores, classes)
result_show(img, boxes, classes, scores,  [
            '__background__', 'Plant', 'Flower', 'Tree'])


