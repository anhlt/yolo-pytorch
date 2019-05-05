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

import torch
from src.utils.display.images import result_show
from src.network.yolo import Yolo
from src.config import VOC_ANCHORS
from PIL import Image

print("test")



classes = ['__background__', 'Tree', 'Car', 'Flower']

# +
from src.network.base import DarkNet, DarknetBody, YoloBody

model = Yolo(VOC_ANCHORS, classes)
model.load_state_dict(torch.load('./save_model/model_16.pth'))
model.cuda()
# -

with torch.no_grad():
    img_path = './test_img/panji.jpg'
    img = Image.open(img_path)
    boxes, scores, classes = model.predict(img_path, score_threshold=0.3, iou_threshold=0.2)
result_show(img, boxes, classes, scores,  ['__background__', 'Tree', 'Car', 'Flower'])


