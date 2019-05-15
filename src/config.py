import os
import numpy as np

# Root directory of project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Use GPU implementation of non-maximum suppression
USE_GPU_NMS = True


FONT_PATH = os.path.abspath(
    os.path.join(ROOT_DIR, 'font', 'Arial-Unicode-Regular.ttf'))


VOC_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

IOU_THRESHOLD = 0.6

LOG_PATH = os.path.join(ROOT_DIR, 'logs')

TENSORBOARD_PATH = os.path.join(ROOT_DIR, 'tensorboard_logs')

RATIOS = [32, 16, 8]
