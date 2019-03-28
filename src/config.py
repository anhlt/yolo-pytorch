import os

# Root directory of project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Use GPU implementation of non-maximum suppression
USE_GPU_NMS = True


FONT_PATH = os.path.abspath(
    os.path.join(ROOT_DIR, 'font', 'Arial-Unicode-Regular.ttf'))
