from yolo.network.yolo import Yolo

from yolo.config import VOC_ANCHORS
from typing_extensions import Protocol
from PIL import Image
from yolo.grpc_server.server_pb2 import PositionsResponse, Position, ImageMessage
from PIL import Image
from yolo.utils.process_boxes import im_path_to_tensor, image_to_tensor
from functools import partial
from torchvision import transforms
import os
import torch


dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path, '..', 'save_model', 'model_16.pth')


classes = ['__background__', 'Plant', 'Flower', 'Tree']
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor()])

func = partial(image_to_tensor, transform=transform)


class ImagePredictor(Protocol):

    def predict(self, image: Image) -> PositionsResponse: ...


class YoLoPredictor(ImagePredictor):

    def __init__(self):

        self.model = Yolo(VOC_ANCHORS, classes)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        pass

    def predict(self, image: Image) -> PositionsResponse:


        boxes, scores, classes = self.model.predict(image, func,  score_threshold=0.5, iou_threshold=0.2)

        positions : List[Position] = []

        for predicted_box, predicted_class, score in zip(boxes, classes, scores):

            left, top, right, bottom = predicted_box

            print("class: {} : {}{}{}{}".format(predicted_class, left, top, right, bottom))

            positions.append(Position(classId=predicted_class, left=left, right=right, bottom=bottom, top=top))
        return PositionsResponse(position=positions)








