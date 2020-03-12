from concurrent import futures
from yolo.grpc_server import server_pb2_grpc
from yolo.grpc_server.server_pb2 import PositionsResponse, Position, ImageMessage
import time
import grpc
import logging
import io
import os
from PIL import Image
from predict_service import YoLoPredictor

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.info('I told you so')

predictor = YoLoPredictor()


class ObjectDetectionServicesServicer(server_pb2_grpc.ObjectDetectionServicesServicer):

    def __init__(self):
        pass

    def detect(self, request: ImageMessage, context):
        img = Image.open(io.BytesIO(request.image))
        return predictor.predict(img)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server_pb2_grpc.add_ObjectDetectionServicesServicer_to_server(
        ObjectDetectionServicesServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


logging.info('I told you so')  # will not print anything


if __name__ == '__main__':
    logging.basicConfig()
    logging.info('Start Server at [::]:50051')  # will not print anything
    serve()
