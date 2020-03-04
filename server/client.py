import os
import grpc
from yolo.grpc_server import server_pb2_grpc
import io
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

from yolo.grpc_server.server_pb2 import FileType, ImageMessage, Metadata, Size

from PIL import Image

dir_path = os.path.dirname(os.path.realpath(__file__))

image_path = os.path.join(dir_path, '..', 'data','example', 'gs5.jpg')



with open(image_path, 'rb') as f:
    file_data = f.read()


img: Image = Image.open(image_path)

imgByteArr = io.BytesIO()
img.save(imgByteArr, format='JPEG')
img_file_in_bytes = imgByteArr.getvalue()


request = ImageMessage(meta=Metadata(size=Size(width=img.width, heigh=img.height), image_format=FileType.JPG), image=img_file_in_bytes)



def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = server_pb2_grpc.ObjectDetectionServicesStub(channel)
        print("-------------- GetFeature --------------")
        result = stub.detect(request)
        print(result)


if __name__ == '__main__':
    logging.basicConfig()
    run()


