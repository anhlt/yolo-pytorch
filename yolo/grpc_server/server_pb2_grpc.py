# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import server_pb2 as server__pb2


class ObjectDetectionServicesStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.detect = channel.unary_unary(
        '/grpc.ObjectDetectionServices/detect',
        request_serializer=server__pb2.ImageMessage.SerializeToString,
        response_deserializer=server__pb2.PositionsResponse.FromString,
        )


class ObjectDetectionServicesServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def detect(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_ObjectDetectionServicesServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'detect': grpc.unary_unary_rpc_method_handler(
          servicer.detect,
          request_deserializer=server__pb2.ImageMessage.FromString,
          response_serializer=server__pb2.PositionsResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'grpc.ObjectDetectionServices', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
