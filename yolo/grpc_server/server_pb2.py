# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: server.proto

from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='server.proto',
  package='grpc',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x0cserver.proto\x12\x04grpc\"$\n\x04Size\x12\r\n\x05width\x18\x01 \x01(\x05\x12\r\n\x05heigh\x18\x02 \x01(\x05\"J\n\x08Metadata\x12\x18\n\x04size\x18\x01 \x01(\x0b\x32\n.grpc.Size\x12$\n\x0cimage_format\x18\x02 \x01(\x0e\x32\x0e.grpc.FileType\"M\n\x0cImageMessage\x12\x1e\n\x04meta\x18\x02 \x01(\x0b\x32\x0e.grpc.MetadataH\x00\x12\x0f\n\x05image\x18\x01 \x01(\x0cH\x00\x42\x0c\n\ntest_oneof\"U\n\x08Position\x12\x0f\n\x07\x63lassId\x18\x01 \x01(\x03\x12\x0c\n\x04left\x18\x02 \x01(\x03\x12\x0b\n\x03top\x18\x03 \x01(\x03\x12\r\n\x05right\x18\x04 \x01(\x03\x12\x0e\n\x06\x62ottom\x18\x05 \x01(\x03\"5\n\x11PositionsResponse\x12 \n\x08position\x18\x01 \x03(\x0b\x32\x0e.grpc.Position*\x1c\n\x08\x46ileType\x12\x07\n\x03PNG\x10\x00\x12\x07\n\x03JPG\x10\x01\x32R\n\x17ObjectDetectionServices\x12\x37\n\x06\x64\x65tect\x12\x12.grpc.ImageMessage\x1a\x17.grpc.PositionsResponse\"\x00\x62\x06proto3'
)

_FILETYPE = _descriptor.EnumDescriptor(
  name='FileType',
  full_name='grpc.FileType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='PNG', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='JPG', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=357,
  serialized_end=385,
)
_sym_db.RegisterEnumDescriptor(_FILETYPE)

FileType = enum_type_wrapper.EnumTypeWrapper(_FILETYPE)
PNG = 0
JPG = 1



_SIZE = _descriptor.Descriptor(
  name='Size',
  full_name='grpc.Size',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='width', full_name='grpc.Size.width', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='heigh', full_name='grpc.Size.heigh', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=22,
  serialized_end=58,
)


_METADATA = _descriptor.Descriptor(
  name='Metadata',
  full_name='grpc.Metadata',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='size', full_name='grpc.Metadata.size', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image_format', full_name='grpc.Metadata.image_format', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=60,
  serialized_end=134,
)


_IMAGEMESSAGE = _descriptor.Descriptor(
  name='ImageMessage',
  full_name='grpc.ImageMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='meta', full_name='grpc.ImageMessage.meta', index=0,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image', full_name='grpc.ImageMessage.image', index=1,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='test_oneof', full_name='grpc.ImageMessage.test_oneof',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=136,
  serialized_end=213,
)


_POSITION = _descriptor.Descriptor(
  name='Position',
  full_name='grpc.Position',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='classId', full_name='grpc.Position.classId', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='left', full_name='grpc.Position.left', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='top', full_name='grpc.Position.top', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='right', full_name='grpc.Position.right', index=3,
      number=4, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bottom', full_name='grpc.Position.bottom', index=4,
      number=5, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=215,
  serialized_end=300,
)


_POSITIONSRESPONSE = _descriptor.Descriptor(
  name='PositionsResponse',
  full_name='grpc.PositionsResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='position', full_name='grpc.PositionsResponse.position', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=302,
  serialized_end=355,
)

_METADATA.fields_by_name['size'].message_type = _SIZE
_METADATA.fields_by_name['image_format'].enum_type = _FILETYPE
_IMAGEMESSAGE.fields_by_name['meta'].message_type = _METADATA
_IMAGEMESSAGE.oneofs_by_name['test_oneof'].fields.append(
  _IMAGEMESSAGE.fields_by_name['meta'])
_IMAGEMESSAGE.fields_by_name['meta'].containing_oneof = _IMAGEMESSAGE.oneofs_by_name['test_oneof']
_IMAGEMESSAGE.oneofs_by_name['test_oneof'].fields.append(
  _IMAGEMESSAGE.fields_by_name['image'])
_IMAGEMESSAGE.fields_by_name['image'].containing_oneof = _IMAGEMESSAGE.oneofs_by_name['test_oneof']
_POSITIONSRESPONSE.fields_by_name['position'].message_type = _POSITION
DESCRIPTOR.message_types_by_name['Size'] = _SIZE
DESCRIPTOR.message_types_by_name['Metadata'] = _METADATA
DESCRIPTOR.message_types_by_name['ImageMessage'] = _IMAGEMESSAGE
DESCRIPTOR.message_types_by_name['Position'] = _POSITION
DESCRIPTOR.message_types_by_name['PositionsResponse'] = _POSITIONSRESPONSE
DESCRIPTOR.enum_types_by_name['FileType'] = _FILETYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Size = _reflection.GeneratedProtocolMessageType('Size', (_message.Message,), {
  'DESCRIPTOR' : _SIZE,
  '__module__' : 'server_pb2'
  # @@protoc_insertion_point(class_scope:grpc.Size)
  })
_sym_db.RegisterMessage(Size)

Metadata = _reflection.GeneratedProtocolMessageType('Metadata', (_message.Message,), {
  'DESCRIPTOR' : _METADATA,
  '__module__' : 'server_pb2'
  # @@protoc_insertion_point(class_scope:grpc.Metadata)
  })
_sym_db.RegisterMessage(Metadata)

ImageMessage = _reflection.GeneratedProtocolMessageType('ImageMessage', (_message.Message,), {
  'DESCRIPTOR' : _IMAGEMESSAGE,
  '__module__' : 'server_pb2'
  # @@protoc_insertion_point(class_scope:grpc.ImageMessage)
  })
_sym_db.RegisterMessage(ImageMessage)

Position = _reflection.GeneratedProtocolMessageType('Position', (_message.Message,), {
  'DESCRIPTOR' : _POSITION,
  '__module__' : 'server_pb2'
  # @@protoc_insertion_point(class_scope:grpc.Position)
  })
_sym_db.RegisterMessage(Position)

PositionsResponse = _reflection.GeneratedProtocolMessageType('PositionsResponse', (_message.Message,), {
  'DESCRIPTOR' : _POSITIONSRESPONSE,
  '__module__' : 'server_pb2'
  # @@protoc_insertion_point(class_scope:grpc.PositionsResponse)
  })
_sym_db.RegisterMessage(PositionsResponse)



_OBJECTDETECTIONSERVICES = _descriptor.ServiceDescriptor(
  name='ObjectDetectionServices',
  full_name='grpc.ObjectDetectionServices',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=387,
  serialized_end=469,
  methods=[
  _descriptor.MethodDescriptor(
    name='detect',
    full_name='grpc.ObjectDetectionServices.detect',
    index=0,
    containing_service=None,
    input_type=_IMAGEMESSAGE,
    output_type=_POSITIONSRESPONSE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_OBJECTDETECTIONSERVICES)

DESCRIPTOR.services_by_name['ObjectDetectionServices'] = _OBJECTDETECTIONSERVICES

# @@protoc_insertion_point(module_scope)
