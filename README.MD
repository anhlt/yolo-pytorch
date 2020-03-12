

#### Generate Server

```
python -m grpc_tools.protoc --python_out=. --mypy_out=. --grpc_python_out=. -I./ ./server.proto
```