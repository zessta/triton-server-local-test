
# Triton Server Local Test

This repository demonstrates how to run **NVIDIA Triton Inference Server** locally with a sample **ResNet18 ONNX model**. It includes setup instructions, model repository structure, and commands to run Triton with live logging.

---

## Directory Structure

Organize your models in the following layout:

```
models/
  resnet18/
    1/
      model.onnx
    config.pbtxt
```

* `model.onnx`: Exported ONNX model (e.g., ResNet18).
* `config.pbtxt`: Triton model configuration file.

---

## Steps to Run Triton Locally

### 1. Pull the Triton Docker Image

```bash
docker pull nvcr.io/nvidia/tritonserver:25.05-py3
```

### 2. Run Triton Server with Model Repository

```bash
docker run --rm --name triton \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /Users/zessta/Desktop/Triton_test/models:/models \
  nvcr.io/nvidia/tritonserver:25.05-py3 \
  tritonserver --model-repository=/models --log-verbose=1
```

**Ports exposed:**

* **HTTP/REST API:** `8000`
* **gRPC API:** `8001`
* **Metrics (Prometheus):** `8002`

---

## Server Logs

When Triton starts successfully, you should see:

```
Triton has started the gRPC server on port 8001.
Triton has started the HTTP/REST server on port 8000.
Triton has started the metrics service on port 8002.
```

---

## Sending Inference Requests

* Use HTTP/REST API on `localhost:8000`
* You can send requests in **JSON** or **binary** format.

Example Python client:

```python
from tritonclient.http import InferenceServerClient

client = InferenceServerClient("localhost:8000")  
print("Is server live:", client.is_server_live())
print("Is server ready:", client.is_server_ready())
print("Is model ready:", client.is_model_ready("resnet18"))  
```

---

## References

* [Triton Inference Server Documentation](https://github.com/triton-inference-server/server)
* [ONNX Model Zoo](https://github.com/onnx/models)

---
