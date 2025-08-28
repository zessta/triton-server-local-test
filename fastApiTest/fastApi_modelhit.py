from fastapi import FastAPI, Request
import onnxruntime as ort
import numpy as np
import time

app = FastAPI()

# Load ONNX model
session = ort.InferenceSession(
    "/Users/zessta/Desktop/triton-server-local-test/models/resnet18/1/model.onnx"
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

@app.post("/infer")
async def infer(request: Request):
    payload = await request.json()

    # Extract input (similar to Triton’s input format)
    inputs = payload.get("inputs", [])
    if not inputs:
        return {"error": "No inputs provided"}

    inp = inputs[0]
    data = np.array(inp["data"], dtype=np.float32).reshape(inp["shape"])

    # Run inference
    start_time = time.time()
    result = session.run([output_name], {input_name: data})
    end_time = time.time()

    output = result[0]

    return {
        "output_shape": output.shape,
        "output_sample": output.flatten().tolist()[:5],  # return first 5 values
        "inference_time_sec": end_time - start_time
    }
