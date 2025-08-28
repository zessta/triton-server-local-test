import numpy as np
from tritonclient.http import InferenceServerClient, InferInput
import time

client = InferenceServerClient("localhost:8000")

data = np.random.rand(1, 3, 224, 224).astype(np.float32)
inputs = InferInput("input", data.shape, "FP32")  # Replace "input" with actual input name
inputs.set_data_from_numpy(data)

start_time = time.time()
result = client.infer(model_name="resnet18", inputs=[inputs])
end_time = time.time()

output = result.as_numpy("output")  # Replace "output" with actual output name
print("Output shape:", output.shape)
print("Time:", end_time - start_time)
