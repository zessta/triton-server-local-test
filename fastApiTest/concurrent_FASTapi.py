import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pandas as pd
import numpy as np

url = "http://127.0.0.1:7000/infer"
concurrent_requests = [5, 10, 15]

# Prepare dummy input in Triton JSON format
def make_payload():
    data = np.random.rand(1, 3, 224, 224).astype(np.float32).tolist()
    return {
        "inputs": [
            {
                "name": "input",  # should match your ONNX model input name
                "shape": [1, 3, 224, 224],
                "datatype": "FP32",
                "data": data
            }
        ]
    }

def hit_model():
    payload = make_payload()
    start = time.time()
    response = requests.post(url, json=payload)
    end = time.time()

    if response.status_code == 200:
        res_json = response.json()
        return {
            "status_code": response.status_code,
            "output_shape": res_json.get("output_shape", None),
            "inference_time_sec": end - start
        }
    else:
        return {
            "status_code": response.status_code,
            "output_shape": None,
            "inference_time_sec": end - start
        }

all_results = []

for n in concurrent_requests:
    print(f"\nHitting {n} concurrent requests:")
    results = []
    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = [executor.submit(hit_model) for _ in range(n)]
        for future in as_completed(futures):
            results.append(future.result())
    
    # Add concurrency info
    for r in results:
        r["concurrent_requests"] = n
    all_results.extend(results)

    # Optional: print results
    for i, r in enumerate(results):
        print(f"Request {i+1}: status={r['status_code']}, "
              f"time={r['inference_time_sec']:.4f}s, "
              f"output_shape={r['output_shape']}")

# Save to CSV
df = pd.DataFrame(all_results)
df.to_csv("inference_results.csv", index=False)
print("\nResults saved to inference_results.csv")
