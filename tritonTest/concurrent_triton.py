import asyncio
import httpx
import time
import numpy as np
import pandas as pd

async def hit_triton(session, url, data):
    start = time.time()
    response = await session.post(url, json=data)
    end = time.time()
    return end - start, response.json()

async def run_concurrent(url, concurrency):
    # Prepare dummy input in Triton JSON format
    data = {
        "inputs": [
            {
                "name": "input", 
                "shape": [1, 3, 224, 224], 
                "datatype": "FP32", 
                "data": np.random.rand(1,3,224,224).astype(float).tolist()
            }
        ]
    }

    async with httpx.AsyncClient() as client:
        tasks = [hit_triton(client, url, data) for _ in range(concurrency)]
        results = await asyncio.gather(*tasks)

    # Add concurrency info to results
    formatted_results = []
    for i, (t, res) in enumerate(results):
        formatted_results.append({
            "concurrent_requests": concurrency,
            "request_id": i+1,
            "inference_time_sec": t,
            "output_shape": tuple(res["outputs"][0]["shape"])
        })
    return formatted_results

async def main():
    url = "http://127.0.0.1:8000/v2/models/resnet18/infer"
    concurrency_levels = [5, 10, 15]
    all_results = []

    for level in concurrency_levels:
        print(f"\nHitting {level} concurrent requests...")
        results = await run_concurrent(url, level)
        all_results.extend(results)
        for r in results:
            print(f"Request {r['request_id']}: time={r['inference_time_sec']:.4f}s, output_shape={r['output_shape']}")

    # Save all results to CSV
    df = pd.DataFrame(all_results)
    df.to_csv("triton_async_results.csv", index=False)
    print("\nResults saved to triton_async_results.csv")

asyncio.run(main())
