import requests
import json
import os
import time
import csv

# Config
onnx_url = "http://127.0.0.1:7000/infer_onnx"
pth_url = "http://127.0.0.1:8000/infer_pth"

images_dir = "/Users/zessta/Desktop/triton-server-local-test/speed_nd_accuracy_test/handle"  
pth_model_path = "/Users/zessta/Desktop/yolox_models/V1_models/handle2.pth"

output_csv = "inference_results_2.csv"

# CSV header
header = ["image_name", "onnx_result", "pth_result", "time_onnx", "time_pth"]

with open(output_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for image_name in os.listdir(images_dir):
        if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(images_dir, image_name)

        # ---- ONNX inference ----
        onnx_payload = json.dumps({
            "image_path": image_path,
            "conf_threshold": 0.2,
            "nms_threshold": 0.45
        })
        headers = {"Content-Type": "application/json"}

        start = time.time()
        try:
            onnx_response = requests.post(onnx_url, headers=headers, data=onnx_payload)
            time_onnx = round((time.time() - start) * 1000, 2)  # ms
            onnx_result = onnx_response.text
        except Exception as e:
            time_onnx = None
            onnx_result = f"Error: {e}"

        # ---- PTH inference ----
        pth_payload = json.dumps({
            "image_path": image_path,
            "model_path": pth_model_path,
            "min_threshold": 0.2,
            "problem_id": "handle"
        })

        start = time.time()
        try:
            pth_response = requests.post(pth_url, headers=headers, data=pth_payload)
            time_pth = round((time.time() - start) * 1000, 2)  # ms
            pth_result = pth_response.text
        except Exception as e:
            time_pth = None
            pth_result = f"Error: {e}"

        # ---- Write row to CSV ----
        writer.writerow([image_name, onnx_result, pth_result, time_onnx, time_pth])

print(f"✅ Inference completed. Results saved to {output_csv}")
