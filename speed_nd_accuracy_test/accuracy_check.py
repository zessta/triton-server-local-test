import pandas as pd
import json

input_csv = "inference_results.csv"
output_csv = "inference_results_with_match.csv"

def boxes_similar(box1, box2, tolerance=1):
    """
    Compare two bounding boxes and check if they are similar within a pixel tolerance.

    Args:
        box1 (list or tuple): First bounding box in the format [x1, y1, x2, y2].
        box2 (list or tuple): Second bounding box in the format [x1, y1, x2, y2].
        tolerance (int or float, optional): Maximum allowed difference (in pixels) 
                                            for each coordinate. Defaults to 1.

    Returns:
        bool: True if all coordinates differ by no more than the tolerance, 
              False otherwise.

    Example:
        >>> boxes_similar([100, 50, 200, 150], [101, 51, 199, 151])
        True
        >>> boxes_similar([100, 50, 200, 150], [110, 60, 210, 160])
        False
    """
    if len(box1) != 4 or len(box2) != 4:
        return False
    return all(abs(a - b) <= tolerance for a, b in zip(box1, box2))


df = pd.read_csv(input_csv)

accuracy_matches = []

for idx, row in df.iterrows():
    try:
        onnx_data = json.loads(row["onnx_result"])
        pth_data = json.loads(row["pth_result"])

        onnx_boxes = onnx_data.get("boxes", [])
        pth_boxes = pth_data.get("boxes", [])

        match = False
        if onnx_boxes and pth_boxes:
            # Compare first detected box
            match = boxes_similar(onnx_boxes[0], pth_boxes[0], tolerance=1)

    except Exception:
        match = False

    accuracy_matches.append(match)

df["accuracy_match"] = accuracy_matches

df.to_csv(output_csv, index=False)

print(f"Updated CSV saved: {output_csv}")
