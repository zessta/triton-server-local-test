import torch
import torchvision.models as models

# Get pretrained resnet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# Dummy input
dummy = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(model, dummy, "resnet18.onnx",
                  input_names=["input"],
                  output_names=["output"],
                  dynamic_axes={"input": {0: "batch"},
                                "output": {0: "batch"}})
