from tritonclient.http import InferenceServerClient

client = InferenceServerClient("localhost:8000")  
print("Is server live:", client.is_server_live())
print("Is server ready:", client.is_server_ready())
print("Is model ready:", client.is_model_ready("resnet18"))  