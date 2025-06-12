import torch
from model import PoseBinaryPT

# -------------------- Configuration --------------------
exercise = 'push_up'
model_path = f"weights/{exercise}_binary_best_norm.pth"
onnx_output_path = f"weights/{exercise}_model.onnx"
dim = 26  # Number of input features (keypoints)
dummy_input = torch.randn(1, 1, dim)  # [batch=1, seq_len=1, features=26]

# -------------------- Load Model --------------------
model = PoseBinaryPT(dim=dim, heads=2, enc_layers=1, alpha=0.5)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()
print("✅ Model loaded")

# -------------------- Export to ONNX --------------------
torch.onnx.export(
    model,
    dummy_input,
    onnx_output_path,
    export_params=True,
    opset_version=16,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["logits", "features"]
)

print(f"✅ ONNX model exported to: {onnx_output_path}")
