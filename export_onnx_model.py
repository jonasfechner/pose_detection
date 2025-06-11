import torch
from model import PoseBinaryPT

# -------------------- Configuration --------------------
model_path = "push_up_binary_best.pth"
onnx_output_path = "push_up_model.onnx"
dim = 26  # Input feature size
dummy_seq_len = 1

# -------------------- Load Model --------------------
model = PoseBinaryPT(dim=dim, heads=2, enc_layers=1, alpha=0.5)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()
print("✅ Model loaded")

# -------------------- Dummy Input --------------------
dummy_input = torch.randn(1, dummy_seq_len, dim)  # [batch, seq_len, dim]

# -------------------- Export to ONNX --------------------
torch.onnx.export(
    model,
    dummy_input,
    onnx_output_path,
    export_params=True,
    opset_version=16,  # 🟢 Use >=13 to support aten::unflatten
    do_constant_folding=True,
    input_names=["input"],
    output_names=["logits", "features"],  # update based on your actual output
    dynamic_axes={
        "input": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
        "features": {0: "batch_size"},
    }
)

print(f"✅ ONNX model exported to: {onnx_output_path}")
