import torch

def apply_quantization(model: torch.nn.Module, mode="int8"):
    quantized_model = torch.ao.quantization.quantize_dynamic(model=model, dtype=torch.qint8)
    quantized_model._name = f"{model._name}_quantized_{mode}"
    return quantized_model
