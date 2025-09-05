import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy


def _apply_dynamic_quantization(model: torch.nn.Module, dtype=torch.qint8):
    """Apply dynamic quantization to the model."""
    return torch.ao.quantization.quantize_dynamic(model=model, dtype=dtype)


def _apply_static_quantization(model: torch.nn.Module, calibration_loader: DataLoader,
                               backend='x86'):
    """Apply static quantization to the model using calibration data."""
    # Set quantization backend
    torch.backends.quantized.engine = backend
    
    # Prepare the model for quantization
    model.eval()
    model_copy = copy.deepcopy(model)
    
    # Set quantization config
    if backend == 'x86':
        qconfig = torch.ao.quantization.get_default_qconfig('x86')
    elif backend == 'qnnpack':
        qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
    else:
        qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    
    model_copy.qconfig = qconfig
    
    # Prepare model for static quantization
    prepared_model = torch.ao.quantization.prepare(model_copy)
    
    # Calibrate with representative data
    with torch.no_grad():
        for batch_idx, data in enumerate(calibration_loader):
            if isinstance(data, (list, tuple)):
                inputs = data[0] if len(data) > 1 else data
            else:
                inputs = data
            
            # Move to CPU for quantization
            if hasattr(inputs, 'cpu'):
                inputs = inputs.cpu()
            
            try:
                prepared_model(inputs)
                # Limit calibration samples for efficiency
                if batch_idx >= 10:
                    break
            except Exception as e:
                print(f"Warning: Calibration failed on batch {batch_idx}: {e}")
                continue
    
    quantized_model = torch.ao.quantization.convert(prepared_model)
    return quantized_model


def apply_quantization(model: torch.nn.Module, mode="int8", quantization_type="dynamic", 
                      calibration_data=None, backend='x86'):
    """
    Apply quantization to a PyTorch model.
    
    Args:
        model: The model to quantize
        mode: Quantization mode ('int8' or 'int16')
        quantization_type: Either 'dynamic' or 'static'
        calibration_data: DataLoader for calibration (required for static quantization)
        backend: Quantization backend ('x86', 'qnnpack', 'fbgemm')
    
    Returns:
        Quantized model
    """
    # Determine dtype based on mode
    dtype = torch.qint8 if mode == "int8" else torch.qint16
    
    # Create a copy of the model to avoid modifying the original
    model_copy = copy.deepcopy(model)
    model_copy.eval()
    
    if quantization_type.lower() == "static":
        if calibration_data is None:
            raise ValueError("Calibration data is required for static quantization")
        quantized_model = _apply_static_quantization(model_copy, calibration_data, backend)
    else:
        quantized_model = _apply_dynamic_quantization(model_copy, dtype)

    # Set name attribute
    quantization_suffix = f"quantized_{quantization_type}_{mode}"
    if hasattr(model, '_name'):
        quantized_model._name = f"{model._name}_{quantization_suffix}"
    else:
        quantized_model._name = f"model_{quantization_suffix}"

    return quantized_model
