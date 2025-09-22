import wandb
import pprint

from baseline_cezary.util.model_names import *


def simplify_sweep_config(sweep_config):
    """
    Parse the sweep configuration and convert any parameters with a single value in 'values'
    into a parameter with a 'value' key and just the value itself.

    Args:
        sweep_config (dict): The original sweep configuration

    Returns:
        dict: Modified sweep configuration with simplified single-value parameters
    """
    # Create a deep copy to avoid modifying the original
    simplified_config = sweep_config.copy()

    # Only process the 'parameters' section
    if "parameters" in simplified_config:
        parameters = simplified_config["parameters"]

        for param_name, param_config in parameters.items():
            # Check if the parameter has a 'values' key with a single element
            if "values" in param_config and len(param_config["values"]) == 1:
                # Replace the 'values' list with a single 'value'
                parameters[param_name] = {"value": param_config["values"][0]}

    return simplified_config


PROJECT_NAME = "alsatian-quantized"

ENTITY_NAME = "cezary17"

SWEEP_CONFIG = {
    "name": "model_quantization_sweep",
    "method": "grid",
    "metric": {"name": "accuracy_retention", "goal": "maximize"},
    "parameters": {
        "model_architecture": {
            "values": ["resnet18"]
        },
        "data_root": {"values": ["/mount-fs/data/"]},
        "dataset": {"values": ["imagenette", "stanford-dogs", "stanford-cars", "cub-birds-200", "food-101", "image-woof"]},
        "model_dataset": {"values": ["stanford-dogs", "stanford-cars", "cub-birds-200", "food-101", "image-woof"]},
        "batch_size": {"values": [128]},
        "num_workers": {"values": [10]},
        "device": {"values": ["cuda"]},  # "cuda", "cpu"
        # "quantization_mode": {"values": ["dynamic"]},  # "dynamic", "static"
        # "backend": {"values": ["x86"]}  # "fbgemm", "qnnpack", "x86"
    },
}

if __name__ == "__main__":
    modified_config = simplify_sweep_config(SWEEP_CONFIG)
    print("Sweep Config:")
    pprint.pprint(modified_config)
    sweep_id = wandb.sweep(sweep=modified_config, entity=ENTITY_NAME, project=PROJECT_NAME)
    print(f"Sweep ID: {sweep_id}")
