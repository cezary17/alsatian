import argparse
import os

import torch
import torchvision.transforms as transforms
import wandb
from torchvision.datasets import Imagenette

from baseline_cezary.approaches.nn_proxy import linear_proxy
from baseline_cezary.models.init_models import initialize_model, initialize_and_quantize_model
from baseline_cezary.util.model_names import *
from baseline_cezary.util.quantization import apply_quantization

CPU = 'cpu'
CUDA = 'cuda'

DEVICE = CUDA


def load_data_to_device(batch):
    if isinstance(batch, list):
        batch = [batch[0].to(DEVICE), batch[1].to(DEVICE)]
    else:
        batch = batch.to(DEVICE)
    return batch


def extract_features(model, data):
    # extract the features for the model
    config = wandb.config if wandb.run else {}
    batch_size = config.get("batch_size", 128)
    num_workers = config.get("num_workers", 10)

    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    model = model.to(DEVICE)

    all_labels = []
    all_features = []
    total_batches = len(data_loader)

    for i, data in enumerate(data_loader):
        print(f"{i}/{total_batches}")

        # Log progress every 10 batches or at the end
        if i % 10 == 0 or i == total_batches - 1:
            progress = (i + 1) / total_batches
            wandb.log({
                "feature_extraction_progress": progress,
                "current_batch": i + 1,
                "total_batches": total_batches
            })

        # TODO remove this line for full experiments
        if i >= 2:
            break

        if len(data) == 2:
            inputs, labels = data
        else:
            inputs, labels = data[:-1], data[-1]

        inputs = load_data_to_device(inputs)
        with torch.no_grad():
            out = model(inputs)

        all_labels.append(labels)
        all_features.append(out)

    return all_features, all_labels


def score_models(model_list, train_data, test_data):
    results = {}

    for i, model in enumerate(model_list):
        model_type = "original" if i == 0 else "quantized"
        print(f"Scoring model: {model._name} ({model_type})")

        train_features, train_labels = extract_features(model, train_data)
        test_features, test_labels = extract_features(model, test_data)
        loss, acc = linear_proxy(train_features, train_labels, test_features, test_labels, 10, DEVICE)

        print('Loss: {:.4f}, Acc: {:.4f}'.format(loss, acc))

        # Store results for logging
        results[f"{model_type}_test_loss"] = loss
        results[f"{model_type}_test_accuracy"] = acc
        results[f"{model_type}_model_name"] = model._name

        # Get model parameter count if available
        if hasattr(model, 'parameters'):
            param_count = sum(p.numel() for p in model.parameters())
            results[f"{model_type}_parameters"] = param_count

    # Log all results at once
    wandb.log(results)

    # Calculate and log compression metrics if we have both models
    if "original_test_accuracy" in results and "quantized_test_accuracy" in results:
        accuracy_retention = results["quantized_test_accuracy"] / results["original_test_accuracy"]
        wandb.log({"accuracy_retention": accuracy_retention})

        if "original_parameters" in results and "quantized_parameters" in results:
            compression_ratio = results["original_parameters"] / results["quantized_parameters"]
            wandb.log({"compression_ratio": compression_ratio})


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate model quantization performance.")

    parser.add_argument(
        "--model-architecture",
        type=str,
        default="resnet18",
        choices=[
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
            "mobilenet_v2", "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32",
            "eff_net_v2_s", "eff_net_v2_l", "bert"
        ],
        help="Model architecture to use"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/mount-fs/data/imagenette2",
        help="Path to Imagenette dataset"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for feature extraction"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of workers for data loading"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to use for computation"
    )
    parser.add_argument(
        "--quantization-mode",
        type=str,
        default="dynamic",
        choices=["dynamic", "static"],
        help="Quantization mode to use"
    )
    parser.add_argument(
        "-a", "--agent",
        action="store_true",
        help="Run as a wandb agent"
    )
    parser.add_argument(
        "-s", "--sweepid",
        type=str,
        default=None,
        help="WandB sweep ID"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of runs for agent mode (default: 1)"
    )

    return parser.parse_args()


def run_experiment():
    config = wandb.config

    # Use config values or defaults
    model_architecture = config.get("model_architecture", "resnet18")
    data_root = config.get("data_root", "/mount-fs/data/imagenette2")
    batch_size = config.get("batch_size", 128)
    num_workers = config.get("num_workers", 10)
    device = config.get("device", "cuda")
    quantization_mode = config.get("quantization_mode", "dynamic")

    # Update global DEVICE variable
    global DEVICE
    DEVICE = device

    # Get model name constant
    model_name_mapping = {
        "resnet18": RESNET_18,
        "resnet34": RESNET_34,
        "resnet50": RESNET_50,
        "resnet101": RESNET_101,
        "resnet152": RESNET_152,
        "mobilenet_v2": MOBILE_V2,
        "vit_b_16": VIT_B_16,
        "vit_b_32": VIT_B_32,
        "vit_l_16": VIT_L_16,
        "vit_l_32": VIT_L_32,
        "eff_net_v2_s": EFF_NET_V2_S,
        "eff_net_v2_l": EFF_NET_V2_L,
        "bert": BERT
    }

    if model_architecture not in model_name_mapping:
        raise ValueError(f"Unsupported model architecture: {model_architecture}")

    model_name = model_name_mapping[model_architecture]

    if quantization_mode == "dynamic":
        model, quantized_model = initialize_and_quantize_model(model_name, pretrained=True, features_only=True)
        model.to(DEVICE)
        quantized_model.to(DEVICE)
        model_list = [model, quantized_model]
    elif quantization_mode == "static":
        model, quantized_model = initialize_and_quantize_model(model_name, pretrained=True, features_only=True,
                                              quantization_type="static", calibration_data=None, backend="x86")
        model.to(DEVICE)
        quantized_model.to(DEVICE)
        model_list = [model, quantized_model]
    else:
        raise ValueError(f"Unsupported quantization mode: {quantization_mode}")

    # Setup data transforms
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Load datasets
    train_data = Imagenette(root=data_root, split="train", transform=transform, download=True)
    test_data = Imagenette(root=data_root, split="val", transform=transform, download=True)

    # Log dataset info
    wandb.log({
        "train_dataset_size": len(train_data),
        "test_dataset_size": len(test_data),
        "num_classes": 10,
        "model_architecture": model_architecture,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": device
    })

    # Score models
    score_models(model_list, train_data, test_data)


def run_experiment_standalone(args):
    wandb.init(
        entity="cezary17",
        project="alsatian-quantized",
        mode="offline" if args.offline else "online",
        config={
            "model_architecture": args.model_architecture,
            "data_root": args.data_root,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": args.device,
        },
    )
    run_experiment()


def run_experiment_agent():
    wandb.init()
    run_experiment()


def main(args):
    if args.agent:
        wandb.agent(
            sweep_id=args.sweepid,
            function=run_experiment_agent,
            project="alsatian-quantized",
            entity="cezary17",
            count=args.count,
        )
    else:
        run_experiment_standalone(args)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
