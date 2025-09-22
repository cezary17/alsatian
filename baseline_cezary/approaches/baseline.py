import argparse
import os
from typing import Tuple

import torch
import torchvision.transforms as transforms
import wandb
import re
from torchvision.datasets import Imagenette, ImageFolder, StanfordCars, Food101

from baseline_cezary.approaches.nn_proxy import linear_proxy, linear_proxy_quantized
from baseline_cezary.models.init_models import initialize_model, initialize_and_quantize_model
from baseline_cezary.util.model_names import *
from baseline_cezary.util.quantization import apply_quantization

CPU = 'cpu'
CUDA = 'cuda'

DEVICE = CPU


def load_dataset(dataset_name, data_root, transform, split="train"):
    """Load the specified dataset with the given split."""
    if dataset_name == "imagenette" or dataset_name == "imagenette2":
        if split == "train":
            return Imagenette(root=data_root, split="train", transform=transform, download=True)
        else:
            return Imagenette(root=data_root, split="val", transform=transform, download=True)
    elif dataset_name == "stanford-dogs":
        if split == "train":
            return ImageFolder(root=os.path.join(data_root, "prepared_data", "train"), transform=transform)
        else:
            return ImageFolder(root=os.path.join(data_root, "prepared_data", "test"), transform=transform)
    elif dataset_name == "stanford-cars":
        if split == "train":
            return ImageFolder(root=os.path.join(data_root, "car_data", "car_data", "train"), transform=transform)
        else:
            return ImageFolder(root=os.path.join(data_root, "car_data", "car_data", "test"), transform=transform)
    elif dataset_name == "food-101":
        if split == "train":
            return Food101(root=data_root, split="train", transform=transform, download=True)
        else:
            return Food101(root=data_root, split="test", transform=transform, download=True)
    elif dataset_name == "cub-birds-200":
        if split == "train":
            return ImageFolder(root=os.path.join(data_root, "prepared_data", "train"), transform=transform)
        else:
            return ImageFolder(root=os.path.join(data_root, "prepared_data", "test"), transform=transform)
    elif dataset_name == "image-woof":
        if split == "train":
            return ImageFolder(root=os.path.join(data_root, "train"), transform=transform)
        else:
            return ImageFolder(root=os.path.join(data_root, "val"), transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_dataset_info(dataset_name):
    """Get dataset-specific information like number of classes."""
    if dataset_name == "imagenette" or dataset_name == "imagenette2":
        return {"num_classes": 10}
    elif dataset_name == "image-woof":
        return {"num_classes": 10}
    elif dataset_name == "stanford-dogs":
        return {"num_classes": 120}
    elif dataset_name == "food-101":
        return {"num_classes": 101}
    elif dataset_name == "stanford-cars":
        return {"num_classes": 196}
    elif dataset_name == "cub-birds-200":
        return {"num_classes": 200}
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_model_ids_from_path(path: str) -> Tuple[set, set, set]:
    pattern = r'(resnet\d{2,3})-(.+)-(epoch-\d{1,2})\.pth'

    lines = os.listdir(path)

    models = set()
    models_with_tags = set()
    epochs = set()

    for line in lines:
        match = re.search(pattern, line)
        if match:
            model_name = match.group(1)
            models.add(model_name)

            model_tag = match.group(2)
            model_with_tag = f"{model_name}-{model_tag}"
            models_with_tags.add(model_with_tag)

            epoch = match.group(3)
            epochs.add(epoch)

    return models, models_with_tags, epochs


def get_trained_model_path(dataset_name: str, model_and_tag: str, epoch: str):
    base_path = "/mount-fs/trained-snapshots/"
    dataset_path = os.path.join(base_path, dataset_name)

    model_path = f"{model_and_tag}-{epoch}.pth"

    full_path = os.path.join(dataset_path, model_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Model snapshot not found: {full_path}")
    return full_path


def load_data_to_device(batch):
    if isinstance(batch, list):
        batch = [batch[0].to(DEVICE), batch[1].to(DEVICE)]
    else:
        batch = batch.to(DEVICE)
    return batch


def extract_features(model, data, device=None):
    # extract the features for the model
    config = wandb.config if wandb.run else {}
    batch_size = config.get("batch_size", 128)
    num_workers = config.get("num_workers", 10)

    # Use provided device or fallback to global DEVICE
    target_device = device if device is not None else DEVICE

    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    model = model.to(target_device)

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

        if len(data) == 2:
            inputs, labels = data
        else:
            inputs, labels = data[:-1], data[-1]

        # if i > 2:
        #     break

        # Move inputs to the same device as the model
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(target_device)
        elif isinstance(inputs, (list, tuple)):
            inputs = [inp.to(target_device) if isinstance(inp, torch.Tensor) else inp for inp in inputs]

        with torch.no_grad():
            out = model(inputs)

        all_labels.append(labels)
        all_features.append(out)

    return all_features, all_labels


def score_models(model_list, train_data, test_data, num_classes, model_name=""):
    results = {}

    for i, model in enumerate(model_list):
        model_qtype = "original" if i == 0 else "quantized"
        model_id = f"{model_name}_{model_qtype}" if model_name else model_qtype
        print(f"Scoring model: {model.model_name} ({model_qtype})")

        # Debug: Check actual device of model parameters
        actual_device = next(model.parameters()).device
        print(f"Model {model_qtype} actual device: {actual_device}")
        print(f"Target device: {DEVICE}")

        model_device = DEVICE

        train_features, train_labels = extract_features(model, train_data, model_device)
        test_features, test_labels = extract_features(model, test_data, model_device)
        loss, acc = linear_proxy(train_features, train_labels, test_features, test_labels, num_classes, model_device)

        print('Loss: {:.4f}, Acc: {:.4f}'.format(loss, acc))

        results[f"{model_id}_test_loss"] = loss
        results[f"{model_id}_test_accuracy"] = acc
        results[f"{model_id}_model_name"] = model.model_name
        results[f"{model_id}_device"] = model_device

    wandb.log(results)

    if "original_test_accuracy" in results and "quantized_test_accuracy" in results:
        accuracy_retention = results["quantized_test_accuracy"] / results["original_test_accuracy"]
        wandb.log({"accuracy_retention": accuracy_retention})


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
        "--dataset",
        type=str,
        default="imagenette",
        choices=["imagenette", "stanford-dogs", "stanford-cars", "cub-birds-200", "food-101", "image-woof"],
        help="Dataset to use for experiments"
    )
    parser.add_argument(
        "--model-dataset",
        type=str,
        default="stanford-dogs",
        choices=["stanford-dogs", "stanford-cars", "cub-birds-200", "food-101", "image-woof"],
        help="Dataset on which the model was originally trained"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/mount-fs/data/",
        help="Path to dataset directory"
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
        "--backend",
        type=str,
        default="fbgemm",
        choices=["x86", "qnnpack", "fbgemm"],
        help="Quantization backend to use"
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous run"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="WandB run ID to resume"
    )

    return parser.parse_args()


def run_experiment():
    config = wandb.config

    model_architecture = config.get("model_architecture")
    dataset = config.get("dataset")
    model_dataset = config.get("model_dataset")
    data_root = config.get("data_root")
    batch_size = config.get("batch_size")
    num_workers = config.get("num_workers")
    device = config.get("device")

    global DEVICE
    DEVICE = device

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

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Determine full data path based on dataset
    if dataset == "imagenette":
        full_data_root = os.path.join(data_root, "imagenette2")
    elif dataset == "stanford-dogs":
        full_data_root = os.path.join(data_root, "stanford-dogs")
    elif dataset == "stanford-cars":
        full_data_root = os.path.join(data_root, "stanford-cars")
    elif dataset == "cub-birds-200":
        full_data_root = os.path.join(data_root, "cub-birds-200")
    elif dataset == "food-101":
        full_data_root = os.path.join(data_root, "food-101")
    elif dataset == "image-woof":
        full_data_root = os.path.join(data_root, "image-woof")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_data = load_dataset(dataset, full_data_root, transform, split="train")
    test_data = load_dataset(dataset, full_data_root, transform, split="test")

    # Get dataset info
    dataset_info = get_dataset_info(dataset)
    num_classes = dataset_info["num_classes"]

    model_names, model_names_tags, epochs = get_model_ids_from_path(f"/mount-fs/trained-snapshots/{model_dataset}/")

    filtered_model_names_tags = sorted({name for name in model_names_tags if name.startswith(model_architecture)})

    print("Model snapshots found:", filtered_model_names_tags)

    for model_tag in filtered_model_names_tags:
        model_path = get_trained_model_path(model_dataset, model_tag, "epoch-20")
        print("Loading model:", model_path)
        wandb.log({"model_snapshot": model_tag})
        try:
            model, quantized_model = initialize_and_quantize_model(model_name, mode="int8", pretrained=True,
                                                               sequential_model=True, features_only=True,
                                                               trained_snapshot_path=model_path)
        except RuntimeError as e:
            print(f"Error loading model {model_tag}: {e}")
            wandb.log({
                "error": str(e),
            })
            continue

        model.to(DEVICE)
        quantized_model.to(DEVICE)
        model_list = [model, quantized_model]

        wandb.log({
            "train_dataset_size": len(train_data),
            "test_dataset_size": len(test_data),
            "num_classes": num_classes,
            "dataset": dataset,
            "model_architecture": model_architecture,
            "model_snapshot": model_tag,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "device": device
        })

        try:
            score_models(model_list, train_data, test_data, num_classes)
        except RuntimeError as e:
            print(f"Error during model scoring: {e}")
            wandb.log({"error": str(e)})
            continue

    print("Finished all models.")

def run_experiment_standalone(args):
    wandb.init(
        entity="cezary17",
        project="alsatian-quantized",
        config={
            "model_architecture": args.model_architecture,
            "dataset": args.dataset,
            "model_dataset": args.model_dataset,
            "data_root": args.data_root,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": args.device,
            "backend": args.backend,
        },
    )
    run_experiment()


def run_experiment_agent():
    wandb.init()
    run_experiment()

def run_experiment_resume(run_id):
    wandb.init(entity="cezary17", project="alsatian-quantized", id=run_id, resume="must")
    run_experiment()


def main(args):
    torch.multiprocessing.set_sharing_strategy("file_system")
    if args.agent:
        wandb.agent(
            sweep_id=args.sweepid,
            function=run_experiment_agent,
            project="alsatian-quantized",
            entity="cezary17",
            count=args.count,
        )
    elif args.resume:
        run_experiment_resume(args.run_id)
    else:
        run_experiment_standalone(args)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
