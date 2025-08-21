import torch
from torch.utils.data import DataLoader
import os
import sys

from baseline_cezary.approaches.nn_proxy import linear_proxy
from baseline_cezary.models.init_models import initialize_model
from baseline_cezary.util.model_names import RESNET_18

CPU = 'cpu'
CUDA = 'cuda'

DEVICE = CPU


def load_data_to_device(batch):
    if isinstance(batch, list):
        batch = [batch[0].to(DEVICE), batch[1].to(DEVICE)]
    else:
        batch = batch.to(DEVICE)
    return batch


def extract_features(model, data):
    # extract the features for the model
    # TODO change to larger batch size, and more workers
    # data_loader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=True, num_workers=10)
    data_loader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True, num_workers=1)
    model = model.to(DEVICE)

    all_labels = []
    all_features = []

    for i, data in enumerate(data_loader):
        print(f"{i}/{len(data_loader)}")
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
    for model in model_list:
        train_features, train_labels = extract_features(model, train_data)
        test_features, test_labels = extract_features(model, test_data)
        loss, acc = linear_proxy(train_features, train_labels, test_features, test_labels, 10, DEVICE)
        print('Loss: {:.4f}, Acc: {:.4f}'.format(loss, acc))


if __name__ == '__main__':
    model = initialize_model(RESNET_18, pretrained=True, features_only=True)
    model_list = [model]
    from torchvision.datasets import Imagenette
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


    data_root = "./data"
    # imagenette_dir = os.path.join(data_root, "imagenette2")
    # download_flag = not os.path.exists(imagenette_dir)
    # if download_flag:
    train_data = Imagenette(root=data_root, split="train", transform=transform)
    test_data = Imagenette(root=data_root, split="val", transform=transform)
    score_models(model_list, train_data, test_data)
