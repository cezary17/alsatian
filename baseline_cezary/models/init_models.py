from collections import OrderedDict

import torch
from torch import nn

from baseline_cezary.models.custom_mobilenet import mobilenet_v2, MobileNet_V2_Weights
from baseline_cezary.models.custom_resnet import *
from baseline_cezary.models.efficientnet import efficientnet_v2_s, efficientnet_v2_l, EfficientNet_V2_L_Weights, \
    EfficientNet_V2_S_Weights
from baseline_cezary.models.sequential_bert import get_sequential_bert_model
from baseline_cezary.models.sequential_hf_dinov2 import get_sequential_dinov2_model
from baseline_cezary.models.sequential_hf_vit import get_sequential_vit_model
from baseline_cezary.models.split_indices import SPLIT_INDEXES
from baseline_cezary.models.vision_transformer import Encoder, vit_b_16, vit_b_32, vit_l_16, vit_l_32, ViT_L_32_Weights, \
    ViT_L_16_Weights, ViT_B_32_Weights, ViT_B_16_Weights
from baseline_cezary.util.init_hf_models import initialize_hf_model, MICROSOFT_RESNETS, \
    get_sequential_microsoft_resnet, GOOGLE_VIT_BASE_PATCH16_224_IN21K, DINO_V2_MODELS
from baseline_cezary.util.model_names import *
from baseline_cezary.util.model_operations import transform_to_sequential, split_model_in_two
from baseline_cezary.util.quantization import apply_quantization

DUMMY_TWO_BLOCK_MODEL = "dummy_two_block_model"

FEATURE_ONLY_MODELS = DINO_V2_MODELS + [GOOGLE_VIT_BASE_PATCH16_224_IN21K] + MICROSOFT_RESNETS + [DUMMY_TWO_BLOCK_MODEL]


def initialize_model(model_name, pretrained=False, new_num_classes=None, features_only=False, sequential_model=False,
                     freeze_feature_extractor=False, hf_base_model_id=None, hf_model_id=None,
                     hf_cache_dir=None):
    if hf_model_id is not None:
        model, model_name = initialize_hf_model(hf_base_model_id, hf_model_id, hf_cache_dir)
        if (new_num_classes is not None or features_only is True or sequential_model is False
                or freeze_feature_extractor is True):
            raise NotImplemented
    else:
        # init base model
        if model_name == RESNET_18:
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) if pretrained else resnet18()
        elif model_name == RESNET_34:
            model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1) if pretrained else resnet34()
        elif model_name == RESNET_50:
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1) if pretrained else resnet50()
        elif model_name == RESNET_101:
            model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1) if pretrained else resnet101()
        elif model_name == RESNET_152:
            model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1) if pretrained else resnet152()
        elif model_name == MOBILE_V2:
            model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1) if pretrained else mobilenet_v2()
        elif model_name == VIT_B_16:
            model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1) if pretrained else vit_b_16()
        elif model_name == VIT_B_32:
            model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1) if pretrained else vit_b_32()
        elif model_name == VIT_L_16:
            model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1) if pretrained else vit_l_16()
        elif model_name == VIT_L_32:
            model = vit_l_32(weights=ViT_L_32_Weights.IMAGENET1K_V1) if pretrained else vit_l_32()
        elif model_name == EFF_NET_V2_S:
            model = efficientnet_v2_s(
                weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1) if pretrained else efficientnet_v2_s()
        elif model_name == EFF_NET_V2_L:
            model = efficientnet_v2_l(
                weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1) if pretrained else efficientnet_v2_l()
        elif model_name == BERT and features_only:
            model = get_sequential_bert_model(pretrained=pretrained)
        elif model_name in MICROSOFT_RESNETS and features_only:
            model = get_sequential_microsoft_resnet(model_name)
        elif model_name == GOOGLE_VIT_BASE_PATCH16_224_IN21K and features_only:
            model = get_sequential_vit_model(model_name)
        elif model_name in DINO_V2_MODELS and features_only:
            model = get_sequential_dinov2_model(model_name)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # replace classification layer
        if new_num_classes:
            if model_name in RESNETS:
                num_features = model.fc.in_features
                model.fc = nn.Linear(num_features, new_num_classes)
            elif model_name == MOBILE_V2:
                model.classifier = nn.Sequential(
                    nn.Dropout(p=model.dropout),
                    nn.Linear(model.last_channel, new_num_classes),
                )
            elif model_name in TRANSFORMER_MODELS:
                model.heads = model._heads(model.hidden_dim, new_num_classes, model.representation_size)
            elif model_name in EFF_NETS:
                model.classifier = nn.Sequential(
                    nn.Dropout(p=model.dropout, inplace=True),
                    nn.Linear(model.lastconv_output_channels, new_num_classes),
                )
            elif model_name == BERT:
                model: torch.nn.Sequential = model
                model.append(
                    torch.nn.Sequential(
                        nn.Dropout(p=0.1, inplace=False),
                        nn.Linear(in_features=768, out_features=new_num_classes, bias=True)
                    )
                )
            else:
                raise ValueError(f"Unknown model: {model_name}")

        if freeze_feature_extractor:
            # freeze all parameters later unfreeze specific ones
            for param in model.parameters():
                param.requires_grad = False

            if model_name in RESNETS:
                for param in model.fc.parameters():
                    param.requires_grad = True
            elif model_name == MOBILE_V2:
                for param in model.classifier.parameters():
                    param.requires_grad = True
            elif model_name in TRANSFORMER_MODELS:
                for param in model.heads.parameters():
                    param.requires_grad = True
            elif model_name in EFF_NETS:
                for param in model.classifier.parameters():
                    param.requires_grad = True
            else:
                raise ValueError(f"Unknown model: {model_name}")

        if sequential_model or features_only:
            split_cls = None
            if model_name in TRANSFORMER_MODELS:
                split_cls = [Encoder, torch.nn.Sequential]

            model = transform_to_sequential(model, split_classes=split_cls)

        if features_only and model_name not in FEATURE_ONLY_MODELS:
            split_index = SPLIT_INDEXES[model_name][0]
            first, _ = split_model_in_two(model, split_index)
            model = first

    model.model_name = model_name

    return model


def initialize_and_quantize_model(model_name, pretrained=False, new_num_classes=None, features_only=False,
                                  sequential_model=False, freeze_feature_extractor=False, hf_base_model_id=None,
                                  hf_model_id=None, hf_cache_dir=None, quantization_type="dynamic",
                                  calibration_data=None, backend='fbgemm', mode="int8", trained_snapshot_path=None):
    model = initialize_model(model_name, pretrained, new_num_classes, features_only, sequential_model,
                             freeze_feature_extractor, hf_base_model_id, hf_model_id, hf_cache_dir)
    expected = model.state_dict()
    print("Expected: ", expected.keys())
    print("Length: ", len(expected.keys()))
    sd = torch.load(trained_snapshot_path, map_location=torch.device("cpu"))
    # Remap keys

    print("Found: ", sd.keys())
    print("Length: ", len(sd.keys()))

    # print("New: ", new_sd.keys())
    # print("Length: ", len(new_sd.keys()))
    if model_name == RESNET_18:
        new_sd_one = remap_state_dict_keys_resnet18_one(sd)
        new_sd_two = remap_state_dict_keys_resnet18_two(sd)
        new_sd_three = remap_state_dict_keys_resnet18_three(sd)
        new_sd_four = remap_state_dict_keys_resnet18_four(sd)

        new_sd_list = [new_sd_one, new_sd_two, new_sd_three, new_sd_four]

        for i, new_sd in enumerate(new_sd_list):
            print(f"Trying to load with remapping function {i + 1}")
            try:
                model.load_state_dict(new_sd)
                print(f"Successfully loaded state dict with remapping function {i + 1}")
                break
            except RuntimeError:
                print(f"Remapping function {i + 1} failed")
        else:
            raise RuntimeError("All remapping functions failed to load the state dict.")
        # try:
        #     print("Loading state dict using first remapping function")
        #     model.load_state_dict(new_sd)
        # except RuntimeError as e:
        #     print("First remapping function failed, trying the other one")
        #     new_sd = remap_state_dict_keys_resnet18_two(sd)
        #     model.load_state_dict(new_sd)
    elif model_name == RESNET_152:
        print("Loading state dict using remapping function for resnet152")
        new_sd_one = remap_state_dict_resnet152_one(sd)
        new_sd_two = remap_state_dict_resnet152_two(sd)
        new_sd_three = remap_state_dict_resnet152_three(sd)
        new_sd_four = remap_state_dict_resnet152_four(sd)
        new_sd_five = remap_state_dict_resnet152_five(sd)

        new_sd_list = [new_sd_one, new_sd_two, new_sd_three, new_sd_four, new_sd_five]

        for i, new_sd in enumerate(new_sd_list):
            print(f"Trying to load with remapping function {i + 1}")
            try:
                model.load_state_dict(new_sd)
                print(f"Successfully loaded state dict with remapping function {i + 1}")
                break
            except RuntimeError:
                print(f"Remapping function {i + 1} failed")
        else:
            raise RuntimeError("All remapping functions failed to load the state dict.")

    quantized_model = apply_quantization(model, mode=mode, quantization_type=quantization_type,
                                         calibration_data=calibration_data, backend=backend)

    return model, quantized_model


"""
The state dict are different for some model snapshots, this is the best I came up with to remap the keys since I have
not discovered any pattern when the '1.0.' or '0.10' version appears.

Note: There is always a layer 12 of the resnet model, probably the decoder head that was trained in the pretrained 
model. We ignore it here, hence no else: in the remapping functions, those keys are simply dropped.
"""


def remap_state_dict_keys_resnet18_one(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_key = k
        if new_key.startswith("0."):
            new_key = new_key[2:]  # remove "0." prefix
            new_sd[new_key] = v
        elif new_key.startswith("1.0."):
            new_key = f"11.{new_key[4:]}"  # change "1.0." to "11."
            new_sd[new_key] = v
    return new_sd


def remap_state_dict_keys_resnet18_two(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_key = k
        if new_key.startswith("0."):
            new_key = new_key[2:]  # remove "0." prefix
            new_sd[new_key] = v
        elif new_key.startswith("1.0."):
            new_key = f"10.{new_key[4:]}"  # change "1.0." to "10."
            new_sd[new_key] = v
        elif new_key.startswith("1.1."):
            new_key = f"11.{new_key[4:]}"  # change "1.1." to "11."
            new_sd[new_key] = v
    return new_sd


def remap_state_dict_keys_resnet18_three(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_key = k
        if new_key.startswith("0."):
            new_key = new_key[2:]  # remove "0." prefix
            new_sd[new_key] = v
        elif new_key.startswith("1.0."):
            new_key = f"9.{new_key[4:]}"  # change "1.0." to "10."
            new_sd[new_key] = v
        elif new_key.startswith("1.1."):
            new_key = f"10.{new_key[4:]}"  # change "1.1." to "11."
            new_sd[new_key] = v
        elif new_key.startswith("1.2."):
            new_key = f"11.{new_key[4:]}"
            new_sd[new_key] = v
    return new_sd


def remap_state_dict_keys_resnet18_four(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_key = k
        if new_key.startswith("0."):
            new_key = new_key[2:]  # remove "0." prefix
            new_sd[new_key] = v
        elif new_key.startswith("1.0."):
            new_key = f"8.{new_key[4:]}"  # change "1.0." to "10."
            new_sd[new_key] = v
        elif new_key.startswith("1.1."):
            new_key = f"9.{new_key[4:]}"  # change "1.1." to "11."
            new_sd[new_key] = v
        elif new_key.startswith("1.2."):
            new_key = f"10.{new_key[4:]}"
            new_sd[new_key] = v
        elif new_key.startswith("1.3."):
            new_key = f"11.{new_key[4:]}"
            new_sd[new_key] = v
    return new_sd


"""
Man fuck this
"""


def remap_state_dict_resnet152_one(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_key = k
        if new_key.startswith("0."):
            new_key = new_key[2:]  # remove "0." prefix
            new_sd[new_key] = v
        if new_key.startswith('1.0.'):
            new_key = new_key.replace('1.0.', '44.')
            new_sd[new_key] = v
        elif new_key.startswith('1.1.'):
            new_key = new_key.replace('1.1.', '45.')
            new_sd[new_key] = v
        elif new_key.startswith('1.2.'):
            new_key = new_key.replace('1.2.', '46.')
            new_sd[new_key] = v
        elif new_key.startswith('1.3.'):
            new_key = new_key.replace('1.3.', '47.')
            new_sd[new_key] = v
        elif new_key.startswith('1.4.'):
            new_key = new_key.replace('1.4.', '48.')
            new_sd[new_key] = v
        elif new_key.startswith('1.5.'):
            new_key = new_key.replace('1.5.', '49.')
            new_sd[new_key] = v
        elif new_key.startswith('1.6.'):
            new_key = new_key.replace('1.6.', '50.')
            new_sd[new_key] = v
        elif new_key.startswith('1.7.'):
            new_key = new_key.replace('1.7.', '51.')
            new_sd[new_key] = v
        elif new_key.startswith('1.8.'):
            new_key = new_key.replace('1.8.', '52.')
            new_sd[new_key] = v
        elif new_key.startswith('1.9.'):
            new_key = new_key.replace('1.9.', '53.')
            new_sd[new_key] = v

    return new_sd


def remap_state_dict_resnet152_two(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_key = k
        if new_key.startswith("0."):
            new_key = new_key[2:]  # remove "0." prefix
            new_sd[new_key] = v
        if new_key.startswith('1.0.'):
            new_key = new_key.replace('1.0.', '43.')
            new_sd[new_key] = v
        elif new_key.startswith('1.1.'):
            new_key = new_key.replace('1.1.', '44.')
            new_sd[new_key] = v
        elif new_key.startswith('1.2.'):
            new_key = new_key.replace('1.2.', '45.')
            new_sd[new_key] = v
        elif new_key.startswith('1.3.'):
            new_key = new_key.replace('1.3.', '46.')
            new_sd[new_key] = v
        elif new_key.startswith('1.4.'):
            new_key = new_key.replace('1.4.', '47.')
            new_sd[new_key] = v
        elif new_key.startswith('1.5.'):
            new_key = new_key.replace('1.5.', '48.')
            new_sd[new_key] = v
        elif new_key.startswith('1.6.'):
            new_key = new_key.replace('1.6.', '49.')
            new_sd[new_key] = v
        elif new_key.startswith('1.7.'):
            new_key = new_key.replace('1.7.', '50.')
            new_sd[new_key] = v
        elif new_key.startswith('1.8.'):
            new_key = new_key.replace('1.8.', '51.')
            new_sd[new_key] = v
        elif new_key.startswith('1.9.'):
            new_key = new_key.replace('1.9.', '52.')
            new_sd[new_key] = v
        elif new_key.startswith('1.10.'):
            new_key = new_key.replace('1.10.', '53.')
            new_sd[new_key] = v

    return new_sd


def remap_state_dict_resnet152_three(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_key = k
        if new_key.startswith("0."):
            new_key = new_key[2:]  # remove "0." prefix
            new_sd[new_key] = v
        elif new_key.startswith('1.0.'):
            new_key = new_key.replace('1.0.', '40.')
            new_sd[new_key] = v
        elif new_key.startswith('1.1.'):
            new_key = new_key.replace('1.1.', '41.')
            new_sd[new_key] = v
        elif new_key.startswith('1.2.'):
            new_key = new_key.replace('1.2.', '42.')
            new_sd[new_key] = v
        elif new_key.startswith('1.3.'):
            new_key = new_key.replace('1.3.', '43.')
            new_sd[new_key] = v
        elif new_key.startswith('1.4.'):
            new_key = new_key.replace('1.4.', '44.')
            new_sd[new_key] = v
        elif new_key.startswith('1.5.'):
            new_key = new_key.replace('1.5.', '45.')
            new_sd[new_key] = v
        elif new_key.startswith('1.6.'):
            new_key = new_key.replace('1.6.', '46.')
            new_sd[new_key] = v
        elif new_key.startswith('1.7.'):
            new_key = new_key.replace('1.7.', '47.')
            new_sd[new_key] = v
        elif new_key.startswith('1.8.'):
            new_key = new_key.replace('1.8.', '48.')
            new_sd[new_key] = v
        elif new_key.startswith('1.9.'):
            new_key = new_key.replace('1.9.', '49.')
            new_sd[new_key] = v
        elif new_key.startswith('1.10.'):
            new_key = new_key.replace('1.10.', '50.')
            new_sd[new_key] = v
        elif new_key.startswith('1.11.'):
            new_key = new_key.replace('1.11.', '51.')
            new_sd[new_key] = v
        elif new_key.startswith('1.12.'):
            new_key = new_key.replace('1.12.', '52.')
            new_sd[new_key] = v
        elif new_key.startswith('1.13.'):
            new_key = new_key.replace('1.13.', '53.')
            new_sd[new_key] = v

    return new_sd


def remap_state_dict_resnet152_four(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_key = k
        if new_key.startswith("0."):
            new_key = new_key[2:]  # remove "0." prefix
            new_sd[new_key] = v
        elif new_key.startswith('1.0.'):
            new_key = new_key.replace('1.0.', '38.')
            new_sd[new_key] = v
        elif new_key.startswith('1.1.'):
            new_key = new_key.replace('1.1.', '39.')
            new_sd[new_key] = v
        elif new_key.startswith('1.2.'):
            new_key = new_key.replace('1.2.', '40.')
            new_sd[new_key] = v
        elif new_key.startswith('1.3.'):
            new_key = new_key.replace('1.3.', '41.')
            new_sd[new_key] = v
        elif new_key.startswith('1.4.'):
            new_key = new_key.replace('1.4.', '42.')
            new_sd[new_key] = v
        elif new_key.startswith('1.5.'):
            new_key = new_key.replace('1.5.', '43.')
            new_sd[new_key] = v
        elif new_key.startswith('1.6.'):
            new_key = new_key.replace('1.6.', '44.')
            new_sd[new_key] = v
        elif new_key.startswith('1.7.'):
            new_key = new_key.replace('1.7.', '45.')
            new_sd[new_key] = v
        elif new_key.startswith('1.8.'):
            new_key = new_key.replace('1.8.', '46.')
            new_sd[new_key] = v
        elif new_key.startswith('1.9.'):
            new_key = new_key.replace('1.9.', '47.')
            new_sd[new_key] = v
        elif new_key.startswith('1.10.'):
            new_key = new_key.replace('1.10.', '48.')
            new_sd[new_key] = v
        elif new_key.startswith('1.11.'):
            new_key = new_key.replace('1.11.', '49.')
            new_sd[new_key] = v
        elif new_key.startswith('1.12.'):
            new_key = new_key.replace('1.12.', '50.')
            new_sd[new_key] = v
        elif new_key.startswith('1.13.'):
            new_key = new_key.replace('1.13.', '51.')
            new_sd[new_key] = v
        elif new_key.startswith('1.14.'):
            new_key = new_key.replace('1.14.', '52.')
            new_sd[new_key] = v
        elif new_key.startswith('1.15.'):
            new_key = new_key.replace('1.15.', '53.')
            new_sd[new_key] = v

    return new_sd


def remap_state_dict_resnet152_five(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_key = k
        if new_key.startswith("0."):
            new_key = new_key[2:]  # remove "0." prefix
            new_sd[new_key] = v
        elif new_key.startswith('1.0.'):
            new_key = new_key.replace('1.0.', '51.')
            new_sd[new_key] = v
        elif new_key.startswith('1.1.'):
            new_key = new_key.replace('1.1.', '52.')
            new_sd[new_key] = v
        elif new_key.startswith('1.2.'):
            new_key = new_key.replace('1.2.', '53.')
            new_sd[new_key] = v

    return new_sd


if __name__ == '__main__':
    bert_model = initialize_model(BERT, features_only=True, sequential_model=True)
    print("test")
