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

    return model


if __name__ == '__main__':
    bert_model = initialize_model(BERT, features_only=True, sequential_model=True)
    print("test")
