# Copyright 2024 by Andrey Ignatov. All Rights Reserved.

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor


class UpsampleConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=2, relu=False):

        super(UpsampleConvLayer, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        out = self.upconv(x)
        out = self.relu(out)

        return out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class DeepCMorphSegmentationModule(nn.Module):

    def __init__(self, use_skips=False, num_classes=7):

        super(DeepCMorphSegmentationModule, self).__init__()

        net = models.efficientnet_b7(weights=None)

        self.return_nodes = {
            "features.2.0.block.0": "f1",
            "features.3.0.block.0": "f2",
            "features.4.0.block.0": "f3",
            "features.6.0.block.0": "f4",
        }

        self.encoder = create_feature_extractor(net, return_nodes=self.return_nodes)

        for p in self.encoder.parameters():
            p.requires_grad = True

        self.use_skips = use_skips

        self.upsample_1 = UpsampleConvLayer(1344, 512, 2)
        self.upsample_2 = UpsampleConvLayer(512, 256, 2)
        self.upsample_3 = UpsampleConvLayer(256, 128, 2)
        self.upsample_4 = UpsampleConvLayer(128, 64, 2)

        self.conv_1 = DoubleConv(992, 512)
        self.conv_2 = DoubleConv(544, 256)
        self.conv_3 = DoubleConv(320, 128)
        self.conv_4 = DoubleConv(64, 64)

        self.conv_segmentation = nn.Conv2d(64, 1, 3, 1, padding="same")
        self.conv_classification = nn.Conv2d(64, num_classes, 3, 1, padding="same")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        features = self.encoder(x)

        net = self.conv_1(torch.cat((self.upsample_1(features["f4"]), features["f3"]), dim=1))
        net = self.conv_2(torch.cat((self.upsample_2(net), features["f2"]), dim=1))
        net = self.conv_3(torch.cat((self.upsample_3(net), features["f1"]), dim=1))
        net = self.conv_4(self.upsample_4(net))

        predictions_segmentation = self.sigmoid(self.conv_segmentation(net))
        predictions_classification = self.sigmoid(self.conv_classification(net))

        return predictions_segmentation, predictions_classification


class DeepCMorph(nn.Module):

    def __init__(self, num_classes=41, dropout_rate=0.0,
                 freeze_classification_module=False, freeze_segmentation_module=True):

        super(DeepCMorph, self).__init__()

        self.num_classes = num_classes
        self.use_dropout = True if dropout_rate > 0 else False
        self.dropout = nn.Dropout(dropout_rate)

        # Defining nuclei segmentation and classification module
        self.model_preprocessing = DeepCMorphSegmentationModule()

        # Freezing the weights of the segmentation module
        for p in self.model_preprocessing.parameters():
            p.requires_grad = False if freeze_segmentation_module else True

        # Defining the DeepCMorph classification module

        # Using the standard Torchvision EfficientNetB7 implementation
        EfficientNetB7_backbone = models.efficientnet_b7(weights=None)
        self.return_nodes = {"flatten": "features"}

        self.encoder = create_feature_extractor(EfficientNetB7_backbone, return_nodes=self.return_nodes)

        # Changing the number of EfficientNet's input channels from 3 to 11:
        # 3 RGB + 1 nuclei segmentation + 7 nuclei classification feature maps
        self.encoder.features._modules['0'] = nn.Conv2d(11, 64, 3, stride=2, padding=1, bias=False)

        for p in self.encoder.parameters():
            p.requires_grad = False if freeze_classification_module else True

        # Defining the final fully-connected layer producing the predictions
        self.output = nn.Linear(2560, num_classes)

        self.output_41 = nn.Linear(2560, 41)
        self.output_32 = nn.Linear(2560, 32)
        self.output_9 = nn.Linear(2560, 9)

    def forward(self, x, return_features=False, return_segmentation_maps=False):

        nuclei_segmentation_map, nuclei_classification_maps = self.model_preprocessing(x)

        if return_segmentation_maps:
            return nuclei_segmentation_map, nuclei_classification_maps

        x = torch.cat((nuclei_segmentation_map, nuclei_classification_maps, x), dim=1)

        features = self.encoder(x)
        extracted_features = features["features"]

        if return_features:
            return extracted_features

        if self.use_dropout:
            extracted_features = self.dropout(extracted_features)

        if self.num_classes == 41:
            return self.output_41(extracted_features)

        if self.num_classes == 32:
            return self.output_32(extracted_features)

        if self.num_classes == 9:
            return self.output_9(extracted_features)

        return self.output(extracted_features)

    def load_weights(self, dataset=None, path_to_checkpoints=None):

        self = torch.nn.DataParallel(self)

        if dataset is None and path_to_checkpoints is None:
            raise Exception("Please provide either the dataset name or the path to a checkpoint!")

        if path_to_checkpoints is None:

            if dataset == "COMBINED":
                path_to_checkpoints = "pretrained_models/DeepCMorph_Datasets_Combined_41_classes_acc_8159.pth"

            if dataset == "TCGA":
                path_to_checkpoints = "pretrained_models/DeepCMorph_Pan_Cancer_32_classes_acc_8273.pth"

            if dataset == "TCGA_REGULARIZED":
                path_to_checkpoints = "pretrained_models/DeepCMorph_Pan_Cancer_Regularized_32_classes_acc_8200.pth"

            if dataset == "CRC":
                path_to_checkpoints = "pretrained_models/DeepCMorph_NCT_CRC_HE_Dataset_9_classes_acc_9699.pth"

        if path_to_checkpoints is None:
            raise Exception("Please provide a valid dataset name = {'COMBINED', 'TCGA', 'TCGA_REGULARIZED', 'CRC'}")

        missing_keys, unexpected_keys = self.load_state_dict(torch.load(path_to_checkpoints), strict=False)

        print("Model loaded, unexpected keys:", unexpected_keys)

