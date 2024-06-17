# Copyright 2024 by Andrey Ignatov. All Rights Reserved.

import torch
from torchvision import datasets, transforms
from torch.utils import data
import imageio
import numpy as np

from model import DeepCMorph

np.random.seed(42)

# Modify the target number of classes and the path to the dataset
NUM_CLASSES = 32
PATH_TO_SAMPLE_FOLDER = "data/sample_TCGA_images/"


if __name__ == '__main__':

    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")

    # Defining the model
    model = DeepCMorph(num_classes=NUM_CLASSES)
    # Loading model weights corresponding to the TCGA Pan Cancer dataset
    # Possible dataset values:  TCGA, TCGA_REGULARIZED, CRC, COMBINED
    model.load_weights(dataset="TCGA")

    model.to(device)
    model.eval()

    # Loading test images
    test_transforms = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.ImageFolder(PATH_TO_SAMPLE_FOLDER, transform=test_transforms)
    test_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=False, drop_last=False)
    TEST_SIZE = len(test_dataloader.dataset)

    print("1. Running sample inference")

    with torch.no_grad():

        image_id = 0

        test_iter = iter(test_dataloader)
        for j in range(len(test_dataloader)):

            image, labels = next(test_iter)
            image = image.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Get the predicted class for each input images
            predictions = model(image)
            _, predictions = torch.max(predictions.data, 1)

            predictions = predictions.detach().cpu().numpy()[0]
            targets = labels.detach().cpu().numpy()[0]

            print("Image %d: predicted class: %d, target class: %d" % (image_id, predictions, targets))
            image_id += 1

    print("2. Generating feature maps for sample input images")

    with torch.no_grad():

        feature_maps = np.zeros((TEST_SIZE, 2560))

        image_id = 0

        test_iter = iter(test_dataloader)
        for j in range(len(test_dataloader)):

            image, labels = next(test_iter)
            image = image.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Get feature vector of size 2560 for each input images
            image_features = model(image, return_features=True)

            image_features = image_features.detach().cpu().numpy()[0]
            feature_maps[image_id] = image_features

            print("Image " + str(image_id) + ", generated features:", image_features)
            image_id += 1

        print("Features generated, feature array shape:", feature_maps.shape)

    print("3. Generating segmentation and classification maps for sample images")

    with torch.no_grad():

        feature_maps = np.zeros((TEST_SIZE, 2560))

        image_id = 0

        test_iter = iter(test_dataloader)
        for j in range(len(test_dataloader)):

            image, labels = next(test_iter)
            image = image.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Get predicted segmentation and classification maps for each input images
            nuclei_segmentation_map, nuclei_classification_maps = model(image, return_segmentation_maps=True)

            # Visualizing the predicted segmentation map
            nuclei_segmentation_map = nuclei_segmentation_map.detach().cpu().numpy()[0].transpose(1,2,0) * 255
            nuclei_segmentation_map = np.dstack((nuclei_segmentation_map, nuclei_segmentation_map, nuclei_segmentation_map))

            # Visualizing the predicted nuclei classification map
            nuclei_classification_maps = nuclei_classification_maps.detach().cpu().numpy()[0].transpose(1, 2, 0)
            nuclei_classification_maps = np.argmax(nuclei_classification_maps, axis=2)

            nuclei_classification_maps_visualized = np.zeros((nuclei_classification_maps.shape[0], nuclei_classification_maps.shape[1], 3))
            nuclei_classification_maps_visualized[nuclei_classification_maps == 1] = [255, 0, 0]
            nuclei_classification_maps_visualized[nuclei_classification_maps == 2] = [0, 255, 0]
            nuclei_classification_maps_visualized[nuclei_classification_maps == 3] = [0, 0, 255]
            nuclei_classification_maps_visualized[nuclei_classification_maps == 4] = [255, 255, 0]
            nuclei_classification_maps_visualized[nuclei_classification_maps == 5] = [255, 0, 255]
            nuclei_classification_maps_visualized[nuclei_classification_maps == 6] = [0, 255, 255]

            image = image.detach().cpu().numpy()[0].transpose(1,2,0) * 255

            # Saving visual results
            combined_image = np.hstack((image, nuclei_segmentation_map, nuclei_classification_maps_visualized))
            imageio.imsave("sample_visual_results/" + str(image_id) + ".jpg", combined_image.astype(np.uint8))
            image_id += 1

        print("All visual results saved")
