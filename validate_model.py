# Copyright 2024 by Andrey Ignatov. All Rights Reserved.

import torch
from torchvision import datasets, transforms
from torch.utils import data
import numpy as np

np.random.seed(42)

NUM_CLASSES = 9
BATCH_SIZE = 64

PATH_TO_TEST_DATASET = "data/CRC-VAL-HE-7K/"

from model import DeepCMorph

# Modify the model training parameters below:

from sklearn.metrics import accuracy_score, balanced_accuracy_score


if __name__ == '__main__':

    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")

    model = DeepCMorph(num_classes=NUM_CLASSES)
    model.load_weights(dataset="CRC")

    model.to(device)
    model.eval()

    test_transforms = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.ImageFolder(PATH_TO_TEST_DATASET, transform=test_transforms)
    test_dataloader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=False, drop_last=False)

    TEST_SIZE = len(test_dataloader.dataset)
    print("Test size:", TEST_SIZE)

    print("Running Evaluation...")

    accuracy_total = 0.0

    targets_array = []
    predictions_array = []

    with torch.no_grad():

        test_iter = iter(test_dataloader)
        for j in range(len(test_dataloader)):

            image, labels = next(test_iter)
            image = image.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            predictions = model(image)
            _, predictions = torch.max(predictions.data, 1)

            predictions = predictions.detach().cpu().numpy()
            targets = labels.detach().cpu().numpy()

            for k in range(targets.shape[0]):

                target = targets[k]
                predicted = predictions[k]

                targets_array.append(target)
                predictions_array.append(predicted)

        print("Accuracy: " + str(accuracy_score(targets_array, predictions_array)))
        print("Balanced Accuracy: " + str(balanced_accuracy_score(targets_array, predictions_array)))

