## Histopathological Image Classification with Cell Morphology Aware Deep Neural Networks

<br/>

<img src="http://people.ee.ethz.ch/~ihnatova/demo_deepcmorph/architecture.png"/>

<br/>

#### 1. Overview [[Paper]](https://openaccess.thecvf.com/content/CVPR2024W/CVMI/papers/Ignatov_Histopathological_Image_Classification_with_Cell_Morphology_Aware_Deep_Neural_Networks_CVPRW_2024_paper.pdf)

This repository provides the implementation of the foundation **DeepCMorph CNN model** designed for histopathological image classification and analysis. Unlike the existing models, DeepCMorph explicitly **learns cell morphology**: its segmentation module is trained to identify different cell types and nuclei morphological features.

Key DeepCMorph features:

1. Achieves the state-of-the-art results on the **TCGA**, **NCT-CRC-HE** and **Colorectal cancer (CRC)** datasets
2. Consists of two independent **nuclei segmentation / classification** and **tissue classification** modules
3. The segmentation module is pre-trained on a combination of **8 segmentation datasets**
4. The classification module is pre-trained on the **Pan-Cancer TCGA dataset** (8736 diagnostic slides / 7175 patients)
5. Can be applied to images of **arbitrary resolutions**
6. Can be trained or fine-tuned on **one GPU**

<br/>

#### 2. Prerequisites

- Python: numpy and imageio packages
- [PyTorch + TorchVision](https://pytorch.org/) libraries
- [Optional] Nvidia GPU

<br/>

#### 3. Download Pre-Trained Models

The segmentation module of all pre-trained models is trained on a combination of 8 publicly available nuclei segmentation / classification datasets: **Lizard, CryoNuSeg, MoNuSAC, BNS, TNBC, KUMAR, MICCAI** and **PanNuke** datasets.

| Dataset                     | #Classes | Accuracy | Download Link |
|-----------------------------|----------|----------|---------------|
| Combined [[TCGA](https://zenodo.org/records/5889558) + [NCT_CRC_HE](https://zenodo.org/records/1214456)]       | 41       | 81.59%   |  [Link](https://data.vision.ee.ethz.ch/ihnatova/public/DeepCMorph/DeepCMorph_Datasets_Combined_41_classes_acc_8159.pth)             |
| [TCGA](https://zenodo.org/records/5889558) [Extreme Augmentations] | 32       | 82.00%   |  [Link](https://data.vision.ee.ethz.ch/ihnatova/public/DeepCMorph/DeepCMorph_Pan_Cancer_Regularized_32_classes_acc_8200.pth)             |
| [TCGA](https://zenodo.org/records/5889558) [Moderate Augmentations] | 32       | 82.73%   |  [Link](https://data.vision.ee.ethz.ch/ihnatova/public/DeepCMorph/DeepCMorph_Pan_Cancer_32_classes_acc_8273.pth)             |
| [NCT_CRC_HE](https://zenodo.org/records/1214456)                  | 9        | 96.99%   |  [Link](https://data.vision.ee.ethz.ch/ihnatova/public/DeepCMorph/DeepCMorph_NCT_CRC_HE_Dataset_9_classes_acc_9699.pth)             |

Download the required models and copy them to the ``pretrained_models/`` directory.

<br/>

#### 4. Pre-Trained Model Usage

Integrating the DeepCMorph model into your project is extremely simple. The code below shows how to define, initialize and run the model on sample histopathological images: 


```python
from model import DeepCMorph

# Defining the model and specifying the number of target classes:
# 41 for combined datasets, 32 for TCGA, 9 for CRC
model = DeepCMorph(num_classes=41)

# Loading model weights corresponding to the network trained on combined datasets
# Possible 'dataset' values: TCGA, TCGA_REGULARIZED, CRC, COMBINED
model.load_weights(dataset="COMBINED")

# Get the predicted class for a sample input image
predictions = model(sample_image)
_, predicted_class = torch.max(predictions.data, 1)

# Get feature vector of size 2560 for a sample input image
features = model(sample_image, return_features=True)

# Get predicted segmentation and classification maps for a sample input image
nuclei_segmentation_map, nuclei_classification_maps = model(sample_image, return_segmentation_maps=True)
```

A detailed model usage example is additionally provided in the script ``run_inference.py``. It applies the pre-trained DeepCMorph model to 32 images from the TCGA dataset to generate 1) sample **classification predictions**, 2) **feature maps of dimension 2560** that can be used for classification with the SVM or other stand-alone model, 3) **nuclei segmentation / classification maps** generation and visualization. 

<br/>

#### 5. Fine-Tuning the Model

The following codes are needed to initialize the model for further fine-tuning: 

```python
from model import DeepCMorph

# Defining the model with frozen segmentation module (typical usage)
# All weights of the classification module are trainable
model = DeepCMorph(num_classes=...)

# Defining the model with frozen segmentation and classificaton modules
# Only last fully-connected layer would be trainable
model = DeepCMorph(num_classes=..., freeze_classification_module=True)

# Defining the model with all layers being trainable
model = DeepCMorph(num_classes=..., freeze_segmentation_module=False)
```

<br/>

#### 6. Pre-Trained Model Evaluation

File ``validate_model.py`` contains sample codes needed for model evaluation on the **NCT-CRC-HE-7K** dataset. To check the model accuracy:

1. Download the corresponding model weights
2. Download the [NCT-CRC-HE-7K](https://zenodo.org/records/1214456) dataset and extract it to the ``data`` directory.
3. Run the test script: ```python validate_model.py ```

The provided script can be also easily modified for other datasets.

<br/>


#### 7. Folder structure

>```data/sample_TCGA_images/```        &nbsp; - &nbsp; the folder with sample TCGA images <br/>
>```pretrained_models/```   &nbsp; - &nbsp; the folder with the provided pre-trained DeepCMorph models <br/>
>```sample_visual_results/``` &nbsp; - &nbsp; visualization of the nuclei segmentation and classification maps<br/>

>```model.py```           &nbsp; - &nbsp; DeepCMorph implementation [PyTorch] <br/>
>```train_model.py```     &nbsp; - &nbsp; the script showing model usage on sample histopathological images <br/>
>```validate_model.py```      &nbsp; - &nbsp; the script for model validation on the NCT-CRC-HE-7K dataset <br/>

<br/>

#### 8. License

Copyright (C) 2024 Andrey Ignatov. All rights reserved.

Licensed under the [CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

The code is released for academic research use only.

<br/>

#### 9. Citation

```
@inproceedings{ignatov2024histopathological,
  title={Histopathological Image Classification with Cell Morphology Aware Deep Neural Networks},
  author={Ignatov, Andrey and Yates, Josephine and Boeva, Valentina},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6913--6925},
  year={2024}
}
```
<br/>

#### 10. Any further questions?

```
Please contact Andrey Ignatov (andrey@vision.ee.ethz.ch) for more information
```
