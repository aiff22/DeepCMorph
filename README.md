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


<br/>
