# Diabetic Retinopathy Prediction
Deep learning solution, based on CNN: **Resnet18**, to predict Diabetic Retinopathy using retinal images.


## Diabetic Retinopathy Overview
Diabetic Retinopathy is a complication of diabetes mellitus, in which glucose blocks blood vessels that feed the eye. It causes swelling and leaking of blood/fluid that can cause eye injury. DR causes abnormal growth of blood which leads to vessels, bleeding, vision loss, and eventuallly blindness in advanced phases. DR amounts to 2.6% of causes of blindness.

Diabetic Retinopathy includes four types of lesions:
### 1. Microaneurysms (MA):

It is an early stage of DR. It is caracterized by small **red round dots** on the retina virtue of vessel wall weakness. The dots have sharp magins (size <= 125μm).

### 2. Haemorrhages (HM):

Caracterized by **large spots** on the retina. It has irregular margin sizes (>= 125μm), and can be split into two categories: 2 categories:
- **Flame**: superficial spots
- **Blot**: deep spots

### 3. Hard exudates:

It is a consequence of plasma leakage. It can be caracterized by **yellow spots**, and it spans the outer retina layers. It also has **sharp margins**.

### 4. Soft exudates:

It is a consequence of nerve fibre swelling and visibly seen as **white ovals** on the retina. 

## Datasets
In this project, we are using the [APTOS](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) dataset, which contains decent quality retinal images.

## Image Enhancements
After exploring the existing datasets of images for the DR detection, we found out that the quality of images is too variant. So, we had to perform image enhancement before defining the model's architecture. For this, we referenced to some research papers [3], to enhance image quality in a global manner.

## Models
In this project, we are training on Resnet18. We can set the `freeze_backbone = True`: Transfer learning, or `freeze_backbone = False`: Fine Tune.

## Setup and Run
To setup and run the project locally, please follow the following steps:
```sh
$ git clone https://github.com/waterflow80/diabetic-retinopathy.git
$ cd diabetic-retinopathy
$ python3.11 -m venv venv
$ . venv/bin/activate
$ pip install -e .

# Run
$ drdetector --mode train --data_path ./data/train/  # Train Mode
$ drdetector --mode test --data_path ./data/test/ --model_path ./models/cnn_resnet18_freeze_backbone_False.pth  # Test Mode
```

## Evaluation
Using **Resnet18**
### Using original Images:

![Screenshot from 2025-01-12 00-35-20](https://github.com/user-attachments/assets/56c337b0-ab76-4f60-af9f-87cade25cb2d)

![cnn_resnet18_freeze_backbone_False_non_preprocessed](https://github.com/user-attachments/assets/dc37ee15-a4ac-423c-94aa-daa053947dd9)

### Using Enhanced Images:

![Screenshot from 2025-01-12 00-35-32](https://github.com/user-attachments/assets/27625fa3-db53-40e2-b8c5-3b4f73b112ab)

![cnn_resnet18_freeze_backbone_False_preprocessed](https://github.com/user-attachments/assets/62951835-c6f2-4a84-93f4-7feb1b1dbd0e)

## References
[1]. https://www.mdpi.com/2076-3417/13/19/10760
https://www.researchgate.net/publication/359414263_Deep_Learning_Techniques_for_Diabetic_Retinopathy_Classification_A_Survey
https://www.researchgate.net/publication/340896792_Deep_neural_networks_to_predict_diabetic_retinopathy

[2]. https://aravinda-gn.medium.com/how-to-split-image-dataset-into-train-validation-and-test-set-5a41c48af332

[3]. https://www.mdpi.com/2076-3417/13/19/10760

# Run Problems
Useful links:
- https://github.com/peterjc/backports.lzma?tab=readme-ov-file#usage

Pytorch / cuda compatibility:
- https://gist.github.com/Hansimov/c2c82c9512245758398bc8b48c2789c0
- https://download.pytorch.org/whl/nightly/cu126 (for Cuda 12.6)
- https://www.reddit.com/r/pytorch/comments/11z9vkf/comment/jm5g09k/?utm_source=share&utm_medium=web2x&context=3 (informative)

To install the correct pytroch version for CUDA 12.6:
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
```
