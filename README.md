# Diabetic Retinopathy Prediction
Deep learning solution, based on CNN: **Resnet18**, to predict Diabetic Retinopathy using retinal images.


## Diabetic Retinopathy Overview
### Diabetic Mellitus
Diabetic Retinopathy is a diabetic complication, in which blood glucose levels tend to increase due to the inability of pancreas to produce sufficient blood insulin. There has been an increasing number of incidents over the last decates: from 108 million in 1980 to 422 million in 2014. It affects liver, heart, kidneys, joints, and eyes, and it is the most prominent reason for blindness.

### Diabetic Retinopathy
Diabetic Retinopathy is a complication of diabetes mellitus, in which glucose blocks blood vessels that feed the eye. It causes swelling and leaking of blood/fluid that can cause eye injury. DR causes abnormal growth of blood which leads to vessels, bleeding, vision loss, and eventuallly blindness in advanced phases. DR amounts to 2.6% of causes of blindness.

Diabetic Retinopathy includes four types of lesions:
#### 1. Microaneurysms (MA):

It is an early stage of DR. It is caracterized by small **red round dots** on the retina virtue of vessel wall weakness. The dots have sharp magins (size <= 125μm).

#### 2. Haemorrhages (HM):

Caracterized by **large spots** on the retina. It has irregular margin sizes (>= 125μm), and can be split into two categories: 2 categories:
- **Flame**: superficial spots
- **Blot**: deep spots

#### 3. Hard exudates:

It is a consequence of plasma leakage. It can be caracterized by **yellow spots**, and it spans the outer retina layers. It also has **sharp margins**.

#### 4. Soft exudates:

It is a consequence of nerve fibre swelling and visibly seen as **white ovals** on the retina. 

## Datasets
In this project, we are using the [APTOS](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) dataset, which contains decent quality retinal images.

## Image Enhancements
After exploring the existing datasets of images for the DR detection, we found out that the quality of images is too variant. So, we had to perform image enhancement before defining the model's architecture. For this, we referenced to some research papers [3], to enhance image quality in a global manner.

## Models
In this project, we are training on Resnet18. We can set the `freeze_backbone = True`: Transfer learning, or `freeze_backbone = False`: Fine Tune.

## Evaluation
We're still working on running the model on the full dataset, using `dvc` and connecting to `Google Drive`. The following is the results of the model's performance on **77 train** images and **20 validation** images:
### Experiment 1:
#### Parameters:
|       Name      |   Value  |
|:---------------:|:--------:|
| backbone        | resnet18 |
| epochs          | 2        |
| freeze_backbone | False    |
| learning_rate   | 1e-05    |
#### Metrics
|       Name      |   Value  |
|:---------------:|:--------:|
| train_loss      | 1.399  |

These results are relatively bad and are only for demostration purposes. Running the model on the full dataset and with a bigger number of epochs requires significant amount of resources and processing time.
## References
[1]. https://www.mdpi.com/2076-3417/13/19/10760
https://www.researchgate.net/publication/359414263_Deep_Learning_Techniques_for_Diabetic_Retinopathy_Classification_A_Survey
https://www.researchgate.net/publication/340896792_Deep_neural_networks_to_predict_diabetic_retinopathy

[2]. https://aravinda-gn.medium.com/how-to-split-image-dataset-into-train-validation-and-test-set-5a41c48af332

[3]. https://www.mdpi.com/2076-3417/13/19/10760

# Run Problems
Useful links:
- https://github.com/peterjc/backports.lzma?tab=readme-ov-file#usage
