# Diabetic Retinopathy Prediction
Deep learning solution to predict Diabetic Retinopathy using retinal images.

**Note:** Still in progress, the rest of the code hasn't been pushed yet.


## Diabetic Retinopathy Overview
### Diabetic Mellitus
Diabetic Retinopathy is a diabetic complication, in which blood glucose levels tend to increase due to the inability of pancreas to produce sufficient blood insulin. There has been an increasing number of incidents over the last decates: from 108 million in 1980 to 422 million in 2014. It affects liver, heart, kidneys, joints, and eyes, and it is the most prominent reason for blindness.

### Diabetic Retinopathy
Diabetic Retinopathy is a complication of diabetes mellitus, in which glucose blocks blood vessels that feed the eye. It causes swelling and leaking of blood/fluid that can cause eye injury. DR causes abnormal growth of blood which leads to vessels, bleeding, vision loss, and eventuallly blindness in advanced phases. DR amounts to 2.6% of causes of blindness.

Diabetic Retinopathy includes four types of lesions:
#### 1. Microaneurysms (MA):
![Microaneurysms-in-retinal-image](https://github.com/user-attachments/assets/a5fec93a-105c-4965-aa66-c5c555db1147)

It is an early stage of DR. It is caracterized by small **red round dots** on the retina virtue of vessel wall weakness. The dots have sharp magins (size <= 125μm).

#### 2. Haemorrhages (HM):
![Fundoscopy-Dot-and-Blot-Haemorrhages](https://github.com/user-attachments/assets/10a900a6-02a1-42ae-a833-7b9d0fe9ca2c)

Caracterized by **large spots** on the retina. It has irregular margin sizes (>= 125μm), and can be split into two categories: 2 categories:
- **Flame**: superficial spots
- **Blot**: deep spots

#### 3. Hard exudates:
![hard-soft-exudates-left](https://github.com/user-attachments/assets/892e4848-0496-4916-8d0e-c3ea9d24a1b7)

It is a consequence of plasma leakage. It can be caracterized by **yellow spots**, and it spans the outer retina layers. It also has **sharp margins**.

#### 4. Soft exudates:
![hard-soft-exudates-right](https://github.com/user-attachments/assets/f55f282e-43cf-477d-9d12-5f640235fade)

It is a consequence of nerve fibre swelling and visibly seen as **white ovals** on the retina. 

## Datasets
In this project, we are using the [APTOS](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) dataset, which contains decent quality retinal images.

## Image Enhancements
After exploring the existing datasets of images for the DR detection, we found out that the quality of images is too variant. So, we had to perform image enhancement before defining the model's architecture. For this, we referenced to some research papers [1], to enhance image quality in a global manner.

## Models
In this project, we are training on CNN, ViT (Vision Transformer), and ResNet to see which model outputs the best results. The choice of the hyper-paramters is very crucial to ensure peformant and efficient models. We should also visualize the training and the test loss together, and maybe use early stoping, to ensure that we don't get overfitting problems.

## Evaluation
TODO

## References
[1]. https://www.mdpi.com/2076-3417/13/19/10760
https://www.researchgate.net/publication/359414263_Deep_Learning_Techniques_for_Diabetic_Retinopathy_Classification_A_Survey
https://www.researchgate.net/publication/340896792_Deep_neural_networks_to_predict_diabetic_retinopathy

[2]. https://aravinda-gn.medium.com/how-to-split-image-dataset-into-train-validation-and-test-set-5a41c48af332

# Run Problems
Useful links:
- https://github.com/peterjc/backports.lzma?tab=readme-ov-file#usage