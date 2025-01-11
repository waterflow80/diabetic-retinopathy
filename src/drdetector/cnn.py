import torch.nn as nn
import torchvision.models as models

class Classifier(nn.Module):
    """
    Classifier is a CNN model with a configurable backbone.

    Parameters:
    ------------
    output_classes: int
        Number of output classes.
    backbone str, optional
        Backbone model name (default: 'resnet18')
    freeze_backbone bool, optional
        If True, freezes backbone layers for transfer learning (default: True)
    """
    def __init__(self, output_classes, backbone='resnet18', freeze_backbone=True):
        super(Classifier, self).__init__()
        assert isinstance(output_classes, int) and output_classes > 0, "output_classes must be a positive integers"
        self.backbone = backbone

        if self.backbone in ["resnet18", "resnet50", "resnet101"]:
            self.resnet = getattr(models, self.backbone)(pretrained=True)

            # Optionally freeze the backbone paramters
            if freeze_backbone:
                for param in self.resnet.parameters():
                    param.requires_grad = False

            # Adapting the resnet model to our task
            # ImageNet has a number of 1000 classes/outputs, we should change it to our desired number of classes
            num_ftrs = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(num_ftrs, output_classes)

        elif self.backbone == "alexnet":
            # Custom CNN Structure (AlexNet)
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
                nn.BatchNorm2d(96, momentum=0.9),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(256, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(384, momentum=0.9),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(384, momentum=0.9),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256, momentum=0.9),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, output_classes)
            )
        else:
            raise ValueError("Invalid model type. Choose either 'resnet18' or 'resnet50' or 'resnet101' or 'alexnet'.")

    def forward(self, x):
        if self.backbone in ["resnet18", "resnet50", "resnet101"]:
            outputs = self.resnet(x)
        else:
            features = self.features(x)
            outputs = self.classifier(features)
        return outputs
