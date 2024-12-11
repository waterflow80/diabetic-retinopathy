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

        elif self.backbone is None:
            # Custom CNN Structure
            self.custom_layers = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=3, bias=False),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(64 * 56 * 56, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, output_classes),
            )
        else:
            raise ValueError("Invalid model type. Choose either 'resnet18' or 'resnet50' or 'resnet101'.")

    def forward(self, x):
        if self.backbone:
            features = self.resnet(x)
        else:
            features = self.custom_layers(x)
        return features
