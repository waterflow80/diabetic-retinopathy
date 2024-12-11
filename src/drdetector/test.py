import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report


def test_classifier(model, test_loader, plot_dir, backbone, freeze_backbone, class_names, device):
    """
    Evaluates the model on labeled data or runs inference on unlabeled data and saves the results


    Parameters
    ----------
    :param model:
        The trained model to evaluate or use for inference.
    :type model: torch.nn.Module
    :param test_loader:
        DataLoader for the test dataset.
    :type test_loader: torch.utils.data.DataLoader
    :param plot_dir:
        Directory path to save the evaluation plots (only used for evaluation).
    :type plot_dir: str
    :param backbone:
        Name of the model's backbone layers during training
    :type backbone: str
    :param freeze_backbone:
        Whether to freeze the backbone layers during training.
    :type freeze_backbone: bool
    :param class_names:
        List of class names (e.g., 'Mild', 'Moderate', 'Severe')
    :type class_names: list
    :param device:
        Device to run the evaluation on (e.g., 'cpu' or 'cuda').
    :return:
        None
    """
    # Set the model to evaluation mode
    model.eval()
    # For evaluation, we need to track accuracy, confusion matrix, etc.
    correct_preds = 0
    incorrect_preds = 0
    total_samples = 0
    true_labels = []
    predictions = []

    # CUDA memory consumption (if using GPU)
    if device.type == 'cuda':
        torch.cuda.reset_max_memory_allocated(device)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).long()

            # Forward pass through the model
            output = model(images).to(device)

            # Get predictions
            _, pred = output.max(1)

            # Compute accuracy
            correct_preds += pred.eq(labels).sum().item()
            incorrect_preds += (pred != labels).sum().item()
            total_samples += labels.size(0)

            # Collect true labels and predictions for metrics
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(pred.cpu().numpy())

    # Evaluation metrics
    accuracy = correct_preds / total_samples
    wrong_pred = incorrect_preds / total_samples  # Can also be calculated by 1 - accuracy

    logging.info(f"Mis-classification rate: {wrong_pred * 100:.2f}%, Accuracy: {accuracy * 100:.2f}%")

    # Confusion matrix and classification report
    cm = confusion_matrix(true_labels, predictions)
    class_report = classification_report(true_labels, predictions, output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()
    logging.info("Confusion matrix:\n%s", pd.DataFrame(cm))
    logging.info("Classification report:\n%s", class_report_df)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(plot_dir, f"cm_{backbone}_freeze_backbone_{freeze_backbone}.png"))
    plt.show()