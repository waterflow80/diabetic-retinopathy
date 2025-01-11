import argparse
import logging
import os
import time
import warnings
from os import killpg
from pickletools import optimize

warnings.filterwarnings('ignore')
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from .datasets import Dataset
from .cnn import Classifier
from .config import *
from .train import train_classifier
from .test import test_classifier
from .load_chkpts import load_checkpoint
import drdetector.preprocessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description="DR Classification")
    parser.add_argument("--mode", type=str, choices=["train", "test", "preprocess"], required=True,
                        help="Mode to run: 'train' or 'test'")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset root directory")
    parser.add_argument("--images_dir", type=str, default="train_images",
                        help="Path to images")
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of images to process")
    parser.add_argument("--model_path", type=str, default="./models/cnn_resnet18_freeze_backbone_False.pth",
                        help="Directory to save or load the model")
    args = parser.parse_args()

    # Define the data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialize the CNN model
    model = Classifier(len(CLASS_NAMES), backbone=BACKBONE, freeze_backbone=FREEZE_BACKBONE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"CUDA IS AVAILABLE: {torch.cuda.is_available()}")
    model.to(device)

    if args.mode == "train":
        # Load the entire dataset
        dataset = Dataset(root_dir=args.data_path, images_dir=args.images_dir, classes_file=TRAIN_CLASSES_FILE, transform=transform, mode=args.mode, limit=args.limit)
        # create dirs
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        if not os.path.exists(PLOTS_DIR):
            os.makedirs(PLOTS_DIR)

        criterion = torch.nn.CrossEntropyLoss()

        # Define k-fold cross-validation with k=5
        k_folds = 5
        #k_folds = 2 # Just for testing purposes

        kfold = KFold(n_splits=k_folds, shuffle=True)

        # k-fold Cross Validation model evaluation
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            # Print current fold
            logging.info(f"FOLD {fold}")
            logging.info('-------------------------------')

            # Sample elements randomly from a given list of ids, no replacement
            train_subsampler = SubsetRandomSampler(train_ids)
            val_subsampler = SubsetRandomSampler(val_ids)

            # Define data loaders for training and validation
            train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_subsampler)
            val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_subsampler)

            # Initialize optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

            # Log fold and training information
            logging.info(f'''Starting training for fold {fold}:
                Training size:   {len(train_subsampler)}
                Validation size: {len(val_subsampler)}
                Backbone:        {BACKBONE}
                Freeze Backbone: {FREEZE_BACKBONE}
                Batch size:      {BATCH_SIZE}
                Epochs:          {MAX_EPOCHS_NUM}
                Device:          {device}
            ''')

            # Train the model
            train_classifier(model, train_loader, val_loader, criterion, optimizer, MAX_EPOCHS_NUM,
                             MODEL_DIR, PLOTS_DIR, device, BACKBONE, FREEZE_BACKBONE)

            # Optionally clear the GPU cache
            torch.cuda.empty_cache()

        logging.info('Training complete.')

    elif args.mode == "test":
        # Create the test dataset
        testset = Dataset(root_dir=args.data_path, images_dir=args.images_dir, classes_file=TEST_CLASSES_FILE, transform=transform, mode=args.mode, limit=args.limit)
        test_loader = DataLoader(testset, batch_size=1, shuffle=False)
        # Load model checkpoint
        model, _, _ = load_checkpoint(model, args.model_path)
        test_classifier(model, test_loader, PLOTS_DIR, BACKBONE, FREEZE_BACKBONE, CLASS_NAMES, device)


    else: # args.mode == "preprocess"
        start_time = time.time()
        try:
            images_path = os.path.join(args.data_path, args.images_dir)
            target_directory = "{}_{}".format(images_path, "preprocessed") # Eg: train_images_preprocessed
            preprocessing.preprocess(images_path, target_directory)
        except KeyboardInterrupt:
            logging.info('Stopping preprocessing. Total Execution Time: %0.2fs', time.time() - start_time)

        logging.info('Total Execution Time: %0.2fs', time.time() - start_time)



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="DR Classification")
    # parser.add_argument("--mode", type=str, choices=["train", "test"], required=True,
    #                     help="Mode to run: 'train' or 'test'")
    # parser.add_argument("--data_path", type=str, required=True,
    #                     help="Path to dataset")
    # parser.add_argument("--model_path", type=str, default="./models/",
    #                     help="Directory to save or load the model")
    #
    # args = parser.parse_args()
    # main(args)
    pass