import os

import pandas as pd

from PIL import Image
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, root_dir, images_dir, classes_file, transform=None, mode=None, limit=5000):
        """
        :param root_dir (String): Root directory path.
        :param classes_file (String): CSV File containing images' names mapped to their corresponding classes/diagnosis.
        :param transform (callable, optional): A function/transform to apply to the images
        :param mode: test or train
        :param limit (int): Number of images to load
        """
        self.images_dir = os.path.join(root_dir, images_dir)
        self.transform = transform
        #self.classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"] # Train.csv and test.csv already
        # contain the encoded classes
        self.classes_df = pd.read_csv(str(os.path.join(root_dir, classes_file)))
        #self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.mode = mode
        self.imgs = self._make_dataset(limit)

    def _get_image_diagnosis(self, img_path):
        """
        Return the diagnosis value of the given image
        :param img:
        :return:
        """
        img_name = img_path.split('/')[-1]
        id_code = img_name.split('.')[0]  # Remove the image extension (e.g. '.png')
        diag = self.classes_df[self.classes_df["id_code"] == id_code]["diagnosis"]
        diag = int(diag.iloc[0])  # Get the int value from the series
        return diag

    def _make_dataset(self, limit):
        imgs = []
        valid_extensions = {'.png', '.jpg', '.jpeg'}

        for indx, fname in enumerate(os.listdir(self.images_dir)[:limit]):
            if any(fname.lower().endswith(ext) for ext in valid_extensions):
                path = os.path.join(self.images_dir, fname)
                class_idx = self._get_image_diagnosis(path)
                #class_idx = self.class_to_idx[class_name]
                imgs.append((path, class_idx))
        return imgs

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Get the item at index 'idx' from the dataset
        :param idx (int): The index of the sample to retrieve
        :return (tuple): A tuple (image, label)
        """
        img_path, label = self.imgs[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        #print("#####==> RETURNING IMAGE:", image,":", label)
        return image, label
