import numpy as np


class DRDataset:
    # This class is a helper class to access and manipulate the images
    def __init__(self, img_dir, df, scale=1, img_dim=None, gray=False, transformations=None):
        # transformations: a map containing the transformation and the corresponding arguments in a tuple: {transform1 : (arg1, arg2, ...)}
        # Note that the first argument of the transform() function is the image, and it's not passed in the map
        self.img_dir = img_dir
        self.df = df
        self.transformations = transformations
        self.gray = gray
        self.scale_rate = scale
        self.img_dim = img_dim


    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, f"{img_name}.png")
        image = Image.open(img_path)
        image_dim = (int(image.width * self.scale_rate), int(image.height * self.scale_rate))
        if self.img_dim:
            image_dim = self.img_dim
        image = image.resize(image_dim)
        if self.gray:
            # Convert to grayscale
            image = image.convert("L")

        if self.transformations:
            for transform, args in transformations.items():
                image = transform(image, *args)

        return np.array(image)