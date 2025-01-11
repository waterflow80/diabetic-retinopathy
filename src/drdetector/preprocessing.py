
import cv2
import numpy as np
import os
import logging


def get_optimal_clip_limit(image):
    """
    Return the optimal clip_limit for the CLAHE Histogram Equalization of for the given
    image.
    ref: https://www.mdpi.com/2076-3417/13/19/10760 | 3.1. Finding Optimal Values
    :param image:
    :return:
    """
    original = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # To obtain the luminance/brightness channel
    original = cv2.resize(original, (500, 600))
    original_l, _, _ = cv2.split(original)
    max_variance = -np.inf
    optimal_clip_limit = 0
    patience = 10  # How many iterations to wait without any change
    iters_no_change = 0

    for clip_limit in range(256):
        clahe = cv2.createCLAHE(clipLimit=clip_limit)
        clahe_transformed_l = clahe.apply(original_l)
        abs_diff = cv2.absdiff(original_l, clahe_transformed_l)
        variance = np.var(abs_diff)
        if variance > max_variance:
            max_variance = variance
            optimal_clip_limit = clip_limit
            iters_no_change = 0
        else:
            iters_no_change += 1
            if iters_no_change >= patience:
                break

    return optimal_clip_limit

def calculate_sharpness_variance(image_l):
    """
    Calculate the sharpness variance of the image using the Laplacian edge-detection
    :param image_l: The luminance channel of the image
    :return:
    """

    laplacian = cv2.Laplacian(image_l, cv2.CV_64F)
    sharpness_var = laplacian.var()
    return sharpness_var

def get_optimal_alpha(image, oc):
    """
    Return the optimal additional brightness (alpha) for the CLAHE Histogram Equalization for the
    given image.
    :param image:
    :param oc: Optimal Cliplimit
    :return:
    """
    clahe = cv2.createCLAHE(clipLimit=oc)
    original = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # To obtain the luminance/brightness channel
    original_l, _, _ = cv2.split(original)
    max_variance = -np.inf
    optimal_alpha = 0
    patience = 10  # How many iterations to wait without any change
    iters_no_change = 0

    for alpha in range(256):
        clahe_transformed_l = clahe.apply(original_l)
        clahe_transformed_l_with_alpha = cv2.add(clahe_transformed_l, alpha)
        sharpness_variance = calculate_sharpness_variance(clahe_transformed_l_with_alpha)
        if sharpness_variance > max_variance:
            max_variance = sharpness_variance
            optimal_alpha = alpha
            iters_no_change = 0
        else:
            iters_no_change += 1
            if iters_no_change >= patience:
                break
    return optimal_alpha


def get_clahe_enhanced_image(image):
    """
    Return the enhanced image using the CLAHE Histogram equalization and the Laplacian edge-detection
    :param image: Image in BGR format
    :return:
    """
    logging.debug(f'Applying CLAHE on image {image}')
    oc = get_optimal_clip_limit(image)
    alpha = get_optimal_alpha(image, oc)

    # Splitting channels
    image_l, image_a, image_b = cv2.split(image)

    clahe = cv2.createCLAHE(clipLimit=oc)
    clahe_transformed_l = clahe.apply(image_l)
    clahe_transformed_l = cv2.add(clahe_transformed_l, alpha)

    # Merging channels
    res_image  = cv2.merge([clahe_transformed_l, image_a, image_b])
    return res_image



def preprocess(data_path, target_directory):
    """
    Preprocess data (enhance contrast, quality), using CLAHE Histogram equalization,
    and save enhanced images to the target_directory.

    :param data_path: Path to the data directory in which the images are located
    :param target_directory: Path to the target directory in which the enhanced images will be saved
    :return:
    """
    logging.info(f'Preprocessing images in {data_path}')
    logging.info(f'Creating target directory in {target_directory}')
    os.makedirs(target_directory, exist_ok=True)
    for idx, image in enumerate(os.listdir(data_path)):
        try:
            with open(os.path.join(target_directory, "processed_images.txt"), "r") as processed_images_file:
                processed_images = set(map(str.strip, processed_images_file.readlines()))
                if image in processed_images:
                    logging.info(f'Skipping {image}. Already processed.')
                    continue
        except FileNotFoundError:
            logging.info(f'Creating cache file processed_images.txt.')
            open(os.path.join(target_directory, "processed_images.txt"), "a").close()  # Creating the file if not exists
        logging.info(f"Processing image #{idx}: {image}...")
        image_bgr = cv2.imread(os.path.join(data_path, image))
        enhanced_image = get_clahe_enhanced_image(image_bgr)
        cv2.imwrite(os.path.join(target_directory, image), enhanced_image)

        with open(os.path.join(target_directory, "processed_images.txt"), "a") as processed_images_file:
            logging.info(f"Caching processed image {image} to {processed_images_file.name}")
            # Save image  name to a file to avoid duplicate processing in the future
            processed_images_file.write(image)
            processed_images_file.write("\n")

    logging.info(f'Finished preprocessing images in {data_path}.')
