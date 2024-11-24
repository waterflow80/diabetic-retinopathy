"""
Image Transformations
"""
import numpy as np
import cv2

def threshold_transform(image, threshold=80, otsu=False):
    if otsu:
        # Convert image to cv2 format, change to Grayscale, and apply Otsu
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, ret_img = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Return image in Pillow format
        return ret_img

    _, ret_img = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return ret_img


def negative_transform(image):
    img_cv2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    neg_img = 255 - 1 - img_cv2
    return Image.fromarray(neg_img)


def log_transform(image, c=59):
    img_cv2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    log_img = c * (np.log(img_cv2 + 1))
    log_img = log_img.astype(np.uint8)
    return log_img


def gamma_transform(image, gamma=0.5):
    img_cv2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gamma_img = np.power(img_cv2 / 255.0, gamma) * 255.0
    gamma_img = gamma_img.astype(np.uint8)
    return gamma_img


def equalize_histogram_transform(image):
    img_cv2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equaliz_hist_img = cv2.equalizeHist(img_cv2)
    return equaliz_hist_img