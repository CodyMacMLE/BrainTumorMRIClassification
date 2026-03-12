import numpy as np

def primarily_white_in_mask(mask: np.ndarray, threshold: float = 0.75) -> bool:
    """
    Function checks if a grayscale mask is primarily white based on a threshold.
    :param mask: The np.ndarray that represents the mask to check. Expected in grayscale format (2D).
    :param threshold: The percentage of pixels required to be white, as a float (0.0 to 1.0).
    :return: True if white pixel ratio meets or exceeds the threshold, False otherwise.
    """
    if not (0 < threshold <= 1):
        raise ValueError("Threshold value must be lower than or equal to 1.0")
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D grayscale array. Convert to grayscale before passing in (e.g. cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))")

    white_pixels = np.sum(mask == 255)
    total_pixels = mask.size
    return (white_pixels / total_pixels) >= threshold

def is_valid_mri(image: np.ndarray) -> bool:
    """
    Checks if the pixel values are a colored image and not a mask or blank image
    :param image:
    :return: True if valid mri image false if not
    """
    return not (np.all((image == 0) | (image == 255)))