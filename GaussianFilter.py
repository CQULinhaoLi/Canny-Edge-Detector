import numpy as np
from scipy.signal import convolve2d

def gaussian_kernel(size, sigma):
    """
    Create a 2d gaussian kernel with a variance of sigma
    :param size: Kernel size (must be odd)
    :param sigma: The variance of gaussian kernel
    :return: Gaussian kernel
    """
    k = size // 2
    x, y = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)  # 归一化
    return kernel

def apply_gaussian_filter(image, kernel):
    """
    Use kernel to filter the input image
    :param image: The image to be filtered
    :param kernel: Gaussian kernel
    :return: Filtered image
    """
    return convolve2d(image, kernel, mode='same', boundary='symm')

def gaussian_blur(image, kernel_length=5, sigmaX=1.0):
    """
    A function to apply gaussian_blur of the given image
    :param image: The image to be blurred
    :param kernel_length: The length of the 2d kernel
    :param sigmaX: The variance of the gaussian kernel
    :return: Blurred image
    """
    gaussian_kernel_2d = gaussian_kernel(kernel_length, sigmaX)
    blurred = apply_gaussian_filter(image, gaussian_kernel_2d)

    return blurred
