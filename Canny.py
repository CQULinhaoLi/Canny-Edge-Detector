import numpy as np
from scipy.signal import convolve2d

def sobel_filters(image):
    """
    Compute gradient magnitude and direction using Sobel filters.
    :param image: Input grayscale image
    :return: Gradient magnitude and gradient direction
    """
    # Sobel kernels
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Horizontal
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # Vertical

    # Convolve with Sobel kernels
    Ix = convolve2d(image, Kx, mode='same', boundary='symm')
    Iy = convolve2d(image, Ky, mode='same', boundary='symm')

    # Compute gradient magnitude and direction
    gradient_magnitude = np.sqrt(Ix**2 + Iy**2)
    gradient_direction = np.arctan2(Iy, Ix)  # Direction in radians

    return gradient_magnitude, gradient_direction

def non_max_suppression(gradient_magnitude, gradient_direction):
    """
    Apply non-maximum suppression to thin edges.
    :param gradient_magnitude: Gradient magnitude
    :param gradient_direction: Gradient direction
    :return: Thinned gradient magnitude
    """
    M, N = gradient_magnitude.shape
    suppressed = np.zeros((M, N), dtype=np.float32)
    angle = gradient_direction * 180.0 / np.pi  # Convert to degrees
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            q = 255
            r = 255
            # Determine the neighbors to interpolate based on the gradient direction
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = gradient_magnitude[i, j + 1]
                r = gradient_magnitude[i, j - 1]
            elif (22.5 <= angle[i, j] < 67.5):
                q = gradient_magnitude[i + 1, j - 1]
                r = gradient_magnitude[i - 1, j + 1]
            elif (67.5 <= angle[i, j] < 112.5):
                q = gradient_magnitude[i + 1, j]
                r = gradient_magnitude[i - 1, j]
            elif (112.5 <= angle[i, j] < 157.5):
                q = gradient_magnitude[i - 1, j - 1]
                r = gradient_magnitude[i + 1, j + 1]

            # Suppress non-maximum pixels
            if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                suppressed[i, j] = gradient_magnitude[i, j]
            else:
                suppressed[i, j] = 0

    return suppressed

def double_threshold(suppressed, low_threshold, high_threshold):
    """
    Apply double threshold to classify strong, weak, and non-edge pixels.
    :param suppressed: Thinned gradient magnitude
    :param low_threshold: Low threshold value
    :param high_threshold: High threshold value
    :return: Thresholded image and edge classifications
    """
    strong = 255
    weak = 75
    result = np.zeros_like(suppressed, dtype=np.uint8)

    # Identify strong and weak edges
    strong_i, strong_j = np.where(suppressed >= high_threshold)
    weak_i, weak_j = np.where((suppressed <= high_threshold) & (suppressed >= low_threshold))

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    return result, strong, weak

def edge_tracking_by_hysteresis(result, strong, weak):
    """
    Perform edge tracking by hysteresis to finalize edges.
    :param result: Thresholded image
    :param strong: Value for strong edges
    :param weak: Value for weak edges
    :return: Final edge map
    """
    M, N = result.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if result[i, j] == weak:
                # Check if weak edge is connected to any strong edge
                if ((result[i + 1, j - 1] == strong) or (result[i + 1, j] == strong) or
                    (result[i + 1, j + 1] == strong) or (result[i, j - 1] == strong) or
                    (result[i, j + 1] == strong) or (result[i - 1, j - 1] == strong) or
                    (result[i - 1, j] == strong) or (result[i - 1, j + 1] == strong)):
                    result[i, j] = strong
                else:
                    result[i, j] = 0
    return result


def canny_edge_detector(blurred, low_threshold, high_threshold):
    """
    :param bluured_image:
    :param low_threshold:
    :param high_threshold:
    :return:
    """
    # 1. Compute gradients
    gradient_magnitude, gradient_direction = sobel_filters(blurred)

    # 2. Apply non-maximum suppression
    suppressed = non_max_suppression(gradient_magnitude, gradient_direction)

    # 3. Apply double threshold
    thresholded, strong, weak = double_threshold(suppressed, low_threshold, high_threshold)

    # 4. Perform edge tracking by hysteresis
    edges = edge_tracking_by_hysteresis(thresholded, strong, weak)

    return edges


