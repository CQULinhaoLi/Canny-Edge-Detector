import cv2
import numpy as np
import matplotlib.pyplot as plt
from GaussianFilter import gaussian_blur
from Canny import canny_edge_detector

image_name = 'Lenna_gray'
# 1. Read the image and convert it to grayscale
image = cv2.imread(f'images/{image_name}.jpg')  # Replace with your image path
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel_length = 3
sigmaX = 0.7
low_threshold = 50  # Low threshold
high_threshold = 150  # High threshold

# 2. Apply Gaussian blur to remove noise
blurred = gaussian_blur(gray, kernel_length, sigmaX)

# 3. Perform edge detection using the Canny function
edges = canny_edge_detector(blurred, low_threshold, high_threshold)

# 4. Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert OpenCV's BGR format to RGB
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Canny Edge Detection")
plt.imshow(edges, cmap='gray')
plt.axis('off')

# Add parameters as a text box
params_text = (f"Parameters:\n"
               f"Kernel Length: {kernel_length}\n"
               f"SigmaX: {sigmaX}\n"
               f"Low Threshold: {low_threshold}\n"
               f"High Threshold: {high_threshold}")
plt.gcf().text(0.7, 0.05, params_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.savefig(f'Output/{image_name}/0.png')
plt.show()
