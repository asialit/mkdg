import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy import signal

img = Image.open('img.png')

# Sobel horizontal
sobelh_matrix = np.asarray([[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]])


# Sobel vertical
sobelv_matrix = np.asarray([[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]])

# Laplace
laplace_matrix = np.asarray([[0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]])

# Prewitt vertical
prewittv_matrix = np.asarray([[-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1]])

img = np.array(img, dtype=np.uint8)
barwy = np.asarray(img, dtype="int32")

R = barwy[:, :, 0]
G = barwy[:, :, 1]
B = barwy[:, :, 2]

img = np.add(R * 0.299, G * 0.587, B * 0.114)

sobelh = signal.convolve2d(img, sobelh_matrix)
sobelv = signal.convolve2d(img, sobelv_matrix)
laplace = signal.convolve2d(img, laplace_matrix)
prewittv = signal.convolve2d(img, prewittv_matrix)

# plt.imshow(sobelh, cmap="gray")
# plt.title("Sobel horizontal")
# plt.show()
#
# plt.imshow(sobelv, cmap="gray")
# plt.title("Sobel vertical")
# plt.show()
#
# plt.imshow(laplace, cmap="gray")
# plt.title("Laplace")
# plt.show()

plt.imshow(prewittv, cmap="flag")
plt.title("Prewitt vertical")
plt.show()
