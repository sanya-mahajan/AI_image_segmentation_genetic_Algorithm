import cv2
import numpy as np


img = cv2.imread('images/nuc.png', 0)
cv2.namedWindow('Input image', cv2.WINDOW_NORMAL)
cv2.imshow('Input image', img)
if img is None:
    print("Error: could not load image")
    exit()


hist, bins = np.histogram(img.flatten(), 256, [0, 256])

#  the cumulative distribution function (CDF) of the histogram
cdf = hist.cumsum()
cdf_normalized = cdf / float(cdf.max())

# Kapur entropy function for all possible threshold values
thresholds = range(256)
entropies = []
for t in thresholds:
    # Calculate the probability distribution function (PDF) for each class
    pdf1 = cdf_normalized[t]
    pdf2 = 1 - pdf1
    if pdf1 == 0 or pdf2 == 0:
        entropies.append(0)
        continue
    #  entropy of each class
    entropy1 = -pdf1 * np.log2(pdf1)
    entropy2 = -pdf2 * np.log2(pdf2)
    #  Kapur entropy function
    entropy = entropy1 + entropy2
    entropies.append(entropy)

# find  threshold value that maximizes the Kapur entropy 
optimal_threshold = thresholds[np.argmax(entropies)]

#  threshold the image
img_seg = np.zeros_like(img)
img_seg[img > optimal_threshold] = 255
cv2.namedWindow('seg image', cv2.WINDOW_NORMAL)

cv2.resizeWindow("seg image", 300, 700)
cv2.resizeWindow("Input image", 300, 700)

cv2.imshow('seg image', img_seg)


cv2.waitKey(0)
cv2.destroyAllWindows()
