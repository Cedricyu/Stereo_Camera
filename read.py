import numpy as np
import matplotlib.pyplot as plt

disp = np.load("video/depth_npy/disp_0028.npy")

# 畫 histogram
plt.hist(disp.ravel(), bins=100, range=(0, 150))
plt.title("Disparity Histogram")
plt.xlabel("Disparity Value")
plt.ylabel("Pixel Count")
plt.grid(True)
plt.show()
