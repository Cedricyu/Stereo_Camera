import numpy as np
import matplotlib.pyplot as plt

# 讀入視差圖
disparity = np.load('../outputs/disp_float.npy')

# 基本過濾：移除 <= 0 或過大（如 > 256）
valid_disp = disparity[(disparity > 0) & (disparity < 256)]


# 畫 histogram（加上 log scale）
plt.figure(figsize=(8, 4))
plt.hist(valid_disp.ravel(), bins=80, color='gray')
plt.yscale('log')  # 尾巴拉平
plt.title("Disparity Histogram LibSGM")
plt.xlabel("Disparity value")
plt.ylabel("Pixel count (log)")
plt.grid(True)
plt.tight_layout()
plt.show()
