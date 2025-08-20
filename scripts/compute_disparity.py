import cv2
import numpy as np
import ctypes
import os
import argparse
# import matplotlib.pyplot as plt

def compute_disparity(imgL, imgR, disparities=144):
    """
    計算視差圖。
    需要左右圖已經是 rectified 且為灰階圖（單通道）。
    
    Args:
        imgL (np.ndarray): 左圖 (grayscale)
        imgR (np.ndarray): 右圖 (grayscale)

    Returns:
        disparity (np.ndarray): 浮點視差圖（已除以 16）
    """
    # 參數根據經驗可再調整
    min_disp = 0
    block_size = 5

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=disparities,
        blockSize=block_size,
        P1=8 * 1 * block_size ** 2,
        P2=32 * 1 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16

    # plt.hist(disparity[disparity > 0].ravel(), bins=50)
    # plt.title("Disparity Histogram")
    # plt.xlabel("Disparity")
    # plt.ylabel("Pixel Count")
    # plt.show()

    return disparity


try:
    _libsgm = ctypes.CDLL('../libsgm/libsgm_wrapper.so')  # 路徑視情況修改

    _libsgm.compute_disparity_from_buffer.argtypes = [
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int,
        ctypes.c_int
    ]
    _libsgm.compute_disparity_from_buffer.restype = ctypes.c_int

    def compute_disparity_libsgm(imgL: np.ndarray, imgR: np.ndarray) -> np.ndarray:
        """
        使用 libSGM 計算視差圖（回傳 uint16 disparity）。
        圖像須為灰階（H, W）。
        """
        assert imgL.ndim == 2 and imgR.ndim == 2
        assert imgL.shape == imgR.shape
        h, w = imgL.shape

        out_disp = np.zeros((h, w), dtype=np.uint16)

        ret = _libsgm.compute_disparity_from_buffer(
            imgL.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            imgR.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            out_disp.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            w, h
        )

        if ret != 0:
            raise RuntimeError("libSGM failed to compute disparity")
        
        disp_float = out_disp.astype(np.float32)

        # ✅ 移除異常值：視差過小（如0）或過大（如>max）
        valid_range = (disp_float >= 1.0) & (disp_float <= 256)  # 可視化區間
        disp_filtered = np.where(valid_range, disp_float, 0.0)

        return disp_filtered

except OSError as e:
    compute_disparity_libsgm = None
    print(f"[WARNING] Failed to load libSGM: {e}")
    print("Make sure libsgm_buffer.so and its dependencies are in the correct location.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compute disparity with StereoSGBM")
    ap.add_argument("--left", required=True, help="左影像路徑（rectified, grayscale 或彩色）")
    ap.add_argument("--right", required=True, help="右影像路徑（rectified, grayscale 或彩色）")
    args = ap.parse_args()

    imgL = cv2.imread(args.left, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(args.right, cv2.IMREAD_GRAYSCALE)

    disp32 = compute_disparity(imgL, imgR)

    disp_vis = cv2.normalize(disp32, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)

    cv2.imshow("Disparity", disp_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()