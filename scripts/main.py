import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import load_stereo_parameters
from compute_disparity import compute_disparity, compute_disparity_libsgm
from compute_cloud import compute_pointcloud, save_pointcloud_ply, visualize_pointcloud
import os

save_dir = "../outputs/snapshots"
os.makedirs(save_dir, exist_ok=True)

# 1. 讀取 stereo_camera.yaml
stereo_yaml_path = "../configs/stereo_camera.yaml"
mtxL, distL, mtxR, distR, R, T = load_stereo_parameters(stereo_yaml_path)

image_size = (640, 480)

# Open camera devices (modify indices if needed)
capL = cv2.VideoCapture(0)  # /dev/video0
capR = cv2.VideoCapture(2)

# Set resolution
capL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Read one frame to get size
retL, frameL = capL.read()
retR, frameR = capR.read()
h, w = frameL.shape[:2]
image_size = (w, h)

# Stereo rectify maps
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0)

left_map1, left_map2 = cv2.initUndistortRectifyMap(
    mtxL, distL, R1, P1, image_size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(
    mtxR, distR, R2, P2, image_size, cv2.CV_16SC2)

mtxL = P1
mtxR = P2

idx = 0

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        print("Error reading from cameras")
        break

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # Rectify
    rectL = cv2.remap(grayL, left_map1, left_map2, interpolation=cv2.INTER_LINEAR)
    rectR = cv2.remap(grayR, right_map1, right_map2, interpolation=cv2.INTER_LINEAR)

    # 將三張圖轉為 3-channel（灰階 → BGR），才能一起拼
    # 假設 frameL 和 frameR 是原始彩色畫面
    rectL_color = cv2.remap(frameL, left_map1, left_map2, interpolation=cv2.INTER_LINEAR)
    rectR_color = cv2.remap(frameR, right_map1, right_map2, interpolation=cv2.INTER_LINEAR)

    # Disparity
    disparity = compute_disparity(rectL, rectR)

    # Normalize for visualization
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)

    # 水平拼接
    disp_bgr = cv2.cvtColor(disp_vis, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((rectL_color, rectR_color, disp_bgr))
    for y in range(0, combined.shape[0], 40):
            cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)
    # 顯示單一視窗
    cv2.imshow("Stereo Rectified + Disparity", combined)


    key = cv2.waitKey(1)
    if key == ord('s'):
        # 存目前畫面（含水平線）
        cv2.imwrite(os.path.join(save_dir, f"screen_{idx}.png"), combined)
        # 另外各自存校正後圖與視差圖
        cv2.imwrite(os.path.join(save_dir, f"rectL_{idx}.png"), rectL_color)
        cv2.imwrite(os.path.join(save_dir, f"rectR_{idx}.png"), rectR_color)
        idx += 1
        # cv2.imwrite(os.path.join(save_dir, f"disp_{ts}.png"), disp_vis)
        # 若想存原始 float 視差，可再加：np.save(os.path.join(save_dir, f"disp_{ts}.npy"), disparity)
        print(f"[✓] Saved snapshots to {save_dir}")
    elif key == ord('p'):

        # --- 放在 disparity 計算後 ---
        valid = np.isfinite(disparity)
        valid_count = int((disparity > 0).sum())
        print(f"[DBG] disparity stats: min={np.nanmin(disparity):.3f}, max={np.nanmax(disparity):.3f}, >0 count={valid_count}")

        # 若幾乎全部都是 0，代表沒有匹配到
        if valid_count < 1000:
            print("[WARN] 幾乎沒有有效視差：請檢查曝光/同步/rectify是否正確，以及 StereoSGBM 參數（numDisparities, blockSize, uniquenessRatio 等）。")

        # ✅ 轉點雲（依需要調整 disp_min / z_max）
        pts, cols = compute_pointcloud(disparity, Q, rectL_color)
        visualize_pointcloud(pts, cols, show_axis=True)
        ply_path = os.path.join(save_dir, f"cloud_{idx}.ply")
        save_pointcloud_ply(ply_path, pts, cols)
        print(f"[✓] Saved point cloud: {ply_path}  (points: {len(pts)})")
    elif key == 27:  # ESC
        break

capL.release()
capR.release()
cv2.destroyAllWindows()

