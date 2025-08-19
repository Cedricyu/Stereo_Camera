import cv2
import numpy as np
from utils import load_stereo_parameters, save_stereo_parameters

import os
os.makedirs("captures", exist_ok=True)

# === 設定 ===
chessboard_size = (9, 6)
square_size = 0.022
required_pairs = 1  # 至少要幾組角點才計算 R/T

# 3D 棋盤座標
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints_left = []
imgpoints_right = []

# 載入左右相機內參（stereo_camera.yaml，R/T 不存在也沒關係）
mtxL, distL, mtxR, distR, _, _ = load_stereo_parameters("calibration/stereo_camera.yaml")

# 開啟左右相機（調整索引為實際相機）
capL = cv2.VideoCapture(0)  # /dev/video0
capR = cv2.VideoCapture(2)  # /dev/video2

capL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("[INFO] 按空白鍵擷取一組圖像角點。擷取完成後按 Enter ⏎ 執行 stereo calibration。")

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        print("攝影機讀取失敗")
        break

    dispL = frameL.copy()
    dispR = frameR.copy()

    # 顯示當前畫面
    combined = np.hstack((dispL, dispR))
    cv2.imshow("Stereo View (Left | Right)", combined)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        # 擷取棋盤角點
        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size)
        retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size)

        if retL and retR:
            objpoints.append(objp)
            imgpoints_left.append(cornersL)
            imgpoints_right.append(cornersR)
            print(f"[INFO] 擷取成功，第 {len(objpoints)} 組")
            # 可視化角點
            visL = cv2.drawChessboardCorners(frameL.copy(), chessboard_size, cornersL, retL)
            visR = cv2.drawChessboardCorners(frameR.copy(), chessboard_size, cornersR, retR)
            idx = len(objpoints)  # 當前擷取的組數
            # 水平拼接角點圖像
            combined = np.hstack((frameL, frameR))

            # 儲存拼接圖
            cv2.imwrite(f"captures/corners_pair_{idx:02d}.png", combined)
            print(f"[INFO] Saved corners pair: corners_pair_{idx:02d}.png")

        else:
            print("[WARN] 沒找到棋盤角點")

    elif key == 13:  # Enter ⏎
        if len(objpoints) < required_pairs:
            print(f"[WARN] 目前只有 {len(objpoints)} 組，至少需要 {required_pairs} 組")
            continue

        print("[INFO] 開始 stereoCalibrate...")
        image_size = grayL.shape[::-1]
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right,
            mtxL, distL, mtxR, distR, image_size,
            criteria=criteria,
            flags=flags
        )

        print("[INFO] Stereo calibration 完成")
        print("R =\n", R)
        print("T =\n", T)

        # === Rectification ===
        RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(
            mtxL, distL, mtxR, distR, image_size, R, T,
            alpha=0  # 0 表示裁切到有效區域
        )

        mapLx, mapLy = cv2.initUndistortRectifyMap(
            mtxL, distL, RL, PL, image_size, cv2.CV_32FC1
        )
        mapRx, mapRy = cv2.initUndistortRectifyMap(
            mtxR, distR, RR, PR, image_size, cv2.CV_32FC1
        )

        # 重新讀一次畫面顯示 rectified 結果
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        rectL = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
        rectR = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)

        # 畫水平輔助線
        vis = np.hstack((rectL, rectR))
        for y in range(0, vis.shape[0], 40):
            cv2.line(vis, (0, y), (vis.shape[1], y), (0, 255, 0), 1)
        cv2.imshow("Rectified Stereo View", vis)
        cv2.waitKey(0)

        rectL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        rectR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
        # 存檔
        cv2.imwrite("rectified_left.png", rectL)
        cv2.imwrite("rectified_right.png", rectR)

        # 儲存 R/T
        save_stereo_parameters("../../configs/stereo_camera.yaml", mtxL, distL, mtxR, distR, R, T)
        print("[INFO] 已儲存到 stereo_camera.yaml")
        break

    elif key == 27:  # ESC 離開
        print("離開中")
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
