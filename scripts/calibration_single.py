import cv2
import numpy as np
import os
from glob import glob

CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE = 0.21
STEREO_FOLDER = "/home/aibox/itri/camera_calibration/captures"  # 放左右合併影像的資料夾
LEFT_FOLDER = "../camera_calibration/captured_frames_left"
RIGHT_FOLDER = "../camera_calibration/captured_frames_right"

RECTIFIED_LEFT_DIR = "rectified_left"
RECTIFIED_RIGHT_DIR = "rectified_right"

def undistort_and_save_images(input_folder, output_folder, mtx, dist):
    os.makedirs(output_folder, exist_ok=True)
    image_paths = sorted(glob(os.path.join(input_folder, "*.jpg")))
    print(f"undistort_and_save_images in {output_folder} with {len(image_paths)} images")
    for path in image_paths:
        img = cv2.imread(path)
        h, w = img.shape[:2]

        # 計算最佳新相機矩陣（可裁剪黑邊）
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha=1.0)  # or 0.5
        undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # # 可裁剪有效 ROI
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]

        # 儲存
        filename = os.path.basename(path)
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, undistorted)
        # print(f"[SAVE] {save_path}")

def calibrate_from_folder(image_folder,
                          chessboard_size=(9, 6),
                          square_size=1.0):
    image_paths = sorted(glob(os.path.join(image_folder, "*.jpg")))
    print(f"\n[INFO] Found {len(image_paths)} images in {image_folder}")

    # 建立棋盤格 3D 世界座標 (z=0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0],
                          0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []
    imgpoints = []
    gray = None

    for path in image_paths:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp.copy())  # 每次都要 copy，不可重複用同一個物件
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            imgpoints.append(corners_refined)

            # 可視化
            cv2.drawChessboardCorners(img, chessboard_size, corners_refined, ret)
            cv2.imshow('Chessboard', img)
            cv2.waitKey(100)
        else:
            print(f"[WARN] Chessboard not found in: {path}")

    cv2.destroyAllWindows()

    if len(objpoints) > 0:
        # flags = (
        #     cv2.CALIB_USE_INTRINSIC_GUESS |
        #     cv2.CALIB_FIX_PRINCIPAL_POINT
        # )
        # init_camera_matrix = np.array([
        #     [1000, 0, 640],
        #     [0, 1000, 360],
        #     [0,  0,  1]
        # ], dtype=np.float64)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        print("\n=== Calibration Results ===")
        print(f"Reprojection error: {ret:.4f}")
        print("Camera matrix:")
        print(mtx)
        print("Distortion coefficients:")
        print(dist.ravel())
        return objpoints, imgpoints, gray.shape[::-1], mtx, dist
    else:
        print("[ERROR] No valid chessboard found for calibration.")
        return None

def create_world_points():
    """建立棋盤格世界座標"""
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE
    return objp


def find_chessboard_points(image_paths, objp):
    """讀取左右圖並擷取棋盤格角點"""
    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    for path in image_paths:
        img = cv2.imread(path)
        h, w = img.shape[:2]
        half_w = w // 2
        imgL, imgR = img[:, :half_w], img[:, half_w:]

        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        retL, cornersL = cv2.findChessboardCorners(grayL, CHESSBOARD_SIZE, None)
        retR, cornersR = cv2.findChessboardCorners(grayR, CHESSBOARD_SIZE, None)

        if retL and retR:
            objpoints.append(objp)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

            imgpoints_left.append(cornersL)
            imgpoints_right.append(cornersR)

            # 可視化
            vis = np.hstack((imgL.copy(), imgR.copy()))
            cv2.drawChessboardCorners(vis[:, :half_w], CHESSBOARD_SIZE, cornersL, retL)
            cv2.drawChessboardCorners(vis[:, half_w:], CHESSBOARD_SIZE, cornersR, retR)
            cv2.imshow("Chessboard Pair", vis)
            cv2.waitKey(0)
        else:
            print(f"[WARN] Chessboard not found in {os.path.basename(path)}")

    cv2.destroyAllWindows()
    return objpoints, imgpoints_left, imgpoints_right, (grayL.shape[1], grayL.shape[0])


def stereo_calibration(objpoints, imgpoints_left, imgpoints_right, img_size, mtxL, distL, mtxR, distR):
    """執行雙目校正"""
    print("[INFO] Performing stereo calibration...")
    ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
        objectPoints=objpoints,
        imagePoints1=imgpoints_left,
        imagePoints2=imgpoints_right,
        cameraMatrix1=mtxL,
        distCoeffs1=distL,
        cameraMatrix2=mtxR,
        distCoeffs2=distR,
        imageSize=img_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    print(f"[INFO] Stereo calibration RMS error: {ret:.4f}")
    print("\n=== Extrinsic Parameters ===")
    print("R:\n", R)
    print("\nT:\n", T)
    return mtxL, distL, mtxR, distR, R, T


def stereo_rectification(mtxL, distL, mtxR, distR, img_size, R, T):
    """計算雙目校正映射表"""
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtxL, distL, mtxR, distR, img_size, R, T, alpha=0
    )
    map1L, map2L = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, img_size, cv2.CV_16SC2)
    map1R, map2R = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, img_size, cv2.CV_16SC2)
    return map1L, map2L, map1R, map2R


def rectify_and_save(image_paths, map1L, map2L, map1R, map2R):
    """對影像進行矯正並儲存"""
    for path in image_paths:
        img = cv2.imread(path)
        h, w = img.shape[:2]
        half_w = w // 2
        imgL, imgR = img[:, :half_w], img[:, half_w:]

        rectL = cv2.remap(imgL, map1L, map2L, cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, map1R, map2R, cv2.INTER_LINEAR)

        basename = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(RECTIFIED_LEFT_DIR, f"left_{basename}.png"), rectL)
        cv2.imwrite(os.path.join(RECTIFIED_RIGHT_DIR, f"right_{basename}.png"), rectR)

        # 可視化 + 水平線
        canvas = np.hstack((rectL, rectR))
        for y in range(0, canvas.shape[0], 40):
            cv2.line(canvas, (0, y), (canvas.shape[1], y), (0, 255, 0), 1)

        cv2.imshow("Rectified", canvas)
        key = cv2.waitKey(0)
        if key == 27:  # ESC
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # 單目標定
    left = calibrate_from_folder(LEFT_FOLDER,
                                chessboard_size=CHESSBOARD_SIZE, square_size=SQUARE_SIZE)
    right = calibrate_from_folder(RIGHT_FOLDER,
                                chessboard_size=CHESSBOARD_SIZE, square_size=SQUARE_SIZE)
    
    # os.makedirs(RECTIFIED_LEFT_DIR, exist_ok=True)
    # os.makedirs(RECTIFIED_RIGHT_DIR, exist_ok=True)
    
    # objp = create_world_points()
    # image_paths = sorted(glob(os.path.join(STEREO_FOLDER, "*.png")))
    # print(f"[INFO] Found {len(image_paths)} stereo images in {STEREO_FOLDER}")

    # objpoints, imgpoints_left, imgpoints_right, img_size = find_chessboard_points(image_paths, objp)

    # _, _, img_size, mtxL, distL = left
    # _, _, _, mtxR, distR = right

    # print("image size:", img_size)
    # mtxL, distL, mtxR, distR, R, T = stereo_calibration(
    #     objpoints, imgpoints_left, imgpoints_right, img_size, mtxL, distL, mtxR, distR
    # )
    # map1L, map2L, map1R, map2R = stereo_rectification(mtxL, distL, mtxR, distR, img_size, R, T)
    # rectify_and_save(image_paths, map1L, map2L, map1R, map2R)