# scripts/camera_utils.py
import cv2
import os

def setup_camera(
    device_id=0,
    frame_width=1280,
    frame_height=720,
    target_fps=60,
    manual_exposure=100,
    use_auto_wb=False,
    wb_temperature=4500,
    fourcc="MJPG"
):

    # 開啟攝影機
    cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟攝影機 /dev/video{device_id}")

    # 設定編碼格式與解析度、FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 鎖定曝光設定
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, float(manual_exposure))

    # 白平衡
    if use_auto_wb:
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    else:
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb_temperature)

    # 自動對焦可選開關（依相機支援情況）
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    # 取得實際設定
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"攝影機啟動成功: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")

    return cap
