# scripts/camera_utils.py
import cv2

def setup_camera(
    device_id=0,
    frame_width=1280,
    frame_height=720,
    target_fps=60,
    manual_exposure=100,      # 依相機定義，通常是 log 值或 1/曝光
    use_auto_wb=False,
    wb_temperature=5000,      # 5000K~5500K 接近日光
    use_auto_exposure=True,
    use_yuyv=False            # 若顏色怪，試 True
):
    cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟攝影機 /dev/video{device_id}")

    # ---- 編碼格式 ----
    fourcc = "YUYV" if use_yuyv else "MJPG"
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # ---- 曝光 ----
    if use_auto_exposure:
        # V4L2: 0.25=auto, 0.75=manual
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        cap.set(cv2.CAP_PROP_EXPOSURE, float(manual_exposure))

    # ---- 白平衡 ----
    if use_auto_wb:
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    else:
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        # 有些攝影機只有在關閉 AUTO_WB 後才會吃溫度
        # 可選：微調 U/V 通道（有支援才會生效）
        # cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4600)
        # cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, 4600)

    # ---- 對焦（依相機支援）----
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    # ---- 讀回實際設定 ----
    actual = {
        "codec": fourcc,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "auto_exposure": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
        "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
        "auto_wb": cap.get(cv2.CAP_PROP_AUTO_WB),
        "wb_temp": cap.get(cv2.CAP_PROP_WB_TEMPERATURE),
    }
    print(f"攝影機啟動: {actual['codec']} {actual['width']}x{actual['height']} @ {actual['fps']:.1f} FPS")
    print(f"AE={actual['auto_exposure']} EXP={actual['exposure']}  AWB={actual['auto_wb']} WB_T={actual['wb_temp']}")
    return cap
