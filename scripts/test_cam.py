import cv2
import os
import time
from calibration.camera_utils import setup_camera

# ======= 設定 =======
device_id = 0
if device_id == 0:
    save_dir = "../outputs/captured_frames_left"
else:
    save_dir = "../outputs/captured_frames_right"

capture_interval = 1
max_frames = 30
frame_width, frame_height = 1280, 720
target_fps = 60
manual_exposure = 100  # 依相機而定，數值大多為「曝光時間」刻度；可微調 50~800

os.makedirs(save_dir, exist_ok=True)

# 開啟攝影機（可換用 CAP_V4L2 在 Linux 比較穩）
cap = setup_camera(device_id, frame_width=1600, frame_height=1200)

print("開始擷取影像，按 q 結束...\n")
frame_count = 0
last_capture_time = time.time()

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        print("擷取失敗，跳過...")
        continue

    scaled_frame = cv2.resize(frame, (640, 360))  # 或 (frame_width // 2, frame_height // 2)
    cv2.imshow('Camera Preview', scaled_frame)

    now = time.time()
    if now - last_capture_time >= capture_interval:
        filename = os.path.join(save_dir, f"frame_{frame_count:03d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[✓] 儲存 {filename}")
        frame_count += 1
        last_capture_time = now

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ 擷取結束。")
