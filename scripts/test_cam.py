import cv2
import os
import time

# ======= 設定 =======
device_id = 0                   # 攝影機編號：0 或 1 (依你的 /dev/videoN)
save_dir = "captured_frames_left"   # 存圖的資料夾
capture_interval = 1           # 每幾秒儲存一張圖
max_frames = 30                # 最多擷取幾張圖

# 建立資料夾
os.makedirs(save_dir, exist_ok=True)

# 開啟攝影機
cap = cv2.VideoCapture(device_id)
if not cap.isOpened():
    print(f"無法開啟攝影機 /dev/video{device_id}")
    exit()

# 擷取邏輯
frame_count = 0
last_capture_time = time.time()

print("開始擷取影像，按 q 結束...\n")

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        print("擷取失敗，跳過...")
        continue

    # 顯示畫面
    cv2.imshow('Camera Preview', frame)

    # 檢查是否該儲存影像
    current_time = time.time()
    if current_time - last_capture_time >= capture_interval:
        filename = os.path.join(save_dir, f"frame_{frame_count:03d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[✓] 儲存 {filename}")
        frame_count += 1
        last_capture_time = current_time

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 結束
cap.release()
cv2.destroyAllWindows()
print("\n✅ 擷取結束。")

