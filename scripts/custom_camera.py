class CustomStereoCamera(Camera):
    def __init__(self,
                 stereo_yaml_path,
                 compute_disparity_fn,
                 left_id=0, right_id=2,
                 stop_after=2000,
                 save_images=False,
                 save_dir="dataset/customstereo_{}".format(datetime.now().strftime("%d%m%Y_%H_%M"))):
        # 不使用 Camera 的 rs.pipeline → 重寫
        self.cap = 0
        self.left_cam = cv2.VideoCapture(left_id)
        self.right_cam = cv2.VideoCapture(right_id)
        self.left_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.left_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.right_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.right_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.disp_fn = compute_disparity_fn
        self.stop_after = stop_after
        self.save_images = save_images
        self.save_dir = save_dir
        self.config_dir = "configs/calibration_{}".format(datetime.now().strftime("%d%m%Y_%H_%M"))

        # 載入 stereo calibration
        fs = cv2.FileStorage(stereo_yaml_path, cv2.FILE_STORAGE_READ)
        self.mtxL = fs.getNode("mtxL").mat()
        self.distL = fs.getNode("distL").mat()
        self.mtxR = fs.getNode("mtxR").mat()
        self.distR = fs.getNode("distR").mat()
        self.R = fs.getNode("R").mat()
        self.T = fs.getNode("T").mat()
        fs.release()

        # 試讀解析度
        okL, frameL = self.left_cam.read()
        h, w = frameL.shape[:2]
        self.W, self.H = w, h

        # Stereo rectify
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.mtxL, self.distL, self.mtxR, self.distR, (w, h), self.R, self.T, alpha=0
        )
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.mtxL, self.distL, self.R1, self.P1, (w, h), cv2.CV_16SC2)
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.mtxR, self.distR, self.R2, self.P2, (w, h), cv2.CV_16SC2)

        # 相機內參
        self.fx = self.P1[0, 0]
        self.fy = self.P1[1, 1]
        self.cx = self.P1[0, 2]
        self.cy = self.P1[1, 2]
        self.baseline = -self.P2[0, 3] / self.fx  # 公尺

        self.depth_trunc = 5.0
        self.depth_scale = 1.0

        if self.save_images:
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "rgb"), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "depth"), exist_ok=True)
            os.makedirs(self.config_dir, exist_ok=True)
            self.write_cam_info()

    def get_images(self):
        if self.cap >= self.stop_after:
            return None, None

        okL, frameL = self.left_cam.read()
        okR, frameR = self.right_cam.read()
        if not okL or not okR:
            return None, None

        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        rectL = cv2.remap(grayL, self.left_map1, self.left_map2, interpolation=cv2.INTER_LINEAR)
        rectR = cv2.remap(grayR, self.right_map1, self.right_map2, interpolation=cv2.INTER_LINEAR)
        rectL_color = cv2.remap(frameL, self.left_map1, self.left_map2, interpolation=cv2.INTER_LINEAR)

        # 視差圖（浮點）
        disp32f = self.disp_fn(rectL, rectR)  # e.g., 已除16、加回 minDisp

        # 轉成深度（Z = fx * B / d）
        depth = np.zeros_like(disp32f, dtype=np.float32)
        valid = disp32f > 0
        depth[valid] = self.fx * self.baseline / disp32f[valid]

        # 清除不合法
        if self.depth_trunc:
            depth[(depth <= 0) | (depth > self.depth_trunc)] = 0

        if self.save_images:
            rgb_dir_path = os.path.join(self.save_dir, "rgb")
            depth_dir_path = os.path.join(self.save_dir, "depth")
            name = f"frame{self.cap:06d}"
            cv2.imwrite(os.path.join(rgb_dir_path, f"{name}.jpg"), rectL_color)
            np.save(os.path.join(depth_dir_path, f"{name}.npy"), depth)

        self.cap += 1
        return rectL_color, depth

    def stop(self):
        self.left_cam.release()
        self.right_cam.release()

    def write_cam_info(self):
        cam_info_path = os.path.join(self.config_dir, 'caminfo.txt')
        with open(cam_info_path, 'w') as cam_info:
            cam_info.write("## camera parameters\n")
            cam_info.write("W H fx fy cx cy depth_scale depth_trunc dataset_type\n")
            cam_info.write(f"{self.W} {self.H} {self.fx} {self.fy} {self.cx} {self.cy} {self.depth_scale} {self.depth_trunc} custom\n")
