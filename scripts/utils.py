import cv2

def load_stereo_parameters(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    mtxL = fs.getNode("camera_matrix_left").mat()
    distL = fs.getNode("dist_coeffs_left").mat()
    mtxR = fs.getNode("camera_matrix_right").mat()
    distR = fs.getNode("dist_coeffs_right").mat()
    R = fs.getNode("R").mat()
    T = fs.getNode("T").mat()
    fs.release()
    return mtxL, distL, mtxR, distR, R, T


def save_stereo_parameters(path, mtxL, distL, mtxR, distR, R, T):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix_left", mtxL)
    fs.write("dist_coeffs_left", distL)
    fs.write("camera_matrix_right", mtxR)
    fs.write("dist_coeffs_right", distR)
    fs.write("R", R)
    fs.write("T", T)
    fs.release()
