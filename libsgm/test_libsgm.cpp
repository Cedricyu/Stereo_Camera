#include <opencv2/opencv.hpp>
#include <libsgm.h>
#include <iostream>
#include "cnpy.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <left_image> <right_image>" << std::endl;
        return -1;
    }

    cv::Mat left  = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat right = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    if (left.empty() || right.empty()) {
        std::cerr << "Error: failed to load input images." << std::endl;
        return -1;
    }
    if (left.size() != right.size()) {
        std::cerr << "Error: left/right size mismatch." << std::endl;
        return -1;
    }

    const int width  = left.cols;
    const int height = left.rows;
    const int disp_size = 256; // 你的 max disparity

    // libSGM 輸出 16-bit disparity（含 4-bit 小數）
    cv::Mat disp16u(height, width, CV_16U);

    sgm::StereoSGM ssgm(
        width,
        height,
        disp_size,
        8,   // input depth bits
        16,  // output depth bits
        sgm::EXECUTE_INOUT_HOST2HOST
    );
    ssgm.execute(left.data, right.data, disp16u.data);

    // 轉成 32F 並把 4-bit 小數縮回真實像素（常見是 /16.0）
    cv::Mat disp32f;
    disp16u.convertTo(disp32f, CV_32F, 1.0 / 16.0);

    // 建立有效遮罩：>0 且 < disp_size（單位：像素）
    // 視情況你也可以用 >=1e-3 避免極小噪聲
    cv::Mat mask = (disp32f > 0.0f) & (disp32f < static_cast<float>(disp_size));

    // 統計有效範圍
    double minVal = 0.0, maxVal = 0.0;
    cv::minMaxLoc(disp32f, &minVal, &maxVal, nullptr, nullptr, mask);
    std::cout << "Filtered disparity range: " << minVal << " ~ " << maxVal << std::endl;

    // 線性映射到 0~255（僅針對 mask 區域做 normalize）
    cv::Mat disp8u = cv::Mat::zeros(disp32f.size(), CV_8U);
    if (maxVal > minVal) {
        cv::Mat disp8u_tmp;
        // normalize 支援 mask；僅計算/輸出 mask 內像素
        cv::normalize(disp32f, disp8u_tmp, 0, 255, cv::NORM_MINMAX, CV_8U, mask);
        disp8u_tmp.copyTo(disp8u, mask);  // 寫回可視化圖
    }

    // 存圖
    cv::imwrite("disp_result.png", disp8u);
    std::cout << "Saved disparity map to disp_result.png" << std::endl;

    std::vector<unsigned long> shape = { (unsigned long)disp32f.rows, (unsigned long)disp32f.cols };
    cnpy::npy_save("disp_float.npy", disp32f.ptr<float>(), shape, "w");
    std::cout << "Saved disparity map to disp_float.npy" << std::endl;

    return 0;
}
