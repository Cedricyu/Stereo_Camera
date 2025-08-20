#include <opencv2/opencv.hpp>
#include "libsgm_wrapper.h"   // 用 wrapper
#include <libsgm.h>           // 為了讀 SUBPIXEL_SCALE
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

    // --- 1) 建立 Wrapper（可依需要調參） ---
    const int numDisp = 128;  // 必為 16 的倍數
    const bool useSubpixel = true;
    sgm::LibSGMWrapper sgmw(
        /*numDisparity=*/numDisp,
        /*P1=*/10,
        /*P2=*/120,
        /*uniquenessRatio=*/0.95f,
        /*subpixel=*/useSubpixel,
        /*pathType=*/sgm::PathType::SCAN_8PATH,
        /*minDisparity=*/0,
        /*lrMaxDiff=*/1,
        /*censusType=*/sgm::CensusType::SYMMETRIC_CENSUS_9x7
    );

    // --- 2) 執行：輸出是 CV_16S（固定小數，若 subpixel=true） ---
    cv::Mat disp16s;  // 不用先配置，wrapper 會幫你配置
    sgmw.execute(left, right, disp16s);

    // --- 3) 轉成 float 視差（像素單位），並處理 invalid/minDisp ---
    const int invalid = sgmw.getInvalidDisparity();
    const int minDisp = sgmw.getMinDisparity();
    const int SUB = sgm::StereoSGM::SUBPIXEL_SCALE;  // 通常 16

    cv::Mat disp32f(disp16s.size(), CV_32F, 0.0f);
    for (int y=0; y<disp16s.rows; ++y) {
        const short* src = disp16s.ptr<short>(y);
        float* dst = disp32f.ptr<float>(y);
        for (int x=0; x<disp16s.cols; ++x) {
            short v = src[x];
            if (v == invalid) { dst[x] = 0.0f; continue; } // 你也可用 NaN
            float d = static_cast<float>(v);
            if (useSubpixel) d /= static_cast<float>(SUB); // 固定小數 -> 像素
            d += static_cast<float>(minDisp);              // 若 minDisp!=0 要加回
            dst[x] = d;
        }
    }

    // --- 4) 建立有效遮罩（>0）並統計範圍 ---
    cv::Mat mask = (disp32f > 0.0f);
    double minVal=0.0, maxVal=0.0;
    if (cv::countNonZero(mask) > 0) {
        cv::minMaxLoc(disp32f, &minVal, &maxVal, nullptr, nullptr, mask);
        std::cout << "Filtered disparity range: " << minVal << " ~ " << maxVal << std::endl;
    } else {
        std::cout << "No valid disparities.\n";
    }

    // --- 5) 視覺化（0~255，僅針對有效區域 normalize） ---
    cv::Mat disp8u = cv::Mat::zeros(disp32f.size(), CV_8U);
    if (maxVal > minVal) {
        cv::Mat tmp8u;
        cv::normalize(disp32f, tmp8u, 0, 255, cv::NORM_MINMAX, CV_8U, mask);
        tmp8u.copyTo(disp8u, mask);
    }
    cv::imwrite("disp_result.png", disp8u);
    std::cout << "Saved disparity map to disp_result.png\n";

    // --- 6) 存 NPY（float 像素視差；invalid=0） ---
    // 若要用 NaN 標 invalid，可先把 mask 反轉後設成 NaN
    // disp32f.setTo(std::numeric_limits<float>::quiet_NaN(), ~mask);
    std::vector<unsigned long> shape = {
        static_cast<unsigned long>(disp32f.rows),
        static_cast<unsigned long>(disp32f.cols)
    };
    // 確保連續（通常是連續的）
    cv::Mat disp32f_cont = disp32f.isContinuous() ? disp32f : disp32f.clone();
    cnpy::npy_save("disp_float.npy", disp32f_cont.ptr<float>(), shape, "w");
    std::cout << "Saved disparity map to disp_float.npy\n";

    return 0;
}
