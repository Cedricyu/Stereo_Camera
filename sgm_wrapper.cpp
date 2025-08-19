// sgm_wrapper_buffer.cpp
#include <opencv2/opencv.hpp>
#include <libsgm.h>
#include <iostream>

extern "C" {
int compute_disparity_from_buffer(
    const unsigned char* left_buf,
    const unsigned char* right_buf,
    unsigned short* out_disp_buf,
    int width,
    int height
) {
    if (!left_buf || !right_buf || !out_disp_buf) return -1;

    const int disp_size = 256;

    sgm::StereoSGM ssgm(
        width,
        height,
        disp_size,
        8,  // input depth bits
        16, // output depth bits
        sgm::EXECUTE_INOUT_HOST2HOST
    );

    ssgm.execute(left_buf, right_buf, out_disp_buf);
    return 0;
}
}
