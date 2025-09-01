#include <iostream>
#include <opencv2/opencv.hpp>
#include "DEALSD.h"
#include <fstream>
#include <string>
#include <vector>


int lineDetect(std::string path, py::array_t<float> heatmap, py::array_t<float> Out) {
    cv::Mat img = cv::imread(path);
    if (img.empty()) {
        return -1;
    }
    auto h = heatmap.mutable_unchecked<2>();
    if (h.shape(0) != img.rows || h.shape(1) != img.cols) {
        return -1;
    }
    upm::DEALSD dealsd;
    upm::Segments segs = dealsd.detect(img, heatmap);
   
    auto outputPointL = Out.mutable_unchecked<2>();
    for (int i = 0; i < segs.size(); i++) {
        for (int j = 0; j < 4; j++) {
            outputPointL(i, j) = segs[i][j];
        }
    }   
    return segs.size();
}

PYBIND11_MODULE(LineDetector, m) {
    m.def("detect", &lineDetect, "detect line segment");
}