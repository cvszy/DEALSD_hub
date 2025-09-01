#ifndef DEALSD_DEALSD_H_
#define DEALSD_DEALSD_H_

#include <ostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "FullSegmentInfo.h"
#include "EdgeDrawer.h"
namespace py = pybind11;

namespace upm {

struct DEALSDParams {
  // Gaussian kernel size
  int ksize = 3;
  // Sigma of the gaussian kernel
  float sigma = 1;
  // The threshold of pixel gradient magnitude.
  float gradientThreshold = 30;

  // Minimum line segment length
  int minLineLen = 15;
  // Threshold used to check if a list of edge points for a line segment
  double lineFitErrThreshold = 0.2;
  // Threshold used to check if a new pixel is part of an already fit line segment
  double pxToSegmentDistTh = 1.5;
  // Threshold used to validate the junction jump region. The first eigenvalue of the gradient
  // auto-correlation matrix should be at least junctionEigenvalsTh times bigger than the second eigenvalue
  double junctionEigenvalsTh = 10;
  // the difference between the perpendicular segment direction and the direction of the gradient
  // in the region to be validated must be less than junctionAngleTh radians
  double junctionAngleTh = 10 * (M_PI / 180.0);
  // the gradient angular error in validation.
  double validationTh = 0.15;

  // Whether to validate or not the generated segments
  bool validate = true;
  // Whether to jump over junctions
  bool treatJunctions = true;
  // List of junction size that will be tested (in pixels)
  std::vector<int> listJunctionSizes = {3,5,7};
};


class DEALSD {
 public:

  explicit DEALSD(const DEALSDParams &params = DEALSDParams());

  Segments detect(const cv::Mat &image, py::array_t<float> heatmap);

  const LineDetectionExtraInfo &getImgInfo() const;

  const LineDetectionExtraInfoPtr &getImgInfoPtr() const;

  void processImage(const cv::Mat &image, py::array_t<float> heatmap);

  void clear();

  static void computeAnchorPoints(const cv::Mat &dirImage,
                                  const cv::Mat &gradImageWO,
                                  const cv::Mat &gradImage,
                                  const cv::Mat &heatmap,
                                  std::vector<Pixel> &anchorPoints);  // NOLINT

  static LineDetectionExtraInfoPtr
  computeGradients(const cv::Mat &srcImg, short gradientTh, py::array_t<float> heatmap);

  ImageEdges getAllEdges() const;

  ImageEdges getSegmentEdges() const;

  const EdgeDrawerPtr &getDrawer() const { return drawer; }

 private:
  void drawAnchorPoints(const uint8_t *dirImg,
                        const std::vector<Pixel> &anchorPoints,
                        uint8_t *pEdgeImg);  // NOLINT

  DEALSDParams params;
  LineDetectionExtraInfoPtr imgInfo;
  ImageEdges edges;
  Segments segments;
  SalientSegments salientSegments;
  std::vector<Pixel> anchors;
  EdgeDrawerPtr drawer;
  cv::Mat blurredImg;
  cv::Mat edgeImg;

};
}
#endif 
