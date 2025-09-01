#include "DEALSD.h"
#include "EdgeDrawer.h"

// Decides if we should take the image gradients as the interpolated version of the pixels right in the segment
// or if they are ready directly from the image
#define UPM_SD_USE_REPROJECTION

namespace upm {

DEALSD::DEALSD(const DEALSDParams &params) : params(params) {
}

Segments DEALSD::detect(const cv::Mat& image, py::array_t<float> heatmap) {
  processImage(image, heatmap);
  return segments;
}

const LineDetectionExtraInfo &DEALSD::getImgInfo() const {
  return *imgInfo;
}

void DEALSD::processImage(const cv::Mat& _image, py::array_t<float> heatmap) {
  // Check that the image is a grayscale image
  cv::Mat image;
  switch (_image.channels()) {
    case 3:
      cv::cvtColor(_image, image, cv::COLOR_BGR2GRAY);
      break;
    case 4:
      cv::cvtColor(_image, image, cv::COLOR_BGRA2GRAY);
      break;
    default:
      image = _image;
      break;
  }
  assert(image.channels() == 1);
  // Clear previous state
  this->clear();

  if (image.empty()) {
    return;
  }

  // Set the global image
  // Filter the image
  if (params.ksize > 2) {
    cv::GaussianBlur(image, blurredImg, cv::Size(params.ksize, params.ksize), params.sigma);
  } else {
    blurredImg = image;
  }

  // Compute the input image derivatives
  imgInfo = computeGradients(blurredImg, params.gradientThreshold,heatmap);

   // Detect edges and segment in the input image
  computeAnchorPoints(imgInfo->dirImg,
                      imgInfo->gImgWO,
                      imgInfo->gImg,
                      imgInfo->heatmap,
                      anchors);

  edgeImg = cv::Mat::zeros(imgInfo->imageHeight, imgInfo->imageWidth, CV_8UC1);
  drawer = std::make_shared<EdgeDrawer>(imgInfo,
                                        edgeImg,
                                        params.lineFitErrThreshold,
                                        params.pxToSegmentDistTh,
                                        params.minLineLen,
                                        params.treatJunctions,
                                        params.listJunctionSizes,
                                        params.junctionEigenvalsTh,
                                        params.junctionAngleTh);

  drawAnchorPoints(imgInfo->dirImg.ptr(), anchors, edgeImg.ptr());
}

LineDetectionExtraInfoPtr DEALSD::computeGradients(const cv::Mat &srcImg, short gradientTh, py::array_t<float> heatmap) {
  LineDetectionExtraInfoPtr dstInfo = std::make_shared<LineDetectionExtraInfo>();
  cv::Sobel(srcImg, dstInfo->dxImg, CV_32FC1, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
  cv::Sobel(srcImg, dstInfo->dyImg, CV_32FC1, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);

  int nRows = srcImg.rows;
  int nCols = srcImg.cols;
  int i,j;

  dstInfo->imageWidth = srcImg.cols;
  dstInfo->imageHeight = srcImg.rows;
  dstInfo->gImgWO = cv::Mat(srcImg.size(), dstInfo->dxImg.type());
  dstInfo->gImg = cv::Mat(srcImg.size(), dstInfo->dxImg.type());
  dstInfo->dirImg = cv::Mat(srcImg.size(), CV_8UC1);
  dstInfo->heatmap = cv::Mat(srcImg.size(), CV_32FC1);
  auto* pHeat = dstInfo->heatmap.ptr<float>();
  auto h = heatmap.mutable_unchecked<2>();

  for(i=0;i<h.shape(0);i++)
      for (j = 0; j < h.shape(1); j++) {
          pHeat[i * nCols + j] = h(i, j);
      }


  float *pDX = dstInfo->dxImg.ptr<float>();
  float *pDY = dstInfo->dyImg.ptr<float>();
  auto *pGr = dstInfo->gImg.ptr<float>();
  auto *pGrWO = dstInfo->gImgWO.ptr<float>();
  auto *pDir = dstInfo->dirImg.ptr<uchar>();


  float abs_dx, abs_dy, sum;
  float ver,hor;
  int index;

  for (i = 1; i < nRows - 1; i++) {
      for (j = 1; j < nCols - 1; j++) {
          index= i * nCols + j;
          // Absolute value
          abs_dx = UPM_ABS(pDX[index]);
          // Absolute value
          abs_dy = UPM_ABS(pDY[index]);
          sum = abs_dx + abs_dy;
          // Divide by 2 the gradient
          pGrWO[index] = sum;

            
          if (sum < gradientTh) {
            ver = pHeat[index - nCols] + pHeat[index + nCols];
            hor = pHeat[index - 1] + pHeat[index + 1];
            sum = 0;
            if (ver >= hor) pDir[index] = UPM_EDGE_VERTICAL;
            else pDir[index] = UPM_EDGE_HORIZONTAL;
          }
          else {
            pDir[index]= abs_dx >= abs_dy ? UPM_EDGE_VERTICAL : UPM_EDGE_HORIZONTAL;
          }
          pGr[index] = sum < pHeat[index] ? pHeat[index] : sum;
      }
  }

  for (i = 0; i < nRows; i++) {//第一列，最后一列
      index = i * nCols;
      // Absolute value
      abs_dx = UPM_ABS(pDX[index]);
      // Absolute value
      abs_dy = UPM_ABS(pDY[index]);
      sum = abs_dx + abs_dy;
      // Divide by 2 the gradient
      pGrWO[index] = sum;
      pGr[index] = sum < gradientTh ? 0 : sum;
      // Select between vertical or horizontal gradient
      pDir[index] = abs_dx >= abs_dy ? UPM_EDGE_VERTICAL : UPM_EDGE_HORIZONTAL;

      index += nCols - 1;
      abs_dx = UPM_ABS(pDX[index]);
      abs_dy = UPM_ABS(pDY[index]);
      sum = abs_dx + abs_dy;
      pGrWO[index] = sum;
      pGr[index] = sum < gradientTh ? 0 : sum;
      pDir[index] = abs_dx >= abs_dy ? UPM_EDGE_VERTICAL : UPM_EDGE_HORIZONTAL;
  }

  for (i = 0; i < nCols; i++) {
      index = i;
      abs_dx = UPM_ABS(pDX[index]);
      abs_dy = UPM_ABS(pDY[index]);
      sum = abs_dx + abs_dy;
      pGrWO[index] = sum;
      pGr[index] = sum < gradientTh ? 0 : sum;
      pDir[index] = abs_dx >= abs_dy ? UPM_EDGE_VERTICAL : UPM_EDGE_HORIZONTAL;
     
      index += (nRows - 1) * nCols;
      abs_dx = UPM_ABS(pDX[index]);
      abs_dy = UPM_ABS(pDY[index]);
      sum = abs_dx + abs_dy;
      pGrWO[index] = sum;
      pGr[index] = sum < gradientTh ? 0 : sum;
      pDir[index] = abs_dx >= abs_dy ? UPM_EDGE_VERTICAL : UPM_EDGE_HORIZONTAL;
  }

   return dstInfo;
}

inline void DEALSD::computeAnchorPoints(const cv::Mat &dirImage,
                                       const cv::Mat &gradImageWO,
                                       const cv::Mat &gradImage,
                                       const cv::Mat &heatmap,
                                       std::vector<Pixel> &anchorPoints) {  

  int imageWidth = gradImage.cols;   
  int imageHeight = gradImage.rows;
  const auto* heat = heatmap.ptr<float>();  

  int indexInArray;
  unsigned int w, h;
  for (w = 1; w < imageWidth - 1; w++) {  
    for (h = 1; h < imageHeight - 1; h++){
      indexInArray = h * imageWidth + w;
      if (heat[indexInArray] == 1) {
          anchorPoints.emplace_back(w, h);
      }
    }
  }
}

void DEALSD::clear() {
  imgInfo = nullptr;
  edges.clear();
  segments.clear();
  salientSegments.clear();
  anchors.clear();
  drawer = nullptr;
  blurredImg = cv::Mat();
  edgeImg = cv::Mat();
}

inline int calculateNumPtsToTrim(int nPoints) {
  return std::min(5.0, nPoints * 0.1);
}

// Linear interpolation. s is the starting value, e the ending value
// and t the point offset between e and s in range [0, 1]
inline float lerp(float s, float e, float t) { return s + (e - s) * t; }

// Bi-linear interpolation of point (tx, ty) in the cell with corner values [[c00, c01], [c10, c11]]
inline float blerp(float c00, float c10, float c01, float c11, float tx, float ty) {
  return lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty);
}

void DEALSD::drawAnchorPoints(const uint8_t *dirImg,
                             const std::vector<Pixel> &anchorPoints,
                             uint8_t *pEdgeImg) {
  assert(imgInfo && imgInfo->imageWidth > 0 && imgInfo->imageHeight > 0);
  assert(!imgInfo->gImg.empty() && !imgInfo->dirImg.empty() && pEdgeImg);
  assert(drawer);
  assert(!edgeImg.empty());

  int imageWidth = imgInfo->imageWidth;
  int imageHeight = imgInfo->imageHeight;
  bool expandHorizontally;
  int indexInArray;
  unsigned char lastDirection;  // up = 1, right = 2, down = 3, left = 4;

  if (anchorPoints.empty()) {
    // No anchor points detected in the image
    return;
  }

  const double validationTh = params.validationTh;

  for (const auto &anchorPoint: anchorPoints) {
    // LOGD << "Managing new Anchor point: " << anchorPoint;
    indexInArray = anchorPoint.y * imageWidth + anchorPoint.x;

    if (pEdgeImg[indexInArray]) {
      // If anchor i is already been an edge pixel
      continue;
    }

    // If the direction of this pixel is horizontal, then go left and right.
    expandHorizontally = dirImg[indexInArray] == UPM_EDGE_HORIZONTAL;

    /****************** First side Expanding (Right or Down) ***************/
    // Select the first side towards we want to move. If the gradient points
    // horizontally select the right direction and otherwise the down direction.
    lastDirection = expandHorizontally ? UPM_RIGHT : UPM_DOWN;

    drawer->drawEdgeInBothDirections(lastDirection, anchorPoint);
  }

  double theta, angle;
  float saliency;
  bool valid;
  int endpointDist, nOriInliers, nOriOutliers;
  float probability;
#ifdef UPM_SD_USE_REPROJECTION
  cv::Point2f p;
  float lerp_dx, lerp_dy, lerp_p;
  int x0, y0, x1, y1;
#endif
  float *pDx = imgInfo->dxImg.ptr<float>();
  float *pDy = imgInfo->dyImg.ptr<float>();
  float* pHeat = imgInfo->heatmap.ptr<float>();
  float* pGr = imgInfo->gImg.ptr<float>();


  segments.reserve(drawer->getDetectedFullSegments().size());
  salientSegments.reserve(drawer->getDetectedFullSegments().size());

  for (const FullSegmentInfo &detectedSeg: drawer->getDetectedFullSegments()) {

    valid = true;
    probability = detectedSeg.getAveHeat();

    if (params.validate) {
      
        // Get the segment angle
        Segment s = detectedSeg.getEndpoints();
        theta = segAngle(s) + M_PI_2;
        // Force theta to be in range [0, M_PI)
        while (theta < 0) theta += M_PI;
        while (theta >= M_PI) theta -= M_PI;

        // Calculate the line equation as the cross product os the endpoints
        cv::Vec3f l = cv::Vec3f(s[0], s[1], 1).cross(cv::Vec3f(s[2], s[3], 1));
        // Normalize the line direction
        l /= std::sqrt(l[0] * l[0] + l[1] * l[1]);
        cv::Point2f perpDir(l[0], l[1]);

        // For each pixel in the segment compute its angle
        int nPixelsToTrim = calculateNumPtsToTrim(detectedSeg.getNumOfPixels());

        Pixel firstPx = detectedSeg.getFirstPixel();
        Pixel lastPx = detectedSeg.getLastPixel();

        nOriInliers = 0;
        nOriOutliers = 0;

        for (auto px: detectedSeg) {

          // If the point is not an inlier avoid it
          if (edgeImg.at<uint8_t>(px.y, px.x) != UPM_ED_SEGMENT_INLIER_PX) {
            continue;
          }

          endpointDist = detectedSeg.horizontal() ?
                         std::min(std::abs(px.x - lastPx.x), std::abs(px.x - firstPx.x)) :
                         std::min(std::abs(px.y - lastPx.y), std::abs(px.y - firstPx.y));

          if (endpointDist < nPixelsToTrim) {
            continue;
          }

#ifdef UPM_SD_USE_REPROJECTION
          // Re-project the point into the segment. To do this, we should move pixel.dot(l)
          // units (the distance between the pixel and the segment) in the direction
          // perpendicular to the segment (perpDir).
          p = cv::Point2f(px.x, px.y) - perpDir * cv::Vec3f(px.x, px.y, 1).dot(l);//cv::Vec3f(px.x, px.y, 1).dot(l)点到直线的距离，此操作将px移动到直线上,p为px在线段上的投影
          // Get the values around the point p to do the bi-linear interpolation
          x0 = p.x < 0 ? 0 : p.x;
          if (x0 >= imageWidth) x0 = imageWidth - 1;
          y0 = p.y < 0 ? 0 : p.y;
          if (y0 >= imageHeight) y0 = imageHeight - 1;
          x1 = p.x + 1;
          if (x1 >= imageWidth) x1 = imageWidth - 1;
          y1 = p.y + 1;
          if (y1 >= imageHeight) y1 = imageHeight - 1;


          /*对于概率点,若概率>平均概率，可认为是支持点*/
          if (pGr[y0 * imageWidth + x0] < 1) {
              lerp_p= blerp(pHeat[y0 * imageWidth + x0], pHeat[y0 * imageWidth + x1],
                            pHeat[y1 * imageWidth + x0], pHeat[y1 * imageWidth + x1],
                            p.x - int(p.x), p.y - int(p.y));

              if (5*lerp_p>(1+ probability)*2)  nOriInliers++;
              else  nOriOutliers++;
              continue;
          }
          
          //Bi-linear interpolation of Dx and Dy
          lerp_dx = blerp(pDx[y0 * imageWidth + x0], pDx[y0 * imageWidth + x1],   
                          pDx[y1 * imageWidth + x0], pDx[y1 * imageWidth + x1],
                          p.x - int(p.x), p.y - int(p.y));
          lerp_dy = blerp(pDy[y0 * imageWidth + x0], pDy[y0 * imageWidth + x1],
                          pDy[y1 * imageWidth + x0], pDy[y1 * imageWidth + x1],
                          p.x - int(p.x), p.y - int(p.y));
          // Get the gradient angle
          angle = std::atan2(lerp_dy, lerp_dx);
#else
          indexInArray = px.y * imageWidth + px.x;
          angle = std::atan2(pDy[indexInArray], pDx[indexInArray]);
#endif
          // Force theta to be in range [0, M_PI)
          if (angle < 0) angle += M_PI;
          if (angle >= M_PI) angle -= M_PI;
          circularDist(theta, angle, M_PI) > validationTh ? nOriOutliers+=2 : nOriInliers+=2;
        }

        valid = nOriInliers >= nOriOutliers;
        saliency = nOriInliers;
    //  }
    } else {
      saliency = segLength(detectedSeg.getEndpoints());
    }
    if (valid) {
      const Segment &endpoints = detectedSeg.getEndpoints();
      segments.push_back(endpoints);
      salientSegments.emplace_back(endpoints, saliency);
    }
  }
}

ImageEdges DEALSD::getAllEdges() const {
  assert(drawer);
  ImageEdges result;
  for (const FullSegmentInfo &s: drawer->getDetectedFullSegments())
    result.push_back(s.getPixels());
  return result;
}

ImageEdges DEALSD::getSegmentEdges() const {
  assert(drawer);
  ImageEdges result;
  for (const FullSegmentInfo &s: drawer->getDetectedFullSegments())
    result.push_back(s.getPixels());
  return result;
}

const LineDetectionExtraInfoPtr &DEALSD::getImgInfoPtr() const {
  return imgInfo;
}

}  // namespace upm
