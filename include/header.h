#ifndef DAOZHA_INCLUDE_HEADER_H_
#define DAOZHA_INCLUDE_HEADER_H_

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <ctime>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
using namespace nvinfer1;

struct KeyPoints{
    cv::Point2f p;
    float prob;
   
};
struct  Object{
  cv::Rect_<float> rect;
  int label;
  float prob;
  std::vector<KeyPoints>result_kp;

};


#endif