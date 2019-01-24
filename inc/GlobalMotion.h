#ifndef 		_GLOBAL_MOTION_
#define		_GLOBAL_MOTION_
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "SceneOptFlow.hpp"

using namespace std;
using namespace cv;

namespace OptFlowTrk
{
Mat estimateGlobalMotionLeastSquares(
        const std::vector<Point2f> &points0, const std::vector<Point2f> &points1,
        int model = AFFINE, float *rmse = 0);


 Mat estimateGlobalMotionRobust(
        std::vector<Point2f> &points0, std::vector<Point2f> &points1,
        int model = AFFINE, const RansacParams &params = RansacParams::affine2dMotionStd(),
        float *rmse = 0, int *ninliers = 0);

 Mat estimateGlobalMotionGMS(
         const std::vector<Point2f> &points0, const std::vector<Point2f> &points1,cv::Size sz,
         int model = AFFINE, const RansacParams &params = RansacParams::affine2dMotionStd(),
         float *rmse = 0, int *ninliers = 0);

};


#endif
