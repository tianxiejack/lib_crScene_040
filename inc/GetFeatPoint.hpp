#ifndef 		_GET_FEAT_POINT_
#define 	_GET_FEAT_POINT_

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

using namespace cv;
using namespace std;

namespace OptFlowTrk{

class FeatPointExtract{

public:
	FeatPointExtract();
	virtual ~FeatPointExtract();

	void SetMinDistance(double distance){minDistance = distance;};

	void ExtractFP(cv::Mat src_gray, std::vector<Point2f> &fpVector, bool bScaleDetect, int mode = -1);

private:
	void extractGoodFP(cv::Mat src_gray, bool bScaleDetect, double minDistance, std::vector<Point2f> &fpVector);
	void extractFastFP(cv::Mat src_gray, bool bScaleDetect, std::vector<Point2f> &fpVector);
	void extractORBFP(cv::Mat src_gray, bool bScaleDetect, std::vector<Point2f> &fpVector);
	void extractGradFP(cv::Mat src_gray, bool bScaleDetect, std::vector<Point2f> &fpVector);
private:
	Ptr<FeatureDetector> fdORB;
	Ptr<FeatureDetector> fdFAST;
	Ptr<FeatureDetector> fdGridFAST;

	double minDistance;
};

}
#endif
