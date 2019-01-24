#include "GetFeatPoint.hpp"
#include "SceneOptFlow.hpp"
#include <stdio.h>
#include <stdlib.h>

#define clip( val, minv, maxv )    (( (val) = ((val) < (minv) ? (minv) : (val) ) ) > (maxv) ? (maxv) : (val) )

namespace OptFlowTrk{

static void KeyPointsToPoints(const vector<KeyPoint>& kps, vector<Point2f>& ps)
{
	ps.clear();
	for (unsigned int i=0; i<kps.size(); i++) ps.push_back(kps[i].pt);
}

void FeatPointExtract::extractGoodFP(cv::Mat src_gray, bool bScaleDetect, double minDistance, std::vector<Point2f> &fpVector)
{
	double qualityLevel = 0.01;
	int blockSize = 3;
	bool useHarrisDetector = false;
	double k = 0.04;
	int maxCorners = 100;
	int i, ncornes;
	cv::Mat detectImage;
	Size subPixWinSize(10,10);
	TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);

	CV_Assert(src_gray.channels() == 1);

	int64 tstart = getTickCount();
	if(bScaleDetect){
#if 1
		resize(src_gray, detectImage, cv::Size(src_gray.cols/2, src_gray.rows/2));//cost time litter
#else
		pyrDown(src_gray, detectImage);
#endif
		minDistance /=2.0;
	}else{
		detectImage = src_gray;
	}
//	printf("resize: time = %f sec \n\n", ( (getTickCount() - tstart)/getTickFrequency()) );

	cv::goodFeaturesToTrack( detectImage,
					fpVector,
				   maxCorners,
				   qualityLevel,
				   minDistance,
				   Mat(),
				   blockSize,
				   useHarrisDetector,
				   k );

	if(bScaleDetect){
		ncornes = fpVector.size();
		for(i=0; i<ncornes; i++){
			fpVector[i].x *=2;
			fpVector[i].y *=2;
		}
	}
//	if(ncornes >0)
//		cornerSubPix(src_gray, fpVector, subPixWinSize, Size(-1,-1), termcrit);
}

void FeatPointExtract::extractFastFP(cv::Mat src_gray, bool bScaleDetect, std::vector<Point2f> &fpVector)
{
	fpVector.clear();
	std::vector<KeyPoint> keypoints;
	fdFAST->detect(src_gray, keypoints);
	KeyPointsToPoints(keypoints, fpVector);

}

void FeatPointExtract::extractORBFP(cv::Mat src_gray, bool bScaleDetect, std::vector<Point2f> &fpVector)
{
	int i, ncornes;
	fpVector.clear();
	std::vector<KeyPoint> keypoints;
	cv::Mat detectImage;
	if(bScaleDetect){
		resize(src_gray, detectImage, cv::Size(src_gray.cols/2, src_gray.rows/2));
	}else{
		detectImage = src_gray;
	}

	fdORB->detect(detectImage, keypoints);
	KeyPointsToPoints(keypoints, fpVector);

	if(bScaleDetect){
		ncornes = fpVector.size();
		for(i=0; i<ncornes; i++){
			fpVector[i].x *=2;
			fpVector[i].y *=2;
		}
	}
}

void FeatPointExtract::extractGradFP(cv::Mat src_gray, bool bScaleDetect, std::vector<Point2f> &fpVector)
{
	fpVector.clear();
	std::vector<KeyPoint> keypoints;
	fdGridFAST->detect(src_gray, keypoints);
	KeyPointsToPoints(keypoints, fpVector);

}

FeatPointExtract::FeatPointExtract()
{
	minDistance = 10.0;
	/*
	 * ORB(int nfeatures = 500, float scaleFactor = 1.2f, int nlevels = 8, int edgeThreshold = 31,
   	* 	 		 int firstLevel = 0, int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, int patchSize=31 );
   	*/
	fdORB = makePtr<ORB>();
	/*
	 *  FastFeatureDetector( int threshold=10, bool nonmaxSuppression=true );
	 */
	fdFAST = makePtr<FastFeatureDetector>();
	/*
	 * GridAdaptedFeatureDetector( const Ptr<FeatureDetector>& detector=0,
	 *	int maxTotalKeypoints=1000,
	 *	int gridRows=4, int gridCols=4 );
	 */
	fdGridFAST = makePtr<GridAdaptedFeatureDetector>();
}

FeatPointExtract::~FeatPointExtract()
{
}

void FeatPointExtract::ExtractFP(cv::Mat src_gray, std::vector<Point2f> &fpVector, bool bScaleDetect, int mode /*= -1*/)
{
	if(mode == GOOD_FEATPOINT){
		extractGoodFP(src_gray, bScaleDetect, minDistance, fpVector);
	}else if(mode == FAST_FEATPOINT){
		extractFastFP(src_gray, bScaleDetect, fpVector);
	}else if(mode == ORB_FEATPOINT){
		extractORBFP(src_gray, bScaleDetect, fpVector);
	}else if(mode == GRAD_FEATPOINT){
		extractGradFP(src_gray, bScaleDetect, fpVector);
	}else{
		fpVector.clear();
	}
}

}
