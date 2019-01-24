#ifndef 		_MAX_VAR_FEAT_TRK_HPP
#define		_MAX_VAR_FEAT_TRK_HPP

#include "SceneOptFlow.hpp"
#include <stdio.h>

using namespace cv;
using namespace std;

namespace OptFlowTrk{

//Bounding Boxes
struct BoundingBox : public cv::Rect {
  BoundingBox(){var=0.f;}
  BoundingBox(cv::Rect r): cv::Rect(r){var =0.f;}
public:
  float var;        //Overlap with current Bounding Box

};

struct BComparator{
	BComparator(const std::vector<BoundingBox>& _grid):grid(_grid){}
  std::vector<BoundingBox> grid;
  bool operator()(BoundingBox idx1,BoundingBox idx2){
    return idx1.var > idx2.var;
  }
};

class MaxVarFeatImpl: public SceneOptFlow{

public:
	MaxVarFeatImpl( const SceneOptFlow::Params &parameters = SceneOptFlow::Params() );

protected:

	bool initImpl( const Mat& image, const Rect2d& validBox );

	void uninitImpl(){};

	bool updateImpl( const Mat& image, Point2f& mvPos );

	cv::Mat getSceneMatrixImpl(){return sceneMatrix;};

	void getFeatPointsImpl(std::vector<Point2f> &fpVector);

	void getMaxVarRegionsImpl(std::vector<cv::Rect> &rcVector){};

	void setRefImage(cv::Mat image){ image.copyTo(image_ref);};

	void buildGrid(const cv::Mat &image);

	float bbOverlap(const BoundingBox& box1,const BoundingBox& box2);

	void extractMaxVarBlk(const cv::Mat &image);

	double getVar(const BoundingBox& box,const Mat& sum,const Mat& sqsum);

	void betweenframeTrk(const cv::Mat oldImage_gray,const cv::Mat newImage_gray,
			std::vector<Point2f> &pointsToTrackOld, std::vector<Point2f> &pointsToTrackNew );

	bool judgeFPDistrib(const Mat &image, const std::vector<Point2f> &ptsvector);

	SceneOptFlow::Params params;

private:
	cv::Mat image_ref;
	cv::Size	imageSize;
	std::vector<BoundingBox> grid;
	std::vector<BoundingBox> bestgrid;

	std::vector<Point2f> pointsPrev_, points_;
	std::vector<Point2f> pointsPrevGood_, pointsGood_;

	//Integral Images
	  cv::Mat iisum;
	  cv::Mat iisqsum;
	  float var;//threshold

	  std::vector<uchar> status;
	  std::vector<float> errors;

	  TermCriteria termcrit;
	  cv::Mat sceneMatrix;
};

}




#endif
