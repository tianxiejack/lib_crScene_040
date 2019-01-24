#ifndef 		SCENE_OPT_FLOW
#define		SCENE_OPT_FLOW

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

using namespace cv;
using namespace std;

namespace OptFlowTrk{

typedef enum
{
	GOOD_FEATPOINT = 0x00,
	FAST_FEATPOINT,
	ORB_FEATPOINT,
	GRAD_FEATPOINT,
	MATCH_FEATPOINT,
	MAX_VAR_FEATHER,
	MAX_VAR_REGION,
}FEATPOINT_MODE;

typedef enum
{
    TRANSLATION = 0,
    TRANSLATION_AND_SCALE = 1,
    LINEAR_SIMILARITY = 2,
    AFFINE = 3
}MotionModel;

struct  RansacParams
{
    int size; // subset size
    float thresh; // max error to classify as inlier
    float eps; // max outliers ratio
    float prob; // probability of success

    RansacParams(int _size = 6, float _thresh = 0.5, float _eps = 0.5, float _prob=0.99)
        : size(_size), thresh(_thresh), eps(_eps), prob(_prob) {}

    int niters() const
	{
		return static_cast<int>(
				std::ceil(std::log(1 - prob) / std::log(1 - std::pow(1 - eps, size))));
	}
/*
    RansacParams &operator = (const RansacParams &params){
    	this->size = params.size;	this->thresh = params.thresh;	this->eps = params.eps;	this->prob = params.prob;
    	return *this;
    }*/

    static RansacParams translationMotionStd() { return RansacParams(4, 0.5f, 0.5f, 0.99f); }
    static RansacParams translationAndScale2dMotionStd() { return RansacParams(4, 0.5f, 0.5f, 0.99f); }
    static RansacParams linearSimilarityMotionStd() { return RansacParams(6, 0.5f, 0.5f, 0.99f); }
    static RansacParams affine2dMotionStd() { return RansacParams(6, 0.5f, 0.5f, 0.99f); }
};

typedef Rect_<double> Rect2d;

class SceneOptFlow{

public:
	struct  Params
	{
		Params();
		int nfeatures;
		int nframes;
		bool	bContinueFrms;
		bool   bScaleDetect;
		float   minDistance;
		bool   bRejectOutliner;
		bool   bFBTrk;
		bool	bZoomHalf;//shrink image
		int pointsInGrid;

		float maxRmse_;
		float minInlierRatio_;
		MotionModel motionModel_;
		RansacParams ransacParams_;

		FEATPOINT_MODE	fpMode;
	};

	virtual ~SceneOptFlow();

	bool init( const Mat& image, const Rect2d& validBox);

	void uninit();

	bool update( const Mat& image, Point2f& mvPos );

	cv::Mat getSceneMatrix();

	void getFeatPoints(std::vector<Point2f> &fpVector);

	void getMaxVarRegions(std::vector<cv::Rect> &rcVector);

	static Ptr<SceneOptFlow> createInterface(const SceneOptFlow::Params &parameters = SceneOptFlow::Params());

protected:

	virtual bool initImpl( const Mat& image, const Rect2d& validBox ) = 0;

	virtual void uninitImpl() = 0;

	virtual bool updateImpl( const Mat& image, Point2f& mvPos ) = 0;

	virtual void getFeatPointsImpl(std::vector<Point2f> &fpVector) = 0;

	virtual void getMaxVarRegionsImpl(std::vector<cv::Rect> &rcVector) = 0;

	virtual cv::Mat getSceneMatrixImpl() = 0;

	bool isInit;

private:

};

}


#endif
