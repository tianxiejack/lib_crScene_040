#include "SceneOptFlow.hpp"
#include <stdio.h>

using namespace cv;
using namespace std;

namespace OptFlowTrk{

class SceneMVCalImpl: public SceneOptFlow{

public:
	SceneMVCalImpl( const SceneOptFlow::Params &parameters = SceneOptFlow::Params() );

protected:

	bool initImpl( const Mat& image, const Rect2d& validBox );

	void uninitImpl(){};

	bool updateImpl( const Mat& image, Point2f& mvPos );

	cv::Mat getSceneMatrixImpl(){return sceneMatrix;};

	void getFeatPointsImpl(std::vector<Point2f> &fpVector);

	void  setRefImage(cv::Mat image){ image.copyTo(image_ref);};

	void getFeatureStruct(const cv::Mat image, cv::Mat &featMap);

	 void getMaxVarRegionsImpl(std::vector<cv::Rect> &rcVector){};

	SceneOptFlow::Params params;

private:
	cv::Mat image_ref;
	cv::Mat featImg_ref;
	int 		m_frameIdx;
	cv::Mat sceneMatrix;
};



}
