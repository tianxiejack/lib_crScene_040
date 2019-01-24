#ifndef		_MAX_VAR_REGION_TRK_HPP
#define		_MAX_VAR_REGION_TRK_HPP

#include "SceneOptFlow.hpp"
#include "MaxVarFeatTrk.hpp"
#include <omp.h>
#include <stdio.h>

using namespace cv;
using namespace std;

namespace OptFlowTrk{

#define		MAX_VAR_BLK	4
typedef struct _scene_blk_t{
	cv::Rect	refPos;
	cv::Rect	curPos;
}SCENE_BLK;

typedef struct _scene_info{
	cv::Point2f		pos;
	float				opt;
	float 				refVar;
}SCENE_INFO;

typedef struct _cluster_t{
	std::vector<int> indexVec;
	cv::Point2f		pos;
}CLUSTER_OBJ;

class MaxVarRegionImpl: public SceneOptFlow{

public:
	MaxVarRegionImpl( const SceneOptFlow::Params &parameters = SceneOptFlow::Params() );

protected:

	bool initImpl( const Mat& image, const Rect2d& validBox );

	void uninitImpl(){};

	bool updateImpl( const Mat& image, Point2f& mvPos );

	cv::Mat getSceneMatrixImpl(){return sceneMatrix;};

	void getFeatPointsImpl(std::vector<Point2f> &fpVector){};

	void getMaxVarRegionsImpl(std::vector<cv::Rect> &rcVector);

	void setRefImage(cv::Mat image);

	void buildGrid(const cv::Mat &image);

	float bbOverlap(const BoundingBox& box1,const BoundingBox& box2);

	void extractMaxVarBlk(const cv::Mat &image);

	double getVar(const BoundingBox& box,const Mat& sum,const Mat& sqsum);

	void compSceneSimilar(const cv::Mat _curImg, const cv::Mat _tmplModel, SCENE_INFO &posConf, SCENE_BLK &blkInfo, float &apce);

	bool calSceneSimilar(const cv::Mat image);

	void clusterAnalysis();

	SceneOptFlow::Params params;

	SCENE_INFO		m_curOptConf;
	bool					m_bOptConf;

private:
	cv::Mat image_ref;
	cv::Size	imageSize;

	std::vector<BoundingBox> grid;
	std::vector<BoundingBox> bestgrid;

	std::vector<SCENE_BLK>		m_blk;

	cv::Mat						m_refSceneMap[MAX_VAR_BLK];
	cv::Mat						m_curSceneMap[MAX_VAR_BLK];
	SCENE_INFO				m_sceneInfo[MAX_VAR_BLK];
	float							m_apce[MAX_VAR_BLK];

	//Integral Images
	cv::Mat iisum;
	cv::Mat iisqsum;
	float var;//threshold

	 cv::Mat sceneMatrix;
};


}
#endif
