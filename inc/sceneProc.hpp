#ifndef 		_SCENE_PROCE_HPP_
#define		_SCENE_PROCE_HPP_

#include "SceneOptFlow.hpp"
#include "kalman_filters.hpp"
#include "MedianFlowTrk.hpp"

using namespace cv;
using namespace std;
using namespace OptFlowTrk;

namespace OptFlowTrk{

typedef unsigned int UInt32;

#define		MAX_SCENE_FRAMES		200

typedef struct{
	UInt32 ts; //us
	UInt32 deltaT;
	cv::Point2f	mv;
	float confidence;
}SceneState;

typedef enum{
	INIT_SCENE = 0x0,
	RUN_SCENE,
}SCENE_STATUS;

typedef std::vector<SceneState>::iterator		StateVectorIterator;

class	SceneProc{
public:
	SceneProc();
	virtual ~SceneProc();

	void InitSceneLock(const cv::Mat image);
	void CalSceneLock(const cv::Mat image);
	void  AnalyseSceneLock();

	void SetStandardSize(cv::Size sz);
	void InitUKFilter(float x, float y);

	SCENE_STATUS GetSceneStatus(){return sceneStatus;};
	void SetSceneStatus(SCENE_STATUS status){sceneStatus = status;};
	void SetSceneParam(const cv::Mat &image, SceneOptFlow::Params &parameters);


	void optFlowInitSceneLock(const cv::Mat image);
	bool optFlowCalcSceneLock(const cv::Mat image);


public:

	cv::Point2f						m_filterVel;//ukf filter result
	cv::Point2f						m_instanVel;
	float								m_instanConfd;
	cv::Size							imageSize;

protected:

	std::vector<SceneState>		m_sceneState;


	Ptr<SceneOptFlow>			m_ScenePtr;
	Ptr<UnscentedKalmanFilter> 	uncsentedKalmanFilter;
	Ptr<UkfSystemModel> 			model;

	Ptr<MedianFlowTracker> 		m_medianFlowTracker;	

	RNG m_rng;
	double	m_accNoise;
	UInt32 m_bakTS;

private:
	cv::Size	standardSize;
	SCENE_STATUS	sceneStatus;


	cv::Point2f							m_sumDelta;
	cv::Point2f							m_bakDetlta;//pre filter result

};

}

#endif

