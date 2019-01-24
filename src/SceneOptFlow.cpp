#include "SceneOptFlow.hpp"
#include "GetFeatPoint.hpp"
#include "GlobalMotion.h"
#include "RejectOutliner.hpp"
#include <stdio.h>
#include "SceneMVCal.hpp"
#include "MaxVarFeatTrk.hpp"
#include "MaxVarRegionTrk.hpp"

namespace OptFlowTrk{

SceneOptFlow::~SceneOptFlow()
{
}

bool SceneOptFlow::init( const Mat& image, const Rect2d& validBox)
{
	if( isInit )
	{
//		return false;
		uninit();
	}

	if( image.empty() )
		return false;

	bool initTracker = initImpl( image, validBox );

	if( initTracker )
	{
		isInit = true;
	}

	return initTracker;
}

void SceneOptFlow::uninit()
{
	isInit =false;
	uninitImpl();
}

bool SceneOptFlow::update( const Mat& image, Point2f& mvPos )
{
	if( !isInit )
	{
		return false;
	}

	if( image.empty() )
		return false;

	return updateImpl( image, mvPos );
}

cv::Mat SceneOptFlow::getSceneMatrix()
{
	if( !isInit )
	{
		return Mat::eye(3, 3, CV_32F);
	}
	return getSceneMatrixImpl();
}

void SceneOptFlow::getFeatPoints(std::vector<Point2f> &fpVector)
{
	fpVector.clear();
	if( !isInit )
	{
		return ;
	}

	getFeatPointsImpl(fpVector);
}

void SceneOptFlow::getMaxVarRegions(std::vector<cv::Rect> &rcVector)
{
	rcVector.clear();
	if( !isInit )
	{
		return ;
	}

	getMaxVarRegionsImpl(rcVector);
}

typedef enum{
		LOST_FEATPOINT = -1,
		IDLE_FEATPOINT	= 0,
		NEWSEL_FEATPOINT,
		TRACK_FEATPOINT,
}FP_Type;

typedef struct{
	cv::Point2f pos;
	FP_Type type;
	int frmIdx;
}FeatPoint_t;

typedef std::vector< std::pair<int, std::vector<FeatPoint_t> > > OptFlowFPVector;

static double l2distance(Point2f p1,Point2f p2)
{
    double dx=p1.x-p2.x, dy=p1.y-p2.y;
    return sqrt(dx*dx+dy*dy);
}

template<typename T>
T getMedian(std::vector<T>& values,int size)
{
    if(size==-1){
        size=(int)values.size();
    }
    std::vector<T> copy(values.begin(),values.begin()+size);
    std::sort(copy.begin(),copy.end());
    if(size%2==0){
        return (copy[size/2-1]+copy[size/2])/((T)2.0);
    }else{
        return copy[(size-1)/2];
    }
}

class SceneOptFlowImpl: public SceneOptFlow{

public:
	SceneOptFlowImpl( const SceneOptFlow::Params &parameters = SceneOptFlow::Params() );

protected:

	bool initImpl( const Mat& image, const Rect2d& validBox );

	void uninitImpl(){};

	bool updateImpl( const Mat& image, Point2f& mvPos );

	cv::Mat getSceneMatrixImpl(){return sceneMatrix;};

	void getFeatPointsImpl(std::vector<Point2f> &fpVector);

	void getMaxVarRegionsImpl(std::vector<cv::Rect> &rcVector){};

	bool selectFeatPoints(const Mat& image, bool init = false);

	bool storeFeatPoints();

	void  refineFeatPoints(const Mat& image, const vector<Point2f> corners);

	void  setRefImage(cv::Mat image){ image.copyTo(image_ref);};

	void continueframeTrk(const cv::Mat oldImage_gray,const cv::Mat newImage_gray,
			std::vector<Point2f> &pointsToTrackOld, std::vector<Point2f> &pointsToTrackNew );

	void betweenframeTrk(const cv::Mat oldImage_gray,const cv::Mat newImage_gray,
			std::vector<Point2f> &pointsToTrackOld, std::vector<Point2f> &pointsToTrackNew );

	void ForwardBackTrk(const cv::Mat oldImage_gray,const cv::Mat newImage_gray,
			const std::vector<Point2f> pointsToTrackOld, const std::vector<Point2f> pointsToTrackNew );

	void rejectOutlinerProc(const Mat& image);

	bool judgeFPDistrib(const Mat &image, const std::vector<Point2f> &ptsvector);

	SceneOptFlow::Params params;
	FeatPointExtract	m_extractFeatPort;
	TranslationOutlierRejector		m_rejectOutliner;

private:
	cv::Mat image_ref;
	cv::Mat sceneMatrix;
	int		m_framIdx;
	TermCriteria termcrit;

	OptFlowFPVector m_fpVector;
	std::vector<FeatPoint_t>	m_refFeatPt, m_curFeatPt;

	vector<uchar> status;
	vector<float> errors;
	vector<int> featIdxVec;
	std::vector<uchar> fb_status;
	vector<float> fb_errors;
	std::vector<uchar> filter_status;

	std::vector<Point2f> pointsPrev_, points_;
	std::vector<Point2f> pointsPrevGood_, pointsGood_;
};

bool SceneOptFlowImpl::initImpl( const Mat& image, const Rect2d& validBox )
{
	if(params.bZoomHalf){
//		resize(image, image_ref, cv::Size(image.cols/2, image.rows/2),  0.0, 0.0, INTER_CUBIC);
		resize(image, image_ref, cv::Size(image.cols/2, image.rows/2));
	}else{
		setRefImage(image);
	}

	m_framIdx = 0;
	for(int i=0; i<params.nfeatures; i++){
		m_fpVector[i].first = i;
		m_fpVector[i].second.clear();

		m_refFeatPt[i].frmIdx = m_framIdx;
		m_refFeatPt[i].type = IDLE_FEATPOINT;
		m_refFeatPt[i].pos = cv::Point2f(-1.f, -1.f);

		m_curFeatPt[i].frmIdx = m_framIdx;
		m_curFeatPt[i].type = IDLE_FEATPOINT;
		m_curFeatPt[i].pos = cv::Point2f(-1.f, -1.f);
	}
	selectFeatPoints(image_ref, true);

	return true;
}

void SceneOptFlowImpl::continueframeTrk(const cv::Mat oldImage_gray,const cv::Mat newImage_gray,
		std::vector<Point2f>  &pointsToTrackOld, std::vector<Point2f> &pointsToTrackNew )
{
	int i, index, nsize = m_refFeatPt.size();
	for(i=0; i<nsize; i++){
		if(m_refFeatPt[i].type>=NEWSEL_FEATPOINT){
			pointsToTrackOld.push_back(m_refFeatPt[i].pos);
			featIdxVec.push_back(i);
		}
	}

	if(nsize > 0){
		calcOpticalFlowPyrLK(oldImage_gray, newImage_gray,pointsToTrackOld,pointsToTrackNew,status,errors,Size(5,5),4,termcrit,0);
	}

	if(params.bFBTrk){
		ForwardBackTrk(oldImage_gray, newImage_gray,pointsToTrackOld,pointsToTrackNew);
	}

	for(i=0;i<(int)pointsToTrackOld.size();i++){
		index = featIdxVec[i];
		if(status[i]==1){
			if(!(pointsToTrackNew[i].x<newImage_gray.cols &&pointsToTrackNew[i].y<newImage_gray.rows)){
				status[i] = false;
				m_curFeatPt[index].pos = Point2f(-1.f, -1.f);
				m_curFeatPt[index].type = LOST_FEATPOINT;
			}else{
				m_curFeatPt[index].pos = pointsToTrackNew[i];
				m_curFeatPt[index].type = TRACK_FEATPOINT;
			}
		}else{
			m_curFeatPt[index].pos = Point2f(-1.f, -1.f);
			m_curFeatPt[index].type = LOST_FEATPOINT;
		}
		m_curFeatPt[index].frmIdx = m_framIdx;
	}

}

void SceneOptFlowImpl::betweenframeTrk(const cv::Mat oldImage_gray,const cv::Mat newImage_gray,
		std::vector<Point2f> &pointsToTrackOld, std::vector<Point2f> &pointsToTrackNew )
{
	int i, nsize = pointsPrev_.size();
	pointsToTrackOld.resize(nsize);
	pointsToTrackNew.resize(nsize);

	for(i=0; i<nsize; i++){
		pointsToTrackOld[i] = pointsPrev_[i];
	}

	if(nsize > 0){
		calcOpticalFlowPyrLK(oldImage_gray, newImage_gray,pointsToTrackOld,pointsToTrackNew,status,errors,Size(5,5),4,termcrit,0);
	}

	if(params.bFBTrk){
		ForwardBackTrk(oldImage_gray, newImage_gray,pointsToTrackOld,pointsToTrackNew);
	}

	points_.resize(nsize);
	for(int i=0;i<nsize;i++){
		points_[i] = pointsToTrackNew[i];
		if(status[i]){
			if(!(pointsToTrackNew[i].x<newImage_gray.cols &&pointsToTrackNew[i].y<newImage_gray.rows)){
				status[i] = false;
			}
		}
	}
}

void SceneOptFlowImpl::ForwardBackTrk(const cv::Mat oldImage_gray,const cv::Mat newImage_gray,
		const std::vector<Point2f> pointsToTrackOld, const std::vector<Point2f> pointsToTrackNew )
{
	int i, npoints, index;
	std::vector<Point2f> pointsToTrackReprojection;
	std::vector<double> FBerror;

	std::vector<Point2f> newPoints, oldPoints;
	std::vector<int> featIndex;

	npoints = pointsToTrackNew.size();
	newPoints.reserve(npoints);
	oldPoints.reserve(npoints);
	featIndex.reserve(npoints);

	for(i=0; i<npoints; i++){
		if(status[i]){
			newPoints.push_back(pointsToTrackNew[i]);
			oldPoints.push_back(pointsToTrackOld[i]);
			featIndex.push_back(i);
		}
	}

	if(npoints>0){
		calcOpticalFlowPyrLK(oldImage_gray, newImage_gray,newPoints, pointsToTrackReprojection,fb_status,fb_errors,Size(5,5),4,termcrit,0);
	}

	FBerror.resize(newPoints.size());
	for(int i=0;i<(int)newPoints.size();i++){
		FBerror[i]=l2distance(oldPoints[i],pointsToTrackReprojection[i]);
	}

	double FBerrorMedian=getMedian(FBerror, (int)FBerror.size());
//	printf("FBerrorMedian=%f\n",FBerrorMedian);

	filter_status.resize(oldPoints.size());
	for(i=0;i<(int)newPoints.size();i++){
		index = featIndex[i];
		filter_status[i]=(FBerror[i]<FBerrorMedian);
		filter_status[i] = (filter_status[i]&&fb_status[i]);
		status[index] = filter_status[i];
	}
}

static unsigned int printCount = 0;
bool SceneOptFlowImpl::updateImpl( const Mat& image, Point2f& mvPos )
{
	std::vector<Point2f> pointsToTrackOld,pointsToTrackNew;
	Mat oldImage_gray,newImage_gray;
	bool iRtn = true;

	if(params.bZoomHalf){
		oldImage_gray = image_ref;
//		resize(image, newImage_gray, cv::Size(image.cols/2, image.rows/2), 0.0, 0.0, INTER_CUBIC);
		resize(image, newImage_gray, cv::Size(image.cols/2, image.rows/2));
	}else{
		oldImage_gray = image_ref;
		newImage_gray = image;
	}

	CV_Assert(oldImage_gray.channels() == 1);
	CV_Assert(newImage_gray.channels() == 1);

	m_framIdx++;
	sceneMatrix = cv::Mat::eye(3,3, CV_32F);

	status.clear();
	errors.clear();
	featIdxVec.clear();
	fb_status.clear();
	fb_errors.clear();
	filter_status.clear();

	pointsToTrackOld.clear();
	pointsToTrackNew.clear();

	int64 tstart;
	tstart = getTickCount();
	if(params.bContinueFrms){
		continueframeTrk(oldImage_gray, newImage_gray, pointsToTrackOld, pointsToTrackNew);
	}else{
		betweenframeTrk(oldImage_gray, newImage_gray, pointsToTrackOld, pointsToTrackNew);
	}
//	printf("OpticalFlowTrk: time = %f sec \n\n", ( (getTickCount() - tstart)/getTickFrequency()) );

	int  i, npoints = pointsToTrackNew.size();
	pointsPrevGood_.clear();
	pointsPrevGood_.reserve(npoints);
	pointsGood_.clear();
	pointsGood_.reserve(npoints);
	for (i = 0; i < npoints; ++i)
	{
		if (status[i])
		{
			pointsPrevGood_.push_back(pointsToTrackOld[i]);
			pointsGood_.push_back(pointsToTrackNew[i]);
		}
	}
	if(params.bRejectOutliner){
		rejectOutlinerProc(newImage_gray);
	}

	static MotionModel motionModelBK = LINEAR_SIMILARITY;
	bool bJudge = judgeFPDistrib(newImage_gray, pointsGood_);
	if(bJudge){
		params.motionModel_ = LINEAR_SIMILARITY;
		params.ransacParams_ = (RansacParams::linearSimilarityMotionStd());

	}else{
		params.motionModel_ = TRANSLATION;
		params.ransacParams_ = (RansacParams::translationMotionStd());
	}
	if(motionModelBK != params.motionModel_){
//		printf("%s:motionModel = %d \n",__func__, params.motionModel_);
	}

	motionModelBK = params.motionModel_;

	float rmse;
	int ninliers;
	std::vector<Point2f> pointsPrevGood_tmp, pointsGood_tmp;
	pointsPrevGood_tmp.clear();
	pointsGood_tmp.clear();

	pointsPrevGood_tmp.assign(pointsPrevGood_.begin(), pointsPrevGood_.end());
	pointsGood_tmp.assign(pointsGood_.begin(), pointsGood_.end());

	if(params.bZoomHalf){
		for(i=0; i<pointsPrevGood_.size(); i++){
			pointsPrevGood_tmp[i].x = pointsPrevGood_[i].x*2;
			pointsPrevGood_tmp[i].y = pointsPrevGood_[i].y*2;
		}
		for(i=0; i<pointsGood_.size(); i++){
			pointsGood_tmp[i].x = pointsGood_[i].x*2;
			pointsGood_tmp[i].y = pointsGood_[i].y*2;
		}
	}
#if 1
	Mat M = estimateGlobalMotionRobust(pointsPrevGood_tmp, pointsGood_tmp, params.motionModel_, params.ransacParams_, &rmse, &ninliers);
#else
	cv::Size sz = image.size();
	Mat M = estimateGlobalMotionGMS(pointsPrevGood_tmp, pointsGood_tmp, sz, params.motionModel_, params.ransacParams_, &rmse, &ninliers);
#endif
	pointsPrevGood_.clear();
	pointsGood_.clear();
	pointsPrevGood_.assign(pointsPrevGood_tmp.begin(), pointsPrevGood_tmp.end());
	pointsGood_.assign(pointsGood_tmp.begin(), pointsGood_tmp.end());

	if(params.bZoomHalf){
		for(i=0; i<pointsPrevGood_tmp.size(); i++){
			pointsPrevGood_[i].x = pointsPrevGood_tmp[i].x/2;
			pointsPrevGood_[i].y = pointsPrevGood_tmp[i].y/2;
		}
		for(i=0; i<pointsGood_tmp.size(); i++){
			pointsGood_[i].x = pointsGood_tmp[i].x/2;
			pointsGood_[i].y = pointsGood_tmp[i].y/2;
		}
	}

	printCount++;
	if (rmse > params.maxRmse_ || static_cast<float>(ninliers) / pointsGood_.size() < params.minInlierRatio_){
//		printf("rmse=%f, ninliers=%d, pointsGood_size=%d \n",rmse, ninliers, pointsGood_tmp.size());
		M = Mat::eye(3, 3, CV_32F);
		iRtn = false;
	}

	mvPos.x = M.at<float>(0,2);
	mvPos.y = M.at<float>(1,2);
	sceneMatrix = M.clone();

	tstart = getTickCount();
	selectFeatPoints(newImage_gray);
//	printf("SelectFeatPoints: time = %f sec \n\n", ( (getTickCount() - tstart)/getTickFrequency()) );

	setRefImage(newImage_gray);

	return iRtn;
}

bool SceneOptFlowImpl::judgeFPDistrib(const Mat &image, const std::vector<Point2f> &ptsvector)
{
	bool iRtn = true;
	int i, j, npoints = ptsvector.size();
	int w = image.cols;
	int h = image.rows;
	int grid_width = 4;
	int grid_height = 4;
	int cell_size_x = (w+grid_width-1)/grid_width;
	int cell_size_y = (h+grid_height-1)/grid_height;
	float x, y;
	int x_cell, y_cell;
	int count0, count1;
	float fration = 5.0;
	std::vector<std::vector<Point2f> > grid(grid_width*grid_height);

	for(i=0; i<npoints; i++){
		x = ptsvector[i].x;		y = ptsvector[i].y;
		x_cell = (int)(x / cell_size_x);		y_cell = (int)(y / cell_size_y);
		grid[y_cell*grid_width + x_cell].push_back(Point2f(x, y));
	}
	//up-down region
	count0 = 0; count1 = 0;
	for(j=0; j<grid_height; j++){
		for(i=0; i<grid_width; i++){
			if(j<grid_height/2){
				count0 += grid[j*grid_width + i].size();
			}else{
				count1 += grid[j*grid_width + i].size();
			}
		}
	}
	if(count1>count0*fration || count0>count1*fration){
		return false;
	}
	//left-right region
	count0 = 0; count1 = 0;
	for(j=0; j<grid_height; j++){
		for(i=0; i<grid_width; i++){
			if(i<grid_width/2){
				count0 += grid[j*grid_width + i].size();
			}else{
				count1 += grid[j*grid_width + i].size();
			}
		}
	}
	if(count1>count0*fration || count0>count1*fration){
		return false;
	}

	return iRtn;
}

void SceneOptFlowImpl::rejectOutlinerProc(const Mat& image)
{
	pointsPrev_.swap(pointsPrevGood_);
	points_.swap(pointsGood_);

	status.clear();
	m_rejectOutliner.process(image.size(), pointsPrev_, points_, status);

	pointsPrevGood_.clear();
	pointsPrevGood_.reserve(points_.size());

	pointsGood_.clear();
	pointsGood_.reserve(points_.size());

	for ( size_t i = 0; i < points_.size(); ++i)
	{
		if (status[i])
		{
			pointsPrevGood_.push_back(pointsPrev_[i]);
			pointsGood_.push_back(points_[i]);
		}
	}
}

void SceneOptFlowImpl::getFeatPointsImpl(std::vector<Point2f> &fpVector)
{
	int i, nsize;
	fpVector.clear();
//	if(params.bContinueFrms){
	if(0){
		nsize = m_refFeatPt.size();
		for(i=0; i<nsize; i++){
			if(m_refFeatPt[i].type >IDLE_FEATPOINT){
				fpVector.push_back(m_refFeatPt[i].pos);
			}
		}
	}else{
		nsize = pointsGood_.size();
		for(i=0; i<nsize; i++){
			if(params.bZoomHalf){
				cv::Point2f	tmpPt = cv::Point2f(pointsGood_[i].x*2, pointsGood_[i].y*2);
				fpVector.push_back(tmpPt);
			}else{
				fpVector.push_back(pointsGood_[i]);
			}
		}
	}
}

bool SceneOptFlowImpl::selectFeatPoints(const Mat& image,  bool init /*=false*/)
{
	std::vector<Point2f> corners;

	m_extractFeatPort.ExtractFP(image, corners, params.bScaleDetect, params.fpMode);

	if(params.bContinueFrms){
		refineFeatPoints(image, corners);
		storeFeatPoints();
		std::swap(m_refFeatPt, m_curFeatPt);
	}

	pointsPrev_.clear();
//	pointsPrev_.assign(corners.begin(), corners.end());
	std::swap(pointsPrev_, corners);

	return true;
}

bool SceneOptFlowImpl::storeFeatPoints()
{
	int i;
	for(i=0; i<params.nfeatures; i++){
		m_fpVector[i].first = i;
		if(m_fpVector[i].second.size()>= params.nframes){
			std::vector<FeatPoint_t>::iterator itPts = m_fpVector[i].second.begin();
			m_fpVector[i].second.erase(itPts);
		}
		m_curFeatPt[i].frmIdx = m_framIdx;
		m_fpVector[i].second.push_back(m_curFeatPt[i]);
	}
	return true;
}

void SceneOptFlowImpl::refineFeatPoints(const Mat& image, const vector<Point2f> corners)
{
	int i, j, ncornes, index;
	float x, y;
	int x_cell, y_cell;
	int x1, y1, x2, y2;
	int w = image.cols;
	int h = image.rows;
	float minDistance = params.minDistance;

	const int cell_size = cvRound(minDistance);
	const int grid_width = (w + cell_size - 1) / cell_size;
	const int grid_height = (h + cell_size - 1) / cell_size;

	std::vector<std::vector<Point2f> > grid(grid_width*grid_height);
	minDistance *= minDistance;

	for(i=0; i<params.nfeatures; i++){
		if(m_curFeatPt[i].type == TRACK_FEATPOINT){
			x = m_curFeatPt[i].pos.x;
			y = m_curFeatPt[i].pos.y;
			x_cell = (int)(x / cell_size);
			y_cell = (int)(y / cell_size);
			if(grid[y_cell*grid_width + x_cell].size() == 0){
				grid[y_cell*grid_width + x_cell].push_back(Point2f(x, y));
			}if(grid[y_cell*grid_width + x_cell].size() == 1){
				std::vector <Point2f> &m = grid[y_cell*grid_width + x_cell];
				float dx = x - m[0].x;
				float dy = y - m[0].y;
				float dist = dx*dx + dy*dy ;
				if( dist > minDistance/2.0 ){
					grid[y_cell*grid_width + x_cell].push_back(Point2f(x, y));
				}else if(dist<minDistance/4.0){
					m_curFeatPt[i].pos = Point2f(-1.f, -1.f);
					m_curFeatPt[i].type = IDLE_FEATPOINT;
				}
			}else{
				m_curFeatPt[i].pos = Point2f(-1.f, -1.f);
				m_curFeatPt[i].type = IDLE_FEATPOINT;
			}
		}
	}

	index = 0;
	ncornes = corners.size();

	for(i=0; i<ncornes; i++){
			bool good = true;
			x = corners[i].x;
			y = corners[i].y;
			x_cell = (int)(x / cell_size);
			y_cell = (int)(y / cell_size);
			x1 = x_cell - 1;			y1 = y_cell - 1;
			x2 = x_cell + 1;			y2 = y_cell + 1;

#if 0
			for(; index<params.nfeatures ; index++){
				if(m_curFeatPt[index].type< NEWSEL_FEATPOINT){
					break;
				}
			}
#else
			while(index<params.nfeatures && m_curFeatPt[index].type> IDLE_FEATPOINT)
				index++;
#endif

			if(index >= params.nfeatures) break;

			// boundary check
			x1 = std::max(0, x1);
			y1 = std::max(0, y1);
			x2 = std::min(grid_width-1, x2);
			y2 = std::min(grid_height-1, y2);

			for( int yy = y1; yy <= y2; yy++ )
			{
				for( int xx = x1; xx <= x2; xx++ )
				{
					std::vector <Point2f> &m = grid[yy*grid_width + xx];
					if( m.size() )
					{
						for(j = 0; j < m.size(); j++)
						{
							float dx = x - m[j].x;
							float dy = y - m[j].y;

							if( dx*dx + dy*dy < minDistance )
							{
								good = false;
								goto break_out;
							}
						}
					}
				}
			}
			break_out:
			if (good)
			{
				grid[y_cell*grid_width + x_cell].push_back(Point2f(x, y));
				m_curFeatPt[index].pos = Point2f(x, y);
				m_curFeatPt[index].type = NEWSEL_FEATPOINT;
				index++;
			}
	}
}

SceneOptFlowImpl::SceneOptFlowImpl( const SceneOptFlow::Params &parameters )
:termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.3)
{
    isInit = false;
    m_framIdx = 0;
    params =  parameters ;

    m_extractFeatPort.SetMinDistance(params.minDistance);

    m_refFeatPt.resize(params.nfeatures);
    m_curFeatPt.resize(params.nfeatures);
    m_fpVector.resize(params.nfeatures);

    sceneMatrix = Mat::eye(3, 3, CV_32F);
}

Ptr<SceneOptFlow> SceneOptFlow::createInterface(const SceneOptFlow::Params &parameters)
{
	if(parameters.fpMode == MATCH_FEATPOINT){
		return Ptr<SceneMVCalImpl>(new SceneMVCalImpl(parameters));
	}else if(parameters.fpMode == MAX_VAR_FEATHER){
		return Ptr<MaxVarFeatImpl>(new MaxVarFeatImpl(parameters));
	}else if(parameters.fpMode == MAX_VAR_REGION){
		return Ptr<MaxVarRegionImpl>(new MaxVarRegionImpl(parameters));
	}else{
		return Ptr<SceneOptFlowImpl>(new SceneOptFlowImpl(parameters));
	}
}

SceneOptFlow::Params::Params()//:ransacParams_(RansacParams::linearSimilarityMotionStd())
{
	fpMode = GOOD_FEATPOINT;//ORB_FEATPOINT;//MATCH_FEATPOINT;//

	nfeatures = 100;//max count
	nframes = 100;
	minDistance = 16.f;
	bContinueFrms = false;//true;
	bScaleDetect = true;
	bRejectOutliner = false;
	bFBTrk = false;//true;
	bZoomHalf = false;
	pointsInGrid = 6;

	motionModel_ = LINEAR_SIMILARITY;
	if(motionModel_ == TRANSLATION){
		ransacParams_ = (RansacParams::translationMotionStd());
	}else if(motionModel_ == TRANSLATION_AND_SCALE){
		ransacParams_ = (RansacParams::translationAndScale2dMotionStd());
	}else if(motionModel_ == LINEAR_SIMILARITY){
		ransacParams_ = (RansacParams::linearSimilarityMotionStd());
	}else if(motionModel_ == AFFINE){
		ransacParams_ = (RansacParams::affine2dMotionStd());
	}
	printf("%s:size=%d, thresh=%f \n",__func__, ransacParams_.size, ransacParams_.thresh);
	maxRmse_ = 0.4;//0.5;
	minInlierRatio_ = 0.1;
}

}
