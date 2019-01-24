#ifndef _MEDIAN_FLOW_TRK_H_
#define _MEDIAN_FLOW_TRK_H_

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

using namespace cv;
using namespace std;

namespace OptFlowTrk{

typedef Rect_<double> Rect2d;

class MedianFlowTracker{

public:
	struct Params
	{
	    Params();
	    int pointsInGrid;
	};

	virtual ~MedianFlowTracker();

	bool init( const Mat& image, const Rect2d& boundingBox );

	void uninit();

	bool update( const Mat& image, Rect2d& boundingBox );

	static Ptr<MedianFlowTracker> CreateMedianFlowTrk(const MedianFlowTracker::Params &parameters = MedianFlowTracker::Params() );

protected:

	virtual bool initImpl( const Mat& image, const Rect2d& boundingBox ) = 0;

	virtual void uninitImpl() = 0;

	virtual bool updateImpl( const Mat& image, Rect2d& boundingBox ) = 0;

	bool isInit;

private:

//	class	MedianFlowTrackerImpl;
//	MedianFlowTrackerImpl	*_impl;
};


}


#endif
