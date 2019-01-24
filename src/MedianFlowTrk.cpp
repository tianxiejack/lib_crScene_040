#include "MedianFlowTrk.hpp"
#include <stdio.h>
#include <omp.h>

#undef ALEX_DEBUG
#ifdef ALEX_DEBUG
#define dfprintf(x) fprintf x
#define dprintf(x) printf x
#else
#define dfprintf(x)
#define dprintf(x)
#endif

namespace OptFlowTrk
{

MedianFlowTracker::~MedianFlowTracker()
{
}

bool MedianFlowTracker::init( const Mat& image, const Rect2d& boundingBox )
{
	if( isInit ){
//		return false;
		uninit();
	}

	if( image.empty() )
		return false;

	bool initTracker = initImpl(image, boundingBox);

	if( initTracker )
		isInit = true;

	return initTracker;
}

void MedianFlowTracker::uninit()
{
	isInit =false;
	uninitImpl();
}

bool MedianFlowTracker::update( const Mat& image, Rect2d& boundingBox )
{
	if( !isInit )
	    return false;

	if( image.empty() )
	    return false;

	return updateImpl(image, boundingBox);
}

class	MedianFlowTrackerImpl:public MedianFlowTracker{

public:

	MedianFlowTrackerImpl(const MedianFlowTracker::Params &parameters = MedianFlowTracker::Params() );

protected:

	bool initImpl( const Mat& image, const Rect2d& boundingBox );

	void uninitImpl(){};

	bool updateImpl( const Mat& image, Rect2d& boundingBox );

	void thredSeg(const Mat &image, Rect2d& boundingBox);

	void drawHist(const cv::String  name, const int hist[], std::vector<int> peakVec);

	MedianFlowTracker::Params params;

private:

	Mat image_;

	Rect2d boundingBox_;

	bool medianFlowImpl(Mat oldImage,Mat newImage,Rect2d& oldBox);

	Rect2d vote(const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,const Rect2d& oldRect,Point2f& mD);

	 template<typename T>
	 T getMedian( std::vector<T>& values,int size=-1);

	 float dist(Point2f p1,Point2f p2);

	 void computeStatistics(std::vector<float>& data,int size=-1);

	 void check_FB(const Mat& oldImage,const Mat& newImage,
			 const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,std::vector<bool>& status);

	 void check_FB_CR(const Mat& oldImage,const Mat& newImage,
	 			 const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,std::vector<bool>& status);

	 void check_NCC(const Mat& oldImage,const Mat& newImage,
			 const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,std::vector<bool>& status);

	 inline double l2distance(Point2f p1,Point2f p2);

	 TermCriteria termcrit;
};

MedianFlowTrackerImpl::MedianFlowTrackerImpl(const MedianFlowTracker::Params &paramsIn ):
		termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.3)
{
	params=paramsIn;
	isInit=false;
}

bool MedianFlowTrackerImpl::initImpl( const Mat& image, const Rect2d& boundingBox )
{
	image.copyTo(image_);
	boundingBox_ = boundingBox;

	return true;
}

void MedianFlowTrackerImpl::drawHist(const cv::String name, const int hist[], std::vector<int> peakVec)
{
	cv::Mat histMat = cv::Mat(260,260,CV_8UC3);
	histMat.setTo(Scalar::all(255));
	int i, maxVal;
	cv::Rect roi;
	cv::Point pt0, pt1;
	maxVal = 0;
	for(i=0; i<256; i++){
		if(hist[i]>maxVal){
			maxVal = hist[i];
		}
	}
	if(maxVal != 0){
		roi.x = 0; roi.y = 2;
		roi.width = histMat.cols;
		roi.height = histMat.rows-4;
		cv::rectangle( histMat, Point( roi.x, roi.y ), Point( roi.x+roi.width, roi.y+roi.height), cvScalar(0,255,0), 2, 8 );
		for(i=0; i<256; i++){
			pt0 = cv::Point(2+i,258);
			pt1 = cv::Point(2+i,(int)(258-hist[i]*255/maxVal));
			line(histMat, pt0, pt1,  cvScalar(0,0,255), 1, 8);
		}
		if(peakVec.size() >0){
			for(i=0; i<peakVec.size(); i++){
				int idx = peakVec[i];
				pt0 = cv::Point(2+idx,258);
				pt1 = cv::Point(2+idx,(int)(258-hist[idx]*255/maxVal));
				line(histMat, pt0, pt1,  cvScalar(255,0,0), 2, 8);
			}
		}
	}

	imshow(name, histMat);
	waitKey(1);
}

static bool IsDimodal(double* HistGram)       // 检测直方图是否为双峰的
{
	// 对直方图的峰进行计数，只有峰数位2才为双峰
	int Count = 0;
	for (int Y = 1; Y < 255; Y++)
	{
		if (HistGram[Y - 1] < HistGram[Y] && HistGram[Y + 1] < HistGram[Y])
		{
			Count++;
			if (Count > 2) return false;
		}
	}
	if (Count == 2)
		return true;
	else
		return false;
}

static int GetIntermodesThreshold(int* HistGram)
{
	int Y, Iter = 0, Index;
	double HistGramC[256];           // 基于精度问题，一定要用浮点数来处理，否则得不到正确的结果
	double HistGramCC[256];          // 求均值的过程会破坏前面的数据，因此需要两份数据
	for (Y = 0; Y < 256; Y++)
	{
		HistGramC[Y] = HistGram[Y];
		HistGramCC[Y] = HistGram[Y];
	}
	// 通过三点求均值来平滑直方图
	while (IsDimodal(HistGramCC) == false)                                                  // 判断是否已经是双峰的图像了
	{
		HistGramCC[0] = (HistGramC[0] + HistGramC[0] + HistGramC[1]) / 3;                   // 第一点
		for (Y = 1; Y < 255; Y++)
			HistGramCC[Y] = (HistGramC[Y - 1] + HistGramC[Y] + HistGramC[Y + 1]) / 3;       // 中间的点
		HistGramCC[255] = (HistGramC[254] + HistGramC[255] + HistGramC[255]) / 3;           // 最后一点
		memcpy(HistGramCC, HistGramC, 256 * sizeof(double));         // 备份数据，为下一次迭代做准备
		Iter++;
		if (Iter >= 10000) return -1;                                                       // 似乎直方图无法平滑为双峰的，返回错误代码
	}
// 阈值为两峰值的平均值
	int Peak[2];
	for (Y = 1, Index = 0; Y < 255; Y++)
		if (HistGramCC[Y - 1] < HistGramCC[Y] && HistGramCC[Y + 1] < HistGramCC[Y]) Peak[Index++] = Y - 1;
	return ((Peak[0] + Peak[1]) / 2);
}

static bool IsDimodalCR(double* HistGram, std::vector<int> &peakVec)       // 检测直方图是否为双峰的
{
	// 对直方图的峰进行计数，只有峰数位2才为双峰
	int Count = 0;
	peakVec.clear();
	for (int Y = 1; Y < 255; Y++)
	{
		if (HistGram[Y - 1] < HistGram[Y] && HistGram[Y + 1] < HistGram[Y])
		{
			Count++;
			peakVec.push_back(Y);
			if (Count > 2) return false;
		}
	}
	if (Count == 2)
		return true;
	else
		return false;
}
void MedianFlowTrackerImpl::thredSeg(const Mat &image, Rect2d& boundingBox)
{
	cv::Rect roi;
	cv::Point2f centPt;
	float scale = 1.2;
	centPt.x = (boundingBox.x + boundingBox.width/2);
	centPt.y = (boundingBox.y + boundingBox.height/2);
	roi.width = (int)(boundingBox.width*scale);
	roi.height = (int)(boundingBox.height*scale);
	roi.x = (int)(centPt.x-roi.width/2);
	roi.y = (int)(centPt.y-roi.height/2);
	roi.x = (roi.x<0)?0:roi.x;
	roi.y = (roi.y<0)?0:roi.y;
	roi.width = ((roi.width+roi.x)>image.cols)?(image.cols-1-roi.width):roi.width;
	roi.height = ((roi.height+roi.y)>image.rows)?(image.rows-1-roi.height):roi.height;
	cv::Mat roiMap, gaussKel;
	uchar *pdata;
	double sigma = 1.0;
	int ksize = 6;
	image(roi).copyTo(roiMap);
	int i, j, val, hist[256]={0,};
	int filterHist[256]={0,};
	double filterPeak[256] = {0.0, };
	int maxVal0, maxIdx0, maxVal1, maxIdx1, minVal, minIdx;
	float sum = 0.f;
	pdata = roiMap.data;
	for(i=0; i<roiMap.cols*roiMap.rows; i++){
		hist[*pdata++]++;
	}

	bool bPeak = false;
	std::vector<int> peakVec;
	for(int itern=0; itern<20; itern++){
		bPeak = false;
		memset(filterHist, 0, 256*sizeof(int));
		memset(filterPeak, 0, 256*sizeof(double));
//		sigma = (0.3*((ksize-1)*0.5)+0.8);
		ksize = (int)(sigma*6+1);
		gaussKel = getGaussianKernel(ksize, sigma, CV_32F);
		for(i=0; i<256; i++){
			sum = 0.f;
			if(i<ksize/2){
				for(j=0; j<ksize; j++){
					val = (i+j-ksize/2)<0?(hist[ksize/2-j-i]):(hist[i+j-ksize/2]);
					sum += val*gaussKel.at<float>(j,0);
				}
			}else if(i>(255-ksize/2)){
				for(j=0; j<ksize; j++){
					val = (i+j-ksize/2)>255?(hist[510+ksize/2-i-j]):(hist[i+j-ksize/2]);
					sum += val*gaussKel.at<float>(j,0);
				}
			}else{
				for(j=0; j<ksize; j++){
					val = (hist[i+j-ksize/2]);
					sum += val*gaussKel.at<float>(j,0);
				}
			}
			filterHist[i] = (int)sum;
			filterPeak[i] = sum;
		}
		if (IsDimodalCR(filterPeak, peakVec)){
			bPeak = true;
			break;
		}

		sigma += 1.0;
	}
	std::vector<int> peakNull;
	peakNull.clear();
	drawHist("origHist", hist, peakNull);
	if(bPeak)
		drawHist("filterHist", filterHist, peakVec);

	if(bPeak){
		cv::Mat roiDisp;
		int adaptiveThred = (peakVec[0]+peakVec[1])/2;
		threshold(roiMap, roiDisp, adaptiveThred, 255, CV_THRESH_BINARY);
		imshow("Disp", roiDisp);
		waitKey(1);
	}
}

bool MedianFlowTrackerImpl::updateImpl( const Mat& image, Rect2d& boundingBox )
{
	Mat oldImage=image_;

	Rect2d oldBox=boundingBox_;

//	thredSeg(image, oldBox);

	if(!medianFlowImpl(oldImage,image,oldBox)){
		return false;
	}
	boundingBox=oldBox;

	image.copyTo(image_);
	boundingBox_ = boundingBox;

	return true;
}

bool MedianFlowTrackerImpl::medianFlowImpl(Mat oldImage,Mat newImage,Rect2d& oldBox)
{
    std::vector<Point2f> pointsToTrackOld,pointsToTrackNew;

    Mat oldImage_gray,newImage_gray;

    CV_Assert(oldImage.channels() == 1);
    CV_Assert(newImage.channels() == 1);
    oldImage_gray = oldImage ;
    newImage_gray = newImage;

    int64 tstart;
    tstart = getTickCount();

    //"open ended" grid
    for(int i=0;i<params.pointsInGrid;i++){
        for(int j=0;j<params.pointsInGrid;j++){
                pointsToTrackOld.push_back(
                        Point2f((float)(oldBox.x+((1.0*oldBox.width)/params.pointsInGrid)*j+.5*oldBox.width/params.pointsInGrid),
                        (float)(oldBox.y+((1.0*oldBox.height)/params.pointsInGrid)*i+.5*oldBox.height/params.pointsInGrid)));
        }
    }

    std::vector<uchar> status(pointsToTrackOld.size());
    std::vector<float> errors(pointsToTrackOld.size());
    calcOpticalFlowPyrLK(oldImage_gray, newImage_gray,pointsToTrackOld,pointsToTrackNew,status,errors,Size(3,3),5,termcrit,0);
    dprintf(("\t%d after LK forward\n",(int)pointsToTrackOld.size()));

    std::vector<Point2f> di;
    for(int i=0;i<(int)pointsToTrackOld.size();i++){
        if(status[i]==1){
            di.push_back(pointsToTrackNew[i]-pointsToTrackOld[i]);
        }
    }

    std::vector<bool> filter_status;
    filter_status.clear();
    check_FB_CR(oldImage_gray,newImage_gray,pointsToTrackOld,pointsToTrackNew,filter_status);
    check_NCC(oldImage_gray,newImage_gray,pointsToTrackOld,pointsToTrackNew,filter_status);

    // filter
    for(int i=0;i<(int)pointsToTrackOld.size();i++){
        if(!filter_status[i]){
            pointsToTrackOld.erase(pointsToTrackOld.begin()+i);
            pointsToTrackNew.erase(pointsToTrackNew.begin()+i);
            filter_status.erase(filter_status.begin()+i);
            i--;
        }
    }
    dprintf(("\t%d after LK backward\n",(int)pointsToTrackOld.size()));

//    printf("medianFlowImpl: time = %f sec \n\n", ( (getTickCount() - tstart)/getTickFrequency()) );

    if(pointsToTrackOld.size()==0 || di.size()==0){
        return false;
    }
    Point2f mDisplacement;
    oldBox=vote(pointsToTrackOld,pointsToTrackNew,oldBox,mDisplacement);

    std::vector<double> displacements;
    for(int i=0;i<(int)di.size();i++){
        di[i]-=mDisplacement;
        displacements.push_back(sqrt(di[i].ddot(di[i])));
    }
    if(getMedian(displacements,(int)displacements.size())>20){
        return false;
    }

    return true;
}

Rect2d MedianFlowTrackerImpl::vote(const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,const Rect2d& oldRect,Point2f& mD)
{
    static int iteration=0;//FIXME -- we don't want this static var in final release
    Rect2d newRect;
    Point2d newCenter(oldRect.x+oldRect.width/2.0,oldRect.y+oldRect.height/2.0);
    int n=(int)oldPoints.size();
    std::vector<double> buf(std::max(n*(n-1)/2,3),0.0);

   if(oldPoints.size()==1){
        newRect.x=oldRect.x+newPoints[0].x-oldPoints[0].x;
        newRect.y=oldRect.y+newPoints[0].y-oldPoints[0].y;
        newRect.width=oldRect.width;
        newRect.height=oldRect.height;
        return newRect;
    }

    double xshift=0,yshift=0;
    for(int i=0;i<n;i++){  buf[i]=newPoints[i].x-oldPoints[i].x;  }
    xshift=getMedian(buf,n);
    newCenter.x+=xshift;
    for(int i=0;i<n;i++){  buf[i]=newPoints[i].y-oldPoints[i].y;  }
    yshift=getMedian(buf,n);
    newCenter.y+=yshift;
    mD=Point2f((float)xshift,(float)yshift);

    if(oldPoints.size()==1){
        newRect.x=newCenter.x-oldRect.width/2.0;
        newRect.y=newCenter.y-oldRect.height/2.0;
        newRect.width=oldRect.width;
        newRect.height=oldRect.height;
        return newRect;
    }

    double nd,od;
    for(int i=0,ctr=0;i<n;i++){
        for(int j=0;j<i;j++){
            nd=l2distance(newPoints[i],newPoints[j]);
            od=l2distance(oldPoints[i],oldPoints[j]);
            buf[ctr]=(od==0.0)?0.0:(nd/od);
            ctr++;
        }
    }

    double scale=getMedian(buf,n*(n-1)/2);
    scale = 1.0;
    dprintf(("iter %d %f %f %f\n",iteration,xshift,yshift,scale));
    newRect.x=newCenter.x-scale*oldRect.width/2.0;
    newRect.y=newCenter.y-scale*oldRect.height/2.0;
    newRect.width=scale*oldRect.width;
    newRect.height=scale*oldRect.height;
    /*if(newRect.x<=0){
        exit(0);
    }*/
    dprintf(("rect old [%f %f %f %f]\n",oldRect.x,oldRect.y,oldRect.width,oldRect.height));
    dprintf(("rect [%f %f %f %f]\n",newRect.x,newRect.y,newRect.width,newRect.height));

    iteration++;
    return newRect;
}

template<typename T>
T MedianFlowTrackerImpl::getMedian(std::vector<T>& values,int size)
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

void MedianFlowTrackerImpl::computeStatistics(std::vector<float>& data,int size){
    int binnum=10;
    if(size==-1){
        size=(int)data.size();
    }
    float mini=*std::min_element(data.begin(),data.begin()+size),maxi=*std::max_element(data.begin(),data.begin()+size);
    std::vector<int> bins(binnum,(int)0);
    for(int i=0;i<size;i++){
        bins[std::min((int)(binnum*(data[i]-mini)/(maxi-mini)),binnum-1)]++;
    }
    for(int i=0;i<binnum;i++){
        dprintf(("[%4f,%4f] -- %4d\n",mini+(maxi-mini)/binnum*i,mini+(maxi-mini)/binnum*(i+1),bins[i]));
    }
}

double MedianFlowTrackerImpl::l2distance(Point2f p1,Point2f p2)
{
    double dx=p1.x-p2.x, dy=p1.y-p2.y;
    return sqrt(dx*dx+dy*dy);
}

void MedianFlowTrackerImpl::check_FB(const Mat& oldImage,const Mat& newImage,
        const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,std::vector<bool>& status)
{
    if(status.size()==0){
        status=std::vector<bool>(oldPoints.size(),true);
    }

    std::vector<uchar> LKstatus(oldPoints.size());
    std::vector<float> errors(oldPoints.size());
    std::vector<double> FBerror(oldPoints.size());
    std::vector<Point2f> pointsToTrackReprojection;
    calcOpticalFlowPyrLK(newImage, oldImage,newPoints,pointsToTrackReprojection,LKstatus,errors,Size(3,3),5,termcrit,0);

    for(int i=0;i<(int)oldPoints.size();i++){
        FBerror[i]=l2distance(oldPoints[i],pointsToTrackReprojection[i]);
    }
    if(FBerror.size() == 0)
        return;
    double FBerrorMedian=getMedian(FBerror);
    dprintf(("point median=%f\n",FBerrorMedian));
    dprintf(("FBerrorMedian=%f\n",FBerrorMedian));
    for(int i=0;i<(int)oldPoints.size();i++){
        status[i]=(FBerror[i]<FBerrorMedian);
    }
}

void MedianFlowTrackerImpl::check_FB_CR(const Mat& oldImage,const Mat& newImage,
        const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,std::vector<bool>& status)
{
    if(status.size()==0){
        status=std::vector<bool>(oldPoints.size(),false);
    }

    std::vector<uchar> LKstatus(oldPoints.size());
    std::vector<float> errors(oldPoints.size());
    std::vector<double> FBerror;
    std::vector<Point2f> pointsToTrackReprojection;
    calcOpticalFlowPyrLK(newImage, oldImage,newPoints,pointsToTrackReprojection,LKstatus,errors,Size(3,3),5,termcrit,0);

    std::vector<int> featIdxVect;
    FBerror.clear();
    for(int i=0;i<(int)oldPoints.size();i++){
    	if(LKstatus[i]){
    		FBerror.push_back( l2distance(oldPoints[i],pointsToTrackReprojection[i]) );
    		featIdxVect.push_back(i);
    	}
    }
    if(FBerror.size() == 0)
    	return;
    double FBerrorMedian=getMedian(FBerror);
    dprintf(("point median=%f\n",FBerrorMedian));
    dprintf(("FBerrorMedian=%f\n",FBerrorMedian));

    for(int i=0;i<(int)FBerror.size();i++){
    	int idx = featIdxVect[i];
    	status[idx]=(FBerror[i]<FBerrorMedian);
    }
}

void MedianFlowTrackerImpl::check_NCC(const Mat& oldImage,const Mat& newImage,
        const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,std::vector<bool>& status)
{
    std::vector<float> NCC(oldPoints.size(),0.0);
    Size patch(30,30);
    Mat p1,p2;

	for (int i = 0; i < (int)oldPoints.size(); i++) {
		getRectSubPix( oldImage, patch, oldPoints[i],p1);
		getRectSubPix( newImage, patch, newPoints[i],p2);

        const int N=900;
        double s1=sum(p1)(0),s2=sum(p2)(0);
        double n1=norm(p1),n2=norm(p2);
        double prod=p1.dot(p2);
        double sq1=sqrt(n1*n1-s1*s1/N),sq2=sqrt(n2*n2-s2*s2/N);
        double ares=(sq2==0)?sq1/abs(sq1):(prod-s1*s2/N)/sq1/sq2;

		NCC[i] = (float)ares;
	}
	if(NCC.size() == 0){
		return;
	}
	float median = getMedian(NCC);
	for(int i = 0; i < (int)oldPoints.size(); i++) {
        status[i] = (status[i] && (NCC[i]>median));
	}
}

Ptr<MedianFlowTracker> MedianFlowTracker::CreateMedianFlowTrk(const MedianFlowTracker::Params &parameters)
{
      return Ptr<MedianFlowTrackerImpl>(new MedianFlowTrackerImpl(parameters));
}

MedianFlowTracker::Params::Params()
{
	pointsInGrid=10;
}

}
