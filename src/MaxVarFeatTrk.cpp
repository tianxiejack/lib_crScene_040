#include "MaxVarFeatTrk.hpp"
#include "GlobalMotion.h"
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>             /* malloc/free declarations */
#include <string.h>             /* memset declaration */

namespace OptFlowTrk{

MaxVarFeatImpl::MaxVarFeatImpl( const SceneOptFlow::Params &parameters )
:termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.3)
{
    isInit = false;
    params =  parameters;
    imageSize = cv::Size(0, 0);
    grid.clear();
    bestgrid.clear();
    var = 10*10;
    sceneMatrix = cv::Mat::eye(3, 3, CV_32F);
}

bool MaxVarFeatImpl::initImpl( const Mat& image, const Rect2d& validBox )
{
	bool iRtn = true;

	if(image.size() != imageSize){
		buildGrid(image);
	}
	extractMaxVarBlk(image);

	setRefImage(image);

	return iRtn;
}

bool MaxVarFeatImpl::updateImpl( const Mat& image, Point2f& mvPos )
{
	std::vector<Point2f> pointsToTrackOld,pointsToTrackNew;
	Mat oldImage_gray,newImage_gray;
	bool iRtn = true;

	oldImage_gray = image_ref;
	newImage_gray = image;

	CV_Assert(oldImage_gray.channels() == 1);
	CV_Assert(newImage_gray.channels() == 1);

	 sceneMatrix = cv::Mat::eye(3, 3, CV_32F);

	status.clear();
	errors.clear();

	pointsToTrackOld.clear();
	pointsToTrackNew.clear();

	int64 tstart;
	tstart = getTickCount();

	betweenframeTrk(oldImage_gray, newImage_gray, pointsToTrackOld, pointsToTrackNew);

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

	static MotionModel motionModelBK = AFFINE;
	bool bJudge = judgeFPDistrib(image, pointsGood_);
	if(bJudge){
		params.motionModel_ = AFFINE;
		params.ransacParams_ = (RansacParams::affine2dMotionStd());

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
#if 1
	Mat M = estimateGlobalMotionRobust(pointsPrevGood_, pointsGood_, params.motionModel_, params.ransacParams_, &rmse, &ninliers);
#else
	cv::Size sz = image.size();
	Mat M = estimateGlobalMotionGMS(pointsPrevGood_, pointsGood_, sz, params.motionModel_, params.ransacParams_, &rmse, &ninliers);
#endif

	if (rmse > params.maxRmse_ || static_cast<float>(ninliers) / pointsGood_.size() < params.minInlierRatio_){
		printf("rmse=%f, ninliers=%d, pointsGood_size=%d \n",rmse, ninliers, pointsGood_.size());
		M = Mat::eye(3, 3, CV_32F);
		iRtn = false;
	}

	mvPos.x = M.at<float>(0,2);
	mvPos.y = M.at<float>(1,2);

	sceneMatrix = M.clone();

	tstart = getTickCount();
	extractMaxVarBlk(image);
//	printf("extractMaxVarBlk: time = %f sec \n\n", ( (getTickCount() - tstart)/getTickFrequency()) );

	setRefImage(image);

	return iRtn;
}

void MaxVarFeatImpl::betweenframeTrk(const cv::Mat oldImage_gray,const cv::Mat newImage_gray,
		std::vector<Point2f> &pointsToTrackOld, std::vector<Point2f> &pointsToTrackNew )
{
	int i, nsize = pointsPrev_.size();
	pointsToTrackOld.resize(nsize);
	pointsToTrackNew.resize(nsize);

	for(i=0; i<nsize; i++){
		pointsToTrackOld[i] = pointsPrev_[i];
	}

	if(nsize > 0){
		calcOpticalFlowPyrLK(oldImage_gray, newImage_gray,pointsToTrackOld,pointsToTrackNew,status,errors,Size(3,3),3,termcrit,0);
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

bool MaxVarFeatImpl::judgeFPDistrib(const Mat &image, const std::vector<Point2f> &ptsvector)
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

void MaxVarFeatImpl::getFeatPointsImpl(std::vector<Point2f> &fpVector)
{
	int i, nsize;
	fpVector.clear();

	nsize = pointsGood_.size();
	for(i=0; i<nsize; i++){
		fpVector.push_back(pointsGood_[i]);
	}
}

void MaxVarFeatImpl::buildGrid(const cv::Mat &image)
{
	cv::Size	boxsize;
	int width = image.cols;
	int height = image.rows;
	imageSize = image.size();
	if(width<=768){ //720x576
		boxsize.width = 72;
		boxsize.height = 60;
	}else if(width<=1024){//1024x768
		boxsize.width = 100;
		boxsize.height = 72;
	}else if(width<=1280){//1280x1024
		boxsize.width = 120;
		boxsize.height = 90;
	}else{//1920x1080
		boxsize.width = 156;
		boxsize.height = 96;
	}
	BoundingBox bbox;
	BoundingBox	centBox;
	centBox.x = width/6;
	centBox.y = height/2-boxsize.height*0.5;
	centBox.width = width*2/3;
	centBox.height = boxsize.height;
	grid.clear();
	for (int y=height/6;y<image.rows-height/6;y+=boxsize.height/3){
		for (int x=width/6;x<image.cols-width/6;x+=boxsize.width/3){
			if((y+boxsize.height)>height ||(x+boxsize.width)>width)
				continue;
			bbox.x = x;
			bbox.y = y;
			bbox.width = boxsize.width;
			bbox.height = boxsize.height;
			bbox.var = 0.0;
			if(bbOverlap(centBox, bbox)>0.0)
				continue;
			grid.push_back(bbox);
		}
	}
}

float MaxVarFeatImpl::bbOverlap(const BoundingBox& box1,const BoundingBox& box2)
{
	if (box1.x > box2.x+box2.width) { return 0.0; }
	if (box1.y > box2.y+box2.height) { return 0.0; }
	if (box1.x+box1.width < box2.x) { return 0.0; }
	if (box1.y+box1.height < box2.y) { return 0.0; }

	float colInt =  min(box1.x+box1.width,box2.x+box2.width) - max(box1.x, box2.x);
	float rowInt =  min(box1.y+box1.height,box2.y+box2.height) - max(box1.y,box2.y);

	float intersection = colInt * rowInt;
	float area1 = box1.width*box1.height;
	float area2 = box2.width*box2.height;
	return intersection / (area1 + area2 - intersection);
}

double MaxVarFeatImpl::getVar(const BoundingBox& box,const Mat& sum,const Mat& sqsum)
{
	double brs = sum.at<int>(box.y+box.height,box.x+box.width);
	double bls = sum.at<int>(box.y+box.height,box.x);
	double trs = sum.at<int>(box.y,box.x+box.width);
	double tls = sum.at<int>(box.y,box.x);
	double brsq = sqsum.at<double>(box.y+box.height,box.x+box.width);
	double blsq = sqsum.at<double>(box.y+box.height,box.x);
	double trsq = sqsum.at<double>(box.y,box.x+box.width);
	double tlsq = sqsum.at<double>(box.y,box.x);
	double mean = (brs+tls-trs-bls)/((double)box.area());
	double sqmean = (brsq+tlsq-trsq-blsq)/((double)box.area());
	return sqmean-mean*mean;
}

static float _calDistance(BoundingBox rc1, BoundingBox rc2)
{
	float dist;
	Point2f pt1, pt2;
	pt1.x = rc1.x + rc1.width/2;		pt1.y = rc1.y + rc1.height/2;
	pt2.x = rc2.x + rc2.width/2;		pt2.y = rc2.y + rc2.height/2;

	dist = sqrt( (pt1.x-pt2.x)*(pt1.x-pt2.x) + (pt1.y-pt2.y)*(pt1.y-pt2.y));

	return dist;
}

static bool bNearestBox(const std::vector<BoundingBox> &tmpBox, BoundingBox box)
{
	int i, nsize = tmpBox.size();
	float dist, length;
	bool iRtn = true;
	if(nsize == 0)
		return iRtn;
	length = sqrt((float)box.width*box.width+box.height*box.height);
	for(i=0; i<nsize; i++){
		dist = _calDistance(box, tmpBox[i]);
		if(dist < (length*1.2)){
			iRtn = false;
			break;
		}
	}
	return iRtn;
}

void MaxVarFeatImpl::extractMaxVarBlk(const cv::Mat &image)
{
	int i, j, k, count;
	double val;
	assert(image.channels() == 1);

	integral(image,iisum,iisqsum);
	bestgrid.clear();
	for (i=0;i<grid.size();i++){//FIXME: BottleNeck
		val = getVar(grid[i],iisum,iisqsum);
		grid[i].var = val;
		if (val>=var){
		  bestgrid.push_back(grid[i]);
		}
	}
	if(bestgrid.size() > 4){
		std::sort(bestgrid.begin(), bestgrid.end(), BComparator(bestgrid));//Descending order
	}

	pointsPrev_.clear();
	if(bestgrid.size() == 0)
		return;
	int itern = std::min((int)bestgrid.size(), 4);

	std::vector<BoundingBox>	tmpBox;
	tmpBox.clear();
	for(k=0,count=0;k<bestgrid.size();k++){
		BoundingBox oldBox;
		oldBox = bestgrid[k];
		if(bNearestBox(tmpBox, oldBox))
		{
			 for(i=0;i<params.pointsInGrid;i++)
			 {
					for(j=0;j<params.pointsInGrid;j++)
					{
						pointsPrev_.push_back(
									Point2f((float)(oldBox.x+((1.0*oldBox.width)/params.pointsInGrid)*j+.5*oldBox.width/params.pointsInGrid),
									(float)(oldBox.y+((1.0*oldBox.height)/params.pointsInGrid)*i+.5*oldBox.height/params.pointsInGrid)));
					}
			 }
			 tmpBox.push_back(oldBox);
			 count++;
		}
		if(count >= itern)
			break;
	}

}

}
