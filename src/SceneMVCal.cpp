#include "SceneMVCal.hpp"

namespace OptFlowTrk{

SceneMVCalImpl::SceneMVCalImpl( const SceneOptFlow::Params &parameters )
{
    isInit = false;
    m_frameIdx = 0;
    params =  parameters;
    sceneMatrix = Mat::eye(3, 3, CV_32F);
}

void SceneMVCalImpl::getFeatPointsImpl(std::vector<Point2f> &fpVector)
{
	fpVector.clear();
}

bool SceneMVCalImpl::initImpl( const Mat& image, const Rect2d& validBox )
{
	setRefImage(image);
	m_frameIdx = 0;
	getFeatureStruct(image, featImg_ref);

	return true;
}

bool SceneMVCalImpl::updateImpl( const Mat& image, Point2f& mvPos )
{
	cv::Rect	imageRect;
	cv::Mat result,featImg_cur, featMap;
	double minVal, maxVal;
	cv::Point minLoc,maxLoc;
	int64 tstart;

	sceneMatrix = cv::Mat::eye(3,3, CV_32F);

	imageRect.x = imageRect.y = 64;
	imageRect.width = image.cols - imageRect.x*2;
	imageRect.height = image.rows - imageRect.y*2;

	getFeatureStruct(image, featMap);
	featMap(imageRect).copyTo(featImg_cur);

	tstart = getTickCount();

	cv::matchTemplate(featImg_cur, featImg_ref, result, CV_TM_CCOEFF_NORMED);

	cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
	if(maxLoc.x == 0 && maxLoc.y == 0){
		mvPos = cv::Point2f(0.f, 0.f);
	}else{
		mvPos = cv::Point2f((float)(maxLoc.x-imageRect.x), (float)(maxLoc.y-imageRect.y));
	}
	sceneMatrix.at<float>(0,2) = mvPos.x;
	sceneMatrix.at<float>(1,2) = mvPos.y;

	featMap.copyTo(featImg_ref);

//	printf("%s: time = %f sec \n\n",__func__,  ( (getTickCount() - tstart)/getTickFrequency()) );

//	cvtColor(featMap, featMap, CV_GRAY2BGR);
//	imshow("featMap", featMap);
//	waitKey(1);
	return true;
}

static void _IMG_sobel(  cv::Mat src, cv::Mat &dst)
{
	unsigned char *in;       /* Input image data  */
	unsigned char  *out;      /* Output image data */
	short cols,rows;         /* Image dimensions  */
	int H;    /* Horizontal mask result            */
	int V;    /* Vertical mask result                   */
	int O;    /* Sum of horizontal and vertical masks   */
	int i;     /* Input pixel offset						 */
	int o;    /* Output pixel offset.                 */
	int xy;   /* Loop counter.                          */

	int i00, i01, i02;
	int i10,      i12;
	int i20, i21, i22;

	cols = src.cols;
	rows = src.rows;
	in = src.data;
	out = dst.data;

	assert(src.rows == dst.rows && src.cols == dst.cols);

#pragma UNROLL(4)
	for (xy = 0, i = cols + 1, o = 1;    xy < cols*(rows-2) - 2;    xy++, i++, o++)
	{
		i00=in[i-cols-1]; i01=in[i-cols]; i02=in[i-cols+1];
		i10=in[i     -1];                 i12=in[i     +1];
		i20=in[i+cols-1]; i21=in[i+cols]; i22=in[i+cols+1];

		H = -i00 - 2*i01 -   i02 +   i20 + 2*i21 + i22;
		V = -i00 +   i02 - 2*i10 + 2*i12 -   i20 + i22;
		O = std::abs(H) + std::abs(V);

		if (O > 255) O = 255;
		out[o] = O;

	}
}

void SceneMVCalImpl::getFeatureStruct(const cv::Mat image, cv::Mat &featMap)
{
	float kernel[3] = {-1.f, 0.f, 1.f};
	cv::Mat kernel_dx = cv::Mat(1, 3, CV_32F, kernel);
	cv::Mat kernel_dy = cv::Mat(3, 1, CV_32F, kernel);
	cv::Mat dx, dy;
	int method = 2;

	if(method == 0){
		cv::filter2D(image, dx, -1, kernel_dx, cv::Point(-1, 0));
		cv::filter2D(image, dy, -1, kernel_dy, cv::Point(0, -1));
		featMap = cv::abs(dx)+cv::abs(dy);
	}else if(method == 1){
//	cv::Sobel(image, featMap, -1, 1, 1);
	cv::Mat imageBlur;
	cv::blur(image, imageBlur, cv::Size(5,5));
	featMap = cv::Mat(image.size(), image.type());
	_IMG_sobel(imageBlur, featMap);
	}else if(method == 2){
		cv::blur(image, featMap, cv::Size(5,5));
	}

}

}
