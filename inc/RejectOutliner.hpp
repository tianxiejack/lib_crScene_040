#ifndef 	_REJECT_OUT_LINER_H_
#define _REJECT_OUT_LINER_H_


#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include "SceneOptFlow.hpp"

using namespace cv;
using namespace std;

namespace OptFlowTrk{

class TranslationOutlierRejector{

public:
    TranslationOutlierRejector();
    virtual ~TranslationOutlierRejector();

    void setCellSize(Size val) { cellSize_ = val; }
    Size cellSize() const { return cellSize_; }

    void setRansacParams(RansacParams val) { ransacParams_ = val; }
    RansacParams ransacParams() const { return ransacParams_; }

    void process(  Size frameSize, InputArray points0, InputArray points1, OutputArray mask);

private:
    Size cellSize_;
    RansacParams ransacParams_;

    typedef std::vector<int> Cell;
    std::vector<Cell> grid_;
};


}







#endif
