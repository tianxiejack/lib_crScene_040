#include "GlobalMotion.h"
#include "gms_matcher.hpp"

namespace OptFlowTrk{

inline float sqr(float x) { return x * x; }

static Mat estimateGlobMotionLeastSquaresTranslation(
        int npoints, const Point2f *points0, const Point2f *points1, float *rmse)
{
    Mat_<float> M = Mat::eye(3, 3, CV_32F);
    for (int i = 0; i < npoints; ++i)
    {
        M(0,2) += points1[i].x - points0[i].x;
        M(1,2) += points1[i].y - points0[i].y;
    }
    M(0,2) /= npoints;
    M(1,2) /= npoints;
    if (rmse)
    {
        *rmse = 0;
        for (int i = 0; i < npoints; ++i)
            *rmse += sqr(points1[i].x - points0[i].x - M(0,2)) +
                     sqr(points1[i].y - points0[i].y - M(1,2));
        *rmse = sqrt(*rmse / npoints);
    }
    return M;
}


static Mat estimateGlobMotionLeastSquaresTranslationAndScale(
        int npoints, const Point2f *points0, const Point2f *points1, float *rmse)
{
    Mat_<float> A(2*npoints, 3), b(2*npoints, 1);
    float *a0, *a1;
    Point2f p0, p1;

    for (int i = 0; i < npoints; ++i)
    {
        a0 = A[2*i];
        a1 = A[2*i+1];
        p0 = points0[i];
        p1 = points1[i];
        a0[0] = p0.x; a0[1] = 1; a0[2] = 0;
        a1[0] = p0.y; a1[1] = 0; a1[2] = 1;
        b(2*i,0) = p1.x;
        b(2*i+1,0) = p1.y;
    }

    Mat_<float> sol;
    solve(A, b, sol, DECOMP_SVD);

    if (rmse)
        *rmse = static_cast<float>(norm(A*sol, b, NORM_L2) / sqrt(static_cast<double>(npoints)));

    Mat_<float> M = Mat::eye(3, 3, CV_32F);
    M(0,0) = M(1,1) = sol(0,0);
    M(0,2) = sol(1,0);
    M(1,2) = sol(2,0);
    return M;
}


static Mat estimateGlobMotionLeastSquaresLinearSimilarity(
        int npoints, const Point2f *points0, const Point2f *points1, float *rmse)
{
    Mat_<float> A(2*npoints, 4), b(2*npoints, 1);
    float *a0, *a1;
    Point2f p0, p1;

    for (int i = 0; i < npoints; ++i)
    {
        a0 = A[2*i];
        a1 = A[2*i+1];
        p0 = points0[i];
        p1 = points1[i];
        a0[0] = p0.x; a0[1] = p0.y; a0[2] = 1;  a0[3] = 0;
        a1[0] = p0.y; a1[1] = -p0.x; a1[2] = 0; a1[3] = 1;
        b(2*i,0) = p1.x;
        b(2*i+1,0) = p1.y;
    }

    Mat_<float> sol;
    solve(A, b, sol, DECOMP_SVD);

    if (rmse)
        *rmse = static_cast<float>(norm(A*sol, b, NORM_L2) / sqrt(static_cast<double>(npoints)));

    Mat_<float> M = Mat::eye(3, 3, CV_32F);
    M(0,0) = M(1,1) = sol(0,0);
    M(0,1) = sol(1,0);
    M(1,0) = -sol(1,0);
    M(0,2) = sol(2,0);
    M(1,2) = sol(3,0);
    return M;
}


static Mat estimateGlobMotionLeastSquaresAffine(
        int npoints, const Point2f *points0, const Point2f *points1, float *rmse)
{
    Mat_<float> A(2*npoints, 6), b(2*npoints, 1);
    float *a0, *a1;
    Point2f p0, p1;

    for (int i = 0; i < npoints; ++i)
    {
        a0 = A[2*i];
        a1 = A[2*i+1];
        p0 = points0[i];
        p1 = points1[i];
        a0[0] = p0.x; a0[1] = p0.y; a0[2] = 1; a0[3] = a0[4] = a0[5] = 0;
        a1[0] = a1[1] = a1[2] = 0; a1[3] = p0.x; a1[4] = p0.y; a1[5] = 1;
        b(2*i,0) = p1.x;
        b(2*i+1,0) = p1.y;
    }

    Mat_<float> sol;
    solve(A, b, sol, DECOMP_SVD);

    if (rmse)
        *rmse = static_cast<float>(norm(A*sol, b, NORM_L2) / sqrt(static_cast<double>(npoints)));

    Mat_<float> M = Mat::eye(3, 3, CV_32F);
    for (int i = 0, k = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j, ++k)
            M(i,j) = sol(k,0);

    return M;
}

Mat estimateGlobalMotionLeastSquares(
        const vector<Point2f> &points0, const vector<Point2f> &points1, int model, float *rmse)
{
    CV_Assert(points0.size() == points1.size());

    typedef Mat (*Impl)(int, const Point2f*, const Point2f*, float*);
    static Impl impls[] = { estimateGlobMotionLeastSquaresTranslation,
                            estimateGlobMotionLeastSquaresTranslationAndScale,
                            estimateGlobMotionLeastSquaresLinearSimilarity,
                            estimateGlobMotionLeastSquaresAffine };

    int npoints = static_cast<int>(points0.size());
    return impls[model](npoints, &points0[0], &points1[0], rmse);
}

Mat estimateGlobalMotionRobust(
        vector<Point2f> &points0, vector<Point2f> &points1, int model,
        const RansacParams &params, float *rmse, int *ninliers)
{
    CV_Assert(points0.size() == points1.size());

    typedef Mat (*Impl)(int, const Point2f*, const Point2f*, float*);
    static Impl impls[] = { estimateGlobMotionLeastSquaresTranslation,
                            estimateGlobMotionLeastSquaresTranslationAndScale,
                            estimateGlobMotionLeastSquaresLinearSimilarity,
                            estimateGlobMotionLeastSquaresAffine };

    const int npoints = static_cast<int>(points0.size());
    if (npoints < params.size)
        return Mat::eye(3, 3, CV_32F);

    const int niters = static_cast<int>(ceil(log(1 - params.prob) /
                                             log(1 - pow(1 - params.eps, params.size))));

    RNG rng(0);
    vector<int> indices(params.size);
    vector<Point2f> subset0(params.size), subset1(params.size);
    vector<Point2f> subset0best(params.size), subset1best(params.size);
    Mat_<float> bestM;
    int ninliersMax = -1;
    Point2f p0, p1;
    float x, y;

    for (int iter = 0; iter < niters; ++iter)
    {
        for (int i = 0; i < params.size; ++i)
        {
            bool ok = false;
            while (!ok)
            {
                ok = true;
                indices[i] = static_cast<unsigned>(rng) % npoints;
                for (int j = 0; j < i; ++j)
                    if (indices[i] == indices[j])
                        { ok = false; break; }
            }
        }
        for (int i = 0; i < params.size; ++i)
        {
            subset0[i] = points0[indices[i]];
            subset1[i] = points1[indices[i]];
        }

        Mat_<float> M = impls[model](params.size, &subset0[0], &subset1[0], 0);

        int _ninliers = 0;
        for (int i = 0; i < npoints; ++i)
        {
            p0 = points0[i]; p1 = points1[i];
            x = M(0,0)*p0.x + M(0,1)*p0.y + M(0,2);
            y = M(1,0)*p0.x + M(1,1)*p0.y + M(1,2);
            if (sqr(x - p1.x) + sqr(y - p1.y) < params.thresh * params.thresh)
                _ninliers++;
        }
        if (_ninliers >= ninliersMax)
        {
            bestM = M;
            ninliersMax = _ninliers;
            subset0best.swap(subset0);
            subset1best.swap(subset1);
        }
    }

    if (ninliersMax < params.size)
    {
        // compute rmse
        bestM = impls[model](params.size, &subset0best[0], &subset1best[0], rmse);
    	std::swap(points0, subset0best);
    	std::swap(points1, subset1best);
    }
    else
    {
        subset0.resize(ninliersMax);
        subset1.resize(ninliersMax);
        for (int i = 0, j = 0; i < npoints; ++i)
        {
            p0 = points0[i]; p1 = points1[i];
            x = bestM(0,0)*p0.x + bestM(0,1)*p0.y + bestM(0,2);
            y = bestM(1,0)*p0.x + bestM(1,1)*p0.y + bestM(1,2);
            if (sqr(x - p1.x) + sqr(y - p1.y) < params.thresh * params.thresh)
            {
                subset0[j] = p0;
                subset1[j] = p1;
                j++;
            }
        }
        bestM = impls[model](ninliersMax, &subset0[0], &subset1[0], rmse);
        std::swap(points0, subset0);
        std::swap(points1, subset1);
    }

    if (ninliers)
        *ninliers = ninliersMax;

    return bestM;
}

static void KeyPointsToPoints(const vector<KeyPoint>& kps, vector<Point2f>& ps)
{
	ps.clear();
	for (unsigned int i=0; i<kps.size(); i++) ps.push_back(kps[i].pt);
}

void PointsToKeyPoints(const vector<Point2f>& ps, vector<KeyPoint>& kps) {
	kps.clear();
	for (unsigned int i=0; i<ps.size(); i++) kps.push_back(KeyPoint(ps[i],1.0f));
}

Mat estimateGlobalMotionGMS(
         const std::vector<Point2f> &points0, const std::vector<Point2f> &points1, cv::Size sz,
         int model /*= AFFINE*/, const RansacParams &params /*= RansacParams::affine2dMotionStd()*/,
         float *rmse /*= 0*/, int *ninliers /*= 0*/)
{
    CV_Assert(points0.size() == points1.size());

    typedef Mat (*Impl)(int, const Point2f*, const Point2f*, float*);
    static Impl impls[] = { estimateGlobMotionLeastSquaresTranslation,
                            estimateGlobMotionLeastSquaresTranslationAndScale,
                            estimateGlobMotionLeastSquaresLinearSimilarity,
                            estimateGlobMotionLeastSquaresAffine };

    const int npoints = static_cast<int>(points0.size());
    if (npoints < params.size)
        return Mat::eye(3, 3, CV_32F);

    const int niters = static_cast<int>(ceil(log(1 - params.prob) /
                                             log(1 - pow(1 - params.eps, params.size))));

    RNG rng(0);
    vector<int> indices(params.size);
    vector<Point2f> subset0(params.size), subset1(params.size);
    vector<Point2f> subset0best(params.size), subset1best(params.size);
    Mat_<float> bestM;
    int ninliersMax = -1;
    Point2f p0, p1;
    float x, y;
    int i;

    vector<KeyPoint> kp0, kp1;
    vector<DMatch> matches_all, matches_gms;

    PointsToKeyPoints(points0, kp0);
    PointsToKeyPoints(points1, kp1);

    for(i=0; i<points0.size(); i++)
    	matches_all.push_back(DMatch(i, i, 1.f));

    // GMS filter
	std::vector<bool> vbInliers;
	gms_matcher gms(kp0, sz, kp1, sz, matches_all);
	ninliersMax = gms.GetInlierMask(vbInliers, false, false);
	subset0best.clear();
	subset1best.clear();

	for (i = 0; i < vbInliers.size(); ++i)
	{
		if (vbInliers[i] == true)
		{
			matches_gms.push_back(matches_all[i]);
			subset0best.push_back(points0[i]);
			subset0best.push_back(points1[i]);
		}
	}

    if (ninliersMax < params.size)
    {
        bestM = impls[model](params.size, &subset0best[0], &subset1best[0], rmse);// compute rmse
    }
    else
    {
        bestM = impls[model](ninliersMax, &subset0best[0], &subset1best[0], rmse);
    }

    if (ninliers)
        *ninliers = ninliersMax;

    return bestM;
}

}
