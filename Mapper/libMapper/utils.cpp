#include "utils.h"

namespace reconstructor::Utils
{

void visualizeKeypoints(const cv::Mat& img,
                        const std::vector<reconstructor::Core::FeatCoord<>>& featureCoords,
                        int imgIdx,
                        bool saveImage)
{

    std::cout << "img.size(): " << img.size() << std::endl;

    cv::Mat imgKeypoints = img.clone();

    
    // convert keypoints to cv format
    std::vector<cv::KeyPoint> cvKeypoints;

    for(const auto& kp : featureCoords)
    {
        cv::KeyPoint cvKeypoint(cv::Point2f(kp.x, kp.y), 2);
        cvKeypoints.push_back(cvKeypoint);
    }

    cv::drawKeypoints(imgKeypoints, cvKeypoints, imgKeypoints);

    if(saveImage)
    {
        std::string imgPath = "img_" + std::to_string(imgIdx) + ".png";
        cv::imwrite(imgPath, imgKeypoints);
    }
    else
    {
        cv::imshow("image keypoints", imgKeypoints);
        cv::waitKey(0);
    }
}

void visualizeKeypoints(const cv::Mat& img,
                        const std::vector<reconstructor::Core::FeaturePtr<>>& features,
                        int imgIdx,
                        bool saveImage)
{
    std::vector<reconstructor::Core::FeatCoord<int>> featureCoords;
    for(const auto& feat : features)
    {
        featureCoords.push_back(feat->featCoord);
    }

    visualizeKeypoints(img, featureCoords, imgIdx, saveImage);
}

void reshapeImg(cv::Mat &img,
                 const int imgMaxSize)
{
    if (img.rows > img.cols)
    {
        if (img.rows > imgMaxSize)
        {
            auto aspectRatio = static_cast<double>(img.cols) / img.rows;
            auto rowSize = imgMaxSize;
            auto colSize = rowSize * aspectRatio;
            colSize = colSize - std::fmod(colSize, 8);

            cv::resize(img, img, cv::Size(colSize, rowSize));
        }
    }
    else
    {
        if (img.cols > imgMaxSize)
        {
            auto aspectRatio = static_cast<double>(img.rows) / img.cols;
            auto colSize = imgMaxSize;
            auto rowSize = colSize * aspectRatio;
            rowSize = rowSize - std::fmod(rowSize, 8);

            cv::resize(img, img, cv::Size(colSize, rowSize));
        }
    }
}

cv::Mat readGrayImg(const std::string& imgPath,
                       const int imgMaxSize)
{
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    reshapeImg(img, imgMaxSize);
    // cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    return img;
}

std::vector<FeaturePtr<double>> normalizeFeatCoords(const std::vector<FeaturePtr<>>& features,
                                                     const int imgHeight,
                                                     const int imgWidth,
                                                     const double normalizationRange)
{   
    std::vector<FeaturePtr<double>> featuresNormalized;

    // scaling to have kpts in range [-0.7, 0.7]
    auto imgCenterX = imgWidth / 2; // = 480/2 = 240
    auto imgCenterY = imgHeight / 2; // = 640/2 = 320
    // casted to double, cause floating point coords required
    auto imgScale = static_cast<double>(std::max(imgHeight, imgWidth) * normalizationRange); // 640*0.7 = 448

    for(size_t i = 0; i < features.size(); ++i)
    {
        // copy into new normalized feature
        FeatureConfPtr<> featOriginal = std::dynamic_pointer_cast<FeatureConf<>>(features[i]);

        FeatCoord<double> coordNormalized;
        // [10, 20]
        coordNormalized.x = (featOriginal->featCoord.x - imgCenterX) / imgScale;
        coordNormalized.y = (featOriginal->featCoord.y - imgCenterY) / imgScale;
        
        FeaturePtr<double> featNormalized = std::make_shared<FeatureConf<double>>(coordNormalized,
                                                                        featOriginal->featDesc,
                                                                        featOriginal->conf);

        featuresNormalized.push_back(featNormalized);
    }

    return featuresNormalized;
}

double focalLengthmmToPx(const double focalLengthmm,
                         const double imgDim,
                         const double fovDegrees)
{
    // convert fov to radians
    double fovRadians = fovDegrees * 3.1415 / 180.0;

    double focalLengthPx = imgDim / (2.0 * tan(fovRadians / 2.0));

    return focalLengthPx;
}

Eigen::Matrix3d getIntrinsicsMat(const double focalLengthmm,
                                 const int imgHeight,
                                 const int imgWidth,
                                 const double fovDegrees)
{
    Eigen::Matrix3d intrinsics = Eigen::Matrix3d::Identity();

    // get focal lengths:
    auto f_x = focalLengthmmToPx(focalLengthmm, imgWidth, fovDegrees);
    auto f_y = focalLengthmmToPx(focalLengthmm, imgHeight, fovDegrees);

    auto c_x = imgWidth / 2;
    auto c_y = imgHeight / 2;

    intrinsics(0,0) = f_x;
    intrinsics(0,2) = c_x;
    intrinsics(1,1) = f_y;
    intrinsics(1,2) = c_y;

    return intrinsics;
}

std::vector<cv::Point2f> featuresToCvPoints(const std::vector<FeaturePtr<>>& features)
{
    std::vector<cv::Point2f> featuresCV(features.size());

    for(size_t featIdx = 0; featIdx < features.size(); ++featIdx)
    {

        cv::Point2f pt(static_cast<float>(features[featIdx]->featCoord.x),
                       static_cast<float>(features[featIdx]->featCoord.y));
        featuresCV[featIdx] = pt;

    }
    return featuresCV;
}

cv::Mat eigen3dToCVMat(const Eigen::Matrix3d& eigenMat, int dType)
{
    cv::Mat cvMat(eigenMat.rows(), eigenMat.cols(), dType);
    for(size_t row=0; row < 3; ++row)
    {
        for(size_t col=0; col < 3; ++col)
        {
            cvMat.at<float>(row,col) = eigenMat(row,col);
        }
    }
    return cvMat;
}

Eigen::Matrix3d cvMatToEigen3d(const cv::Mat& cvMat)
{
    Eigen::Matrix3d eigenMat;
    for(size_t row=0; row < 3; ++row)
    {
        for(size_t col=0; col < 3; ++col)
        {
            eigenMat(row, col) = cvMat.at<double>(row,col);
        }
    }
    return eigenMat;
}

}