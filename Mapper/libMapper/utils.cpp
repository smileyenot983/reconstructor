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

}