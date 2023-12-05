#include "utils.h"

namespace reconstructor::Utils
{

void visualizeKeypoints(const cv::Mat& img,
                        const std::vector<reconstructor::Core::FeatCoord>& featureCoords,
                        bool saveImage,
                        bool showImage)
{
    std::cout << "img input type for drawing kps: " << img.type() << std::endl;

    std::cout << "img.size(): " << img.size() << std::endl;

    cv::Mat imgKeypoints = img.clone();

    // multiply by 255, because it was divided previously
    if(img.type())
    // for(size_t i=0;i<imgKeypoints.rows;++i)
    // {
    //     for(size_t j=0; j<imgKeypoints.cols; ++j)
    //     {
    //         imgKeypoints.at<float>(i,j) *= 255;
    //     }
    // }

    // imgKeypoints.convertTo(imgKeypoints, CV_8U);

    std::cout << "img draw type for drawing kps: " << imgKeypoints.type() << std::endl;
    
    // convert keypoints to cv format
    std::vector<cv::KeyPoint> cvKeypoints;

    for(const auto& kp : featureCoords)
    {
        cv::KeyPoint cvKeypoint(cv::Point2f(kp.x, kp.y), 2);
        cvKeypoints.push_back(cvKeypoint);
    }

    cv::drawKeypoints(imgKeypoints, cvKeypoints, imgKeypoints);

    cv::imshow("image keypoints", imgKeypoints);
    cv::waitKey(0);
}

void visualizeKeypoints(const cv::Mat& img,
                        const std::vector<reconstructor::Core::Feature>& features,
                        bool saveImage,
                        bool showImage)
{
    std::vector<reconstructor::Core::FeatCoord> featureCoords;
    for(const auto& feat : features)
    {
        featureCoords.push_back(feat.featCoord);
    }

    visualizeKeypoints(img, featureCoords, saveImage, showImage);
}

}