#include "FeatureDetector.h"

#include <opencv2/opencv.hpp>

namespace reconstructor::Core
{
    FeatureORB::FeatureORB()
    {
        orbDetector = cv::ORB::create();
    }

    void FeatureORB::detect(const cv::Mat &img,
                            std::vector<Feature> &features)
    {
        std::cout << "img.size: " << img.size << std::endl;

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        orbDetector->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

        features.resize(keypoints.size());
        int descSize = descriptors.size[1];

        for (size_t featIdx = 0; featIdx < keypoints.size(); ++featIdx)
        {
            features[featIdx].featCoord.x = keypoints[featIdx].pt.x;
            features[featIdx].featCoord.y = keypoints[featIdx].pt.y;
            features[featIdx].featDesc.desc.resize(descSize);
            descriptors.row(featIdx).copyTo(features[featIdx].featDesc.desc);
        }
    }

    cv::Mat FeatureORB::prepImg(const cv::Mat &img)
    {
        cv::Mat imgPrepared;

        if (img.type() != inputImgType)
        {
            img.convertTo(imgPrepared, inputImgType);
        }
        else
        {
            img.copyTo(imgPrepared);
        }
        return imgPrepared;
    }
}