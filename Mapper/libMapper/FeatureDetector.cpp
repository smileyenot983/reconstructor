#include "FeatureDetector.h"

#include <opencv2/opencv.hpp>

namespace reconstructor::Core
{
    FeatureClassic::FeatureClassic()
    {
        // orbDetector = cv::ORB::create();
        siftDetector = cv::SIFT::create();
    }

    void FeatureClassic::detect(const cv::Mat& img,
                            std::vector<FeaturePtr<>>& features)
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        // orbDetector->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
        siftDetector->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

        int descSize = descriptors.size[1];

        descriptors.convertTo(descriptors, CV_32F);
        for (size_t featIdx = 0; featIdx < keypoints.size(); ++featIdx)
        {
            FeaturePtr<> feat = std::make_shared<Feature<>>(); 
            feat->featCoord.x = keypoints[featIdx].pt.x;
            feat->featCoord.y = keypoints[featIdx].pt.y;
            feat->featDesc.desc.resize(descSize);

            descriptors.row(featIdx).copyTo(feat->featDesc.desc);
            features.push_back(feat);
        }
    }

    cv::Mat FeatureClassic::prepImg(const cv::Mat &img)
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