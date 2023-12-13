#pragma once
/*
Contains several useful utilities for debugging/visualization purposes
*/
#include "FeatureDetector.h"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>


namespace reconstructor::Utils
{

void visualizeKeypoints(const cv::Mat& img,
                        const std::vector<reconstructor::Core::FeatCoord<>>& featureCoords,
                        int imgIdx = 0,
                        bool saveImage = false);

void visualizeKeypoints(const cv::Mat& img,
                        const std::vector<reconstructor::Core::FeaturePtr<>>& features,
                        int imgIdx = 0,
                        bool saveImage = false);

}