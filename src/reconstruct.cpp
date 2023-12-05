// TODO:
// 1. make ImgPreparator class

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <chrono>  


#include <opencv2/opencv.hpp>

#include <libMapper/FeatureDetector.h>
#include <libMapper/FeatureSuperPoint.h>
#include <libMapper/utils.cpp>

unsigned IMG_MAX_SIZE = 512;

/*
Reshapes input image in place, making sure that resulting
image sides are divisible by 8
*/
void reshape_img(cv::Mat &img)
{
    if (img.rows > img.cols)
    {
        if (img.rows > IMG_MAX_SIZE)
        {
            auto aspectRatio = static_cast<double>(img.cols) / img.rows;
            auto rowSize = IMG_MAX_SIZE;
            auto colSize = rowSize * aspectRatio;
            colSize = colSize - std::fmod(colSize, 8);

            cv::resize(img, img, cv::Size(colSize, rowSize));
        }
    }
    else
    {
        if (img.cols > IMG_MAX_SIZE)
        {
            auto aspectRatio = static_cast<double>(img.rows) / img.cols;
            auto colSize = IMG_MAX_SIZE;
            auto rowSize = colSize * aspectRatio;
            rowSize = rowSize - std::fmod(rowSize, 8);

            cv::resize(img, img, cv::Size(colSize, rowSize));
        }
    }
}

int main()
{
    std::string test_img_path = "../data/DPP_0001.JPG";
    auto imgOriginal = cv::imread(test_img_path);
    reshape_img(imgOriginal);

    cv::Mat imgGray;
    cv::cvtColor(imgOriginal, imgGray, cv::COLOR_BGR2GRAY);

    std::vector<reconstructor::Core::Feature> features;

    auto start = std::chrono::high_resolution_clock::now();

    std::unique_ptr<reconstructor::Core::FeatureDetector> detector = std::make_unique<reconstructor::Core::FeatureORB>();
    // std::unique_ptr<reconstructor::Core::FeatureDetector> detector = std::make_unique<reconstructor::Core::FeatureSuperPoint>("../models/superpoint_model.zip");
    auto imgPrepared = detector->prepImg(imgGray);
    detector->detect(imgPrepared, features);

    std::cout << "n features detected: " << features.size() << std::endl;

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";



    // reconstructor::Utils::visualizeKeypoints(imgOriginal, features);

    return 0;
}