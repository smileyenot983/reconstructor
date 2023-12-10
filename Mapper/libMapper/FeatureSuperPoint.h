#pragma once

#include <vector>

#include <torch/torch.h>
#include <torch/script.h>

#include "opencv2/features2d.hpp"

#include "FeatureDetector.h"

// Elapsed time: 3.96186 s
namespace reconstructor::Core
{
    class FeatureSuperPoint : public FeatureDetector
    {
    public:
        FeatureSuperPoint(const std::string &networkPath);

        void detect(const cv::Mat &img,
                    std::vector<FeaturePtr<>> &features) override;

        cv::Mat prepImg(const cv::Mat &img) override;

    private:
        torch::jit::script::Module superNet;

        // min keypoint confidence to use it as keypoint
        double CONF_THRESH = 0.015;
        // remove points closer than this to the borders
        unsigned BORDER_SIZE = 4;

        int inputImgType = CV_32F;
    };

} // namespace reconstructor::Core