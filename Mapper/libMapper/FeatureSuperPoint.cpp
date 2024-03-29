#include "FeatureSuperPoint.h"

#include "BasicFlags.h"
#include "utils.h"

namespace reconstructor::Core
{
    bool kpComp(FeatCoordConf<> kp1, FeatCoordConf<> kp2)
    {
        return kp1.conf > kp2.conf;
    }
    /*
       Applies non-maxima suppression to remove repeating keypoints
    */
    std::vector<FeatCoordConf<>> nmsFast(std::vector<FeatCoordConf<>> &keypoints,
                                       const int imgHeight,
                                       const int imgWidth,
                                       const int distThresh = 4)
    {
        // create 2d matrices for storing keypoints and corresponding idx
        std::vector<std::vector<int>> grid(imgHeight + 2 * distThresh, std::vector<int>(imgWidth + 2 * distThresh, 0));
        std::vector<std::vector<int>> inds(imgHeight + 2 * distThresh, std::vector<int>(imgWidth + 2 * distThresh, 0));

        std::sort(keypoints.begin(), keypoints.end(), kpComp);

        for (size_t i = 0; i < keypoints.size(); ++i)
        {
            auto kpX = keypoints[i].x + distThresh;
            auto kpY = keypoints[i].y + distThresh;

            grid[kpY][kpX] = 1;
            inds[kpY][kpX] = i;
        }

        // suppressing points near current keypoint
        for (const auto &keypoint : keypoints)
        {
            auto kpX = keypoint.x + distThresh;
            auto kpY = keypoint.y + distThresh;

            if (grid[kpY][kpX] == 1)
            {
                for (size_t pxY = kpY - distThresh; pxY < kpY + distThresh + 1; ++pxY)
                {
                    for (size_t pxX = kpX - distThresh; pxX < kpX + distThresh + 1; ++pxX)
                    {
                        grid[pxY][pxX] = 0;
                    }
                }
                grid[kpY][kpX] = 1;
            }
        }

        std::vector<FeatCoordConf<>> keypointsFiltered;
        // find survived keypoints, i.e. where grid value = 1 after filtering
        for (size_t pxY = 0; pxY < grid.size(); ++pxY)
        {
            for (size_t pxX = 0; pxX < grid[pxY].size(); ++pxX)
            {
                // subtract distThresh from coords
                if (grid[pxY][pxX] == 1)
                {
                    auto kpIdx = inds[pxY][pxX];
                    keypointsFiltered.push_back(keypoints[kpIdx]);
                }
            }
        }

        return keypointsFiltered;
    }

    template <typename T>
    std::vector<T> removeBorderKeypoints(const std::vector<T> keypoints,
                                         const int imgHeight,
                                         const int imgWidth,
                                         const unsigned borderSize = 4)
    {
        std::vector<T> keypointsNoBorder;

        for (const auto &keypoint : keypoints)
        {
            if (keypoint.x < borderSize || keypoint.x >= (imgWidth - borderSize) || keypoint.y < borderSize || keypoint.y >= (imgHeight - borderSize))
            {
                continue;
            }
            keypointsNoBorder.push_back(keypoint);
        }
        return keypointsNoBorder;
    }

    /*
    Converts squeezed(h/8, w/8, 64) neural network output to original image shape(h,w,1)
    where values represent point confidence
    */
    torch::Tensor extractHeatMap(const torch::Tensor& keypointTensor,
                        const int imgHeight,
                        const int imgWidth)
    {
        // 1. apply softmax over depth to get probs
        auto dense = at::exp(keypointTensor);

        const auto nDepthsIn = dense.sizes()[0];
        const auto nRowsIn = dense.sizes()[1];
        const auto nColsIn = dense.sizes()[2];

        for (size_t depthIdx = 0; depthIdx < nDepthsIn; ++depthIdx)
        {
            for (size_t rowIdx = 0; rowIdx < nRowsIn; ++rowIdx)
            {
                auto expSum = at::sum(dense[depthIdx]) + 1e-5;
                for (size_t colIdx = 0; colIdx < nColsIn; ++colIdx)
                {
                    dense[depthIdx][rowIdx][colIdx] /= expSum;
                }
            }
        }

        // last value in depth is used for "noKeypoint" prob-ty, remove it
        auto noDust = dense.slice(0, 0, 64);
        // put depth in last dim
        noDust = noDust.permute({1, 2, 0});

        // cellSize is defined by network architecture,
        // it defines how many times the initial image will decrease
        // it will have this compressed pixels in depth layer
        const int cellSize = 8;
        const int imgComprHeight = imgHeight / cellSize;
        const int imgComprWidth = imgWidth / cellSize;

        // reshape into initial width and height
        auto heatMap = noDust.reshape({imgComprHeight, imgComprWidth, cellSize, cellSize});
        heatMap = heatMap.permute({0, 2, 1, 3});
        heatMap = heatMap.reshape({imgComprHeight * cellSize, imgComprWidth * cellSize});

        // make sure we get same shape as original image
        assert(imgHeight == heatMap.sizes()[0]);
        assert(imgWidth == heatMap.sizes()[1]);

        return heatMap;
    }

    /*
    keypoints should be postprocessed after network predictions
    */
    std::vector<FeatCoordConf<>> processKeypoints(const torch::Tensor& keypointTensor,
                                                const int imgHeight,
                                                const int imgWidth,
                                                const double confThresh,
                                                const int borderSize)
    {
        auto heatMap = extractHeatMap(keypointTensor, imgHeight, imgWidth);

        std::vector<FeatCoordConf<>> keypointCoordConfs;

        for (size_t i = 0; i < imgHeight; ++i)
        {
            for (size_t j = 0; j < imgWidth; ++j)
            {
                auto kpConfidence = heatMap[i][j].item<float>();
                if (kpConfidence >= confThresh)
                {
                    FeatCoordConf<> keypointCoordConf(j, i, kpConfidence);
                    keypointCoordConfs.push_back(keypointCoordConf);
                }
            }
        }

        auto keypointsFiltered = nmsFast(keypointCoordConfs, imgHeight, imgWidth);


        auto keypointsNoBorder = removeBorderKeypoints(keypointsFiltered,
                                                        imgHeight,
                                                        imgWidth,
                                                        borderSize);



        return keypointsNoBorder;
    }

    // extracts descriptors on a given keypoints positions
    template <typename keypointType>
    std::vector<FeatDesc> processDescriptors(const torch::Tensor &descriptors,
                                             std::vector<keypointType> keypoints)
    {
        // convert keypoint positions from image coords to compressed coords(/8)
        std::vector<FeatDesc> descriptorsFiltered;

        for (const auto &keypoint : keypoints)
        {
            int xCompressed = keypoint.x / 8;
            int yCompressed = keypoint.y / 8;


            auto descTensor = descriptors[0][yCompressed][xCompressed].slice(0, 0, 256);
            descTensor = descTensor.contiguous();

            FeatDesc descFloat(descTensor.data_ptr<float>(), descTensor.data_ptr<float>() + descTensor.numel());

            // normalize desc:
            double descNorm = descFloat.norm();
            for (size_t idx = 0; idx < descFloat.desc.size(); ++idx)
            {
                descFloat.desc[idx] /= descNorm;
            }

            descriptorsFiltered.push_back(descFloat);
        }

        return descriptorsFiltered;
    }

    FeatureSuperPoint::FeatureSuperPoint(const std::string &networkPath)
    {
        try
        {
            superNet = torch::jit::load(networkPath);
        }
        catch (const c10::Error &e)
        {
            std::cerr << "erro loading superNet" << std::endl;
        }
    }

    /*
    img should be float, single channel, i.e. CV_32FC1
    */
    void FeatureSuperPoint::detect(const cv::Mat &img,
                                   std::vector<FeaturePtr<>> &features)
    {
        auto imgTensor = torch::from_blob(img.data, {1, img.rows, img.cols}, torch::kFloat32).unsqueeze(0);

        assert(img.at<float>(0, 0) == imgTensor[0][0][0][0].item<float>());

        std::vector<torch::jit::IValue> netInputs;
        netInputs.push_back(imgTensor);

        auto netOut = superNet.forward(netInputs);
        auto keypoints = netOut.toTuple()->elements()[0].toTensor();

        // filtered
        auto keypointsProcessed = processKeypoints(keypoints[0],
                                                   img.rows,
                                                   img.cols,
                                                   CONF_THRESH,
                                                   BORDER_SIZE);

        auto descriptors = netOut.toTuple()->elements()[1].toTensor();
        descriptors = descriptors.permute({0, 2, 3, 1});

        auto descriptorsProcessed = processDescriptors(descriptors,
                                                       keypointsProcessed);

        // assert that number of keypoints equal to number of descriptors
        assert(keypointsProcessed.size() == descriptorsProcessed.size());

        for (size_t i = 0; i < keypointsProcessed.size(); ++i)
        {
            FeaturePtr<> feat = std::make_shared<FeatureConf<>>(keypointsProcessed[i],
                                                                descriptorsProcessed[i]);
            features.push_back(feat);
        }
    }

    cv::Mat FeatureSuperPoint::prepImg(const cv::Mat &img)
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

        // normalization:
        for (size_t i = 0; i < imgPrepared.rows; ++i)
        {
            for (size_t j = 0; j < imgPrepared.cols; ++j)
            {
                imgPrepared.at<float>(i, j) /= 255.0;
            }
        }

        return imgPrepared;
    }

} // namespace reconstructor::Core