#include "FeatureSuperPoint.h"

#include "BasicFlags.h"
#include "utils.h"

namespace reconstructor::Core
{

    /*
    Adds feature confidence in addition to coords
    */
    struct FeatCoordConf : FeatCoord
    {
        FeatCoordConf() {}

        FeatCoordConf(const int x, const int y, const double conf)
            : FeatCoord(x, y), conf(conf)
        {
        }

        double conf;
    };

    bool kpComp(FeatCoordConf kp1, FeatCoordConf kp2)
    {
        return kp1.conf > kp2.conf;
    }

    /*
    applies non-maxima suppression to remove repeating keypoints
    */
    std::vector<FeatCoordConf> nmsFast(std::vector<FeatCoordConf> &keypoints,
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

        std::vector<FeatCoordConf> keypointsFiltered;
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
    keypoints should be postprocessed after network predictions
    */
    std::vector<FeatCoord> processKeypoints(torch::Tensor keypointTensor,
                                            const int imgHeight,
                                            const int imgWidth,
                                            const double confThresh,
                                            const int borderSize)
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
        std::cout << "noDust.sizes(): " << noDust.sizes() << std::endl;
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

        std::vector<FeatCoordConf> keypointCoordConfs;

        for (size_t i = 0; i < imgHeight; ++i)
        {
            for (size_t j = 0; j < imgWidth; ++j)
            {
                auto kpConfidence = heatMap[i][j].item<float>();
                if (kpConfidence >= confThresh)
                {
                    FeatCoordConf keypointCoordConf(j, i, kpConfidence);
                    keypointCoordConfs.push_back(keypointCoordConf);
                }
            }
        }

        // if (verbose)
        // {
        //     std::cout << "keyPoints found: " << keypointCoordConfs.size() << std::endl;
        // }

        // std::vector<int> filteredIds;
        auto keypointsFiltered = nmsFast(keypointCoordConfs, imgHeight, imgWidth);

        LOG_MSG("keypointsFiltered.size(): " + std::to_string(keypointsFiltered.size()));

        auto keypointsNoBorder = removeBorderKeypoints(keypointsFiltered,
                                                       imgHeight,
                                                       imgWidth,
                                                       borderSize);

        // remove confidence from keypoints as it is not needed now
        std::vector<FeatCoord> featCoords;
        for (const auto &keypoint : keypointsNoBorder)
        {
            FeatCoord featCoord(keypoint.x, keypoint.y);
            featCoords.push_back(featCoord);
        }

        return featCoords;
    }

    // extracts descriptors on a given keypoints positions
    std::vector<FeatDesc> processDescriptors(const torch::Tensor &descriptors,
                                             std::vector<FeatCoord> keypoints)
    {
        // convert keypoint positions from image coords to compressed coords(/8)
        std::vector<FeatDesc> descriptorsFiltered;

        for (const auto &keypoint : keypoints)
        {
            int xCompressed = keypoint.x / 8;
            int yCompressed = keypoint.y / 8;

            // std::cout << "keypoint.x: " << keypoint.x << std::endl;
            // std::cout << "xCompressed: " << xCompressed << std::endl;

            auto descTensor = descriptors[0][yCompressed][xCompressed].slice(0, 0, 256);
            descTensor = descTensor.contiguous();
            // https://discuss.pytorch.org/t/how-to-convert-at-tensor-to-std-vector-float/92764
            // std::vector<float> v(t.data_ptr<float>(), t.data_ptr<float>() + t.numel());

            FeatDesc descFloat(descTensor.data_ptr<float>(), descTensor.data_ptr<float>() + descTensor.numel());

            // normalize desc:
            auto descSum = std::accumulate(descFloat.desc.begin(), descFloat.desc.end(), 0);
            for (size_t idx = 0; idx < descFloat.desc.size(); ++idx)
            {
                descFloat.desc[idx] /= descSum;
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
                                   std::vector<Feature> &features)
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
            Feature feat(keypointsProcessed[i], descriptorsProcessed[i]);
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