#include "utils.h"

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>

#include <thread>

namespace reconstructor::Utils
{

    void visualizeKeypoints(const cv::Mat &img,
                            const std::vector<reconstructor::Core::FeatCoord<>> &featureCoords,
                            int imgIdx,
                            bool saveImage)
    {

        std::cout << "img.size(): " << img.size() << std::endl;

        cv::Mat imgKeypoints = img.clone();

        // convert keypoints to cv format
        std::vector<cv::KeyPoint> cvKeypoints;

        for (const auto &kp : featureCoords)
        {
            cv::KeyPoint cvKeypoint(cv::Point2f(kp.x, kp.y), 2);
            cvKeypoints.push_back(cvKeypoint);
        }

        cv::drawKeypoints(imgKeypoints, cvKeypoints, imgKeypoints);

        if (saveImage)
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

    void visualizeKeypoints(const cv::Mat &img,
                            const std::vector<reconstructor::Core::FeaturePtr<>> &features,
                            int imgIdx,
                            bool saveImage)
    {
        std::vector<reconstructor::Core::FeatCoord<int>> featureCoords;
        for (const auto &feat : features)
        {
            featureCoords.push_back(feat->featCoord);
        }

        visualizeKeypoints(img, featureCoords, imgIdx, saveImage);
    }

    double reshapeImg(cv::Mat &img,
                      const int imgMaxSize)
    {
        if (img.rows > img.cols)
        {
            if (img.rows > imgMaxSize)
            {
                auto rowSizeOriginal = img.rows;
                auto aspectRatio = static_cast<double>(img.cols) / img.rows;
                auto rowSize = imgMaxSize;
                auto colSize = rowSize * aspectRatio;
                colSize = colSize - std::fmod(colSize, 8);

                cv::resize(img, img, cv::Size(colSize, rowSize));

                return static_cast<double>(rowSize) / rowSizeOriginal;
            }

            return 1.0;
        }
        else
        {
            if (img.cols > imgMaxSize)
            {
                auto colSizeOriginal = img.cols;
                auto aspectRatio = static_cast<double>(img.rows) / img.cols;
                auto colSize = imgMaxSize;
                auto rowSize = colSize * aspectRatio;
                rowSize = rowSize - std::fmod(rowSize, 8);

                cv::resize(img, img, cv::Size(colSize, rowSize));

                return static_cast<double>(colSize) / colSizeOriginal;
            }

            return 1.0;
        }
    }

    double readImg(cv::Mat &img,
                   const std::string &imgPath,
                   const int imgMaxSize)
    {
        img = cv::imread(imgPath);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        auto downScaleFactor = reshapeImg(img, imgMaxSize);

        return downScaleFactor;
    }

    cv::Mat readGrayImg(const std::string &imgPath,
                        const int imgMaxSize)
    {
        cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
        reshapeImg(img, imgMaxSize);
        return img;
    }

    std::vector<FeaturePtr<double>> normalizeFeatCoords(const std::vector<FeaturePtr<>> &features,
                                                        const int imgHeight,
                                                        const int imgWidth,
                                                        const double normalizationRange)
    {
        std::vector<FeaturePtr<double>> featuresNormalized;

        // scaling to have kpts in range [-0.7, 0.7]
        auto imgCenterX = imgWidth / 2;  // = 480/2 = 240
        auto imgCenterY = imgHeight / 2; // = 640/2 = 320
        // casted to double, cause floating point coords required
        auto imgScale = static_cast<double>(std::max(imgHeight, imgWidth) * normalizationRange); // 640*0.7 = 448

        for (size_t i = 0; i < features.size(); ++i)
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
        double fovRadians = fovDegrees * 3.1415 / 180.0; // 0,689384722

        double focalLengthPx = imgDim / (2.0 * tan(fovRadians / 2.0)); // fx = 1092 / 0.718  = 1520
                                                                       // fy = 728 / 0.718 = 1014

        return focalLengthPx;
    }

    std::vector<cv::Point2f> featuresToCvPoints(const std::vector<FeaturePtr<>> &features)
    {
        std::vector<cv::Point2f> featuresCV(features.size());

        for (size_t featIdx = 0; featIdx < features.size(); ++featIdx)
        {

            cv::Point2f pt(static_cast<float>(features[featIdx]->featCoord.x),
                           static_cast<float>(features[featIdx]->featCoord.y));
            featuresCV[featIdx] = pt;
        }
        return featuresCV;
    }

    cv::Mat eigen3dToCVMat(const Eigen::Matrix3d &eigenMat, int dType)
    {
        cv::Mat cvMat(eigenMat.rows(), eigenMat.cols(), dType);
        for (size_t row = 0; row < 3; ++row)
        {
            for (size_t col = 0; col < 3; ++col)
            {
                cvMat.at<float>(row, col) = eigenMat(row, col);
            }
        }
        return cvMat;
    }

    Eigen::Matrix3d cvMatToEigen3d(const cv::Mat &cvMat)
    {
        Eigen::Matrix3d eigenMat;
        for (size_t row = 0; row < 3; ++row)
        {
            for (size_t col = 0; col < 3; ++col)
            {
                eigenMat(row, col) = cvMat.at<double>(row, col);
            }
        }
        return eigenMat;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr landmarksToPclCloud(const std::vector<Landmark> &landmarks)
    {
        // it is better to use it like that, using default shared_ptr would conflict with other pcl functions
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr landmark_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        for (const auto &landmark : landmarks)
        {
            pcl::PointXYZRGB pt(landmark.red, landmark.green, landmark.blue);
            pt.x = landmark.x;
            pt.y = landmark.y;
            pt.z = landmark.z;
            landmark_cloud->points.push_back(pt);
        }

        return landmark_cloud;
    }

    // same as previous one, but will paint outliers with red color
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr landmarksToPclCloud(const std::vector<Landmark> &landmarks,
                                                               const std::vector<bool> &inliers)
    {
        // it is better to use it like that, using default shared_ptr would conflict with other pcl functions
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr landmark_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        for (size_t landmarkId = 0; landmarkId < landmarks.size(); ++landmarkId)
        {
            auto channelRed = inliers[landmarkId] ? landmarks[landmarkId].red : 253;
            auto channelGreen = inliers[landmarkId] ? landmarks[landmarkId].green : 0;
            auto channelBlue = inliers[landmarkId] ? landmarks[landmarkId].blue : 0;

            pcl::PointXYZRGB pt(channelRed, channelGreen, channelBlue);
            pt.x = landmarks[landmarkId].x;
            pt.y = landmarks[landmarkId].y;
            pt.z = landmarks[landmarkId].z;
            landmark_cloud->points.push_back(pt);
        }

        for (const auto &landmark : landmarks)
        {
            pcl::PointXYZRGB pt(landmark.red, landmark.green, landmark.blue);
            pt.x = landmark.x;
            pt.y = landmark.y;
            pt.z = landmark.z;
            landmark_cloud->points.push_back(pt);
        }

        return landmark_cloud;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cameraPosesToPclCloud(const std::unordered_map<int, Eigen::Matrix4d> &imgIdx2camPose,
                                                                 int red, int green, int blue)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr camera_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        for (const auto &[imgIdx, camPose] : imgIdx2camPose)
        {
            Eigen::Vector3d translationVector = camPose.block<3, 1>(0, 3);
            Eigen::Matrix3d rotationMat = camPose.block<3, 3>(0, 0);

            // camera center = -R^T * t
            Eigen::Vector3d cameraCenter = -rotationMat.transpose() * translationVector;

            pcl::PointXYZRGB pt(red, green, blue);
            pt.x = cameraCenter(0);
            pt.y = cameraCenter(1);
            pt.z = cameraCenter(2);
            camera_cloud->points.push_back(pt);
        }

        return camera_cloud;
    }

    using namespace std::chrono_literals;
    void viewCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudLandmark,
                   const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudCamera)
    {
        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
        viewer->addPointCloud<pcl::PointXYZRGB>(cloudLandmark, "cloud_landmark");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud_landmark");

        viewer->addPointCloud<pcl::PointXYZRGB>(cloudCamera, "cloud_camera");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "cloud_camera");

        viewer->setBackgroundColor(0.1, 0.1, 0.1);
        viewer->initCameraParameters();

        while (!viewer->wasStopped())
        {
            viewer->spinOnce(100);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    using namespace std::chrono_literals;
    void viewCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudLandmark1,
                   const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudLandmark2,
                   const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudCamera1,
                   const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudCamera2)
    {

        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
        viewer->addPointCloud<pcl::PointXYZRGB>(cloudLandmark1, "cloud_landmark1");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud_landmark1");

        viewer->addPointCloud<pcl::PointXYZRGB>(cloudLandmark2, "cloud_landmark2");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud_landmark2");

        viewer->addPointCloud<pcl::PointXYZRGB>(cloudCamera1, "cloud_camera1");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "cloud_camera1");

        viewer->addPointCloud<pcl::PointXYZRGB>(cloudCamera2, "cloud_camera2");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "cloud_camera2");

        viewer->setBackgroundColor(0.1, 0.1, 0.1);
        viewer->initCameraParameters();

        while (!viewer->wasStopped())
        {
            viewer->spinOnce(100);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    void writeInliersToVector(const cv::Mat &inliersCV,
                              std::vector<bool> &inlierMatchIds)
    {
        for (size_t matchIdx = 0; matchIdx < inliersCV.rows; ++matchIdx)
        {

            if (inliersCV.at<uchar>(matchIdx))
            {
                inlierMatchIds.push_back(true);
            }
            else
            {
                inlierMatchIds.push_back(false);
            }
        }
    }

    void saveCloud(std::vector<Landmark> &landmarks,
                   std::unordered_map<int, Eigen::Matrix4d> &imgIdx2camPose,
                   const std::string &cloudPath)
    {
        auto posesCloud = cameraPosesToPclCloud(imgIdx2camPose, 0, 250, 0);
        auto landmarksCloud = landmarksToPclCloud(landmarks);

        *landmarksCloud += *posesCloud;

        pcl::io::savePLYFile(cloudPath, *landmarksCloud);
    }

    void saveCloud(std::vector<Landmark> &landmarks,
                   std::unordered_map<int, Eigen::Matrix4d> &imgIdx2camPose,
                   const std::string &cloudPath,
                   const std::vector<bool> &inliers)
    {
        auto posesCloud = cameraPosesToPclCloud(imgIdx2camPose, 0, 250, 0);
        auto landmarksCloud = landmarksToPclCloud(landmarks, inliers);

        *landmarksCloud += *posesCloud;

        pcl::io::savePLYFile(cloudPath, *landmarksCloud);
    }

    void deleteDirectoryContents(const std::filesystem::path &dir)
    {
        for (const auto &entry : std::filesystem::directory_iterator(dir))
            std::filesystem::remove_all(entry.path());
    }

}