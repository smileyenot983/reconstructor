#include "utils.h"

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <thread>

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

void reshapeImg(cv::Mat &img,
                 const int imgMaxSize)
{
    if (img.rows > img.cols)
    {
        if (img.rows > imgMaxSize)
        {
            auto aspectRatio = static_cast<double>(img.cols) / img.rows;
            auto rowSize = imgMaxSize;
            auto colSize = rowSize * aspectRatio;
            colSize = colSize - std::fmod(colSize, 8);

            cv::resize(img, img, cv::Size(colSize, rowSize));
        }
    }
    else
    {
        if (img.cols > imgMaxSize)
        {
            auto aspectRatio = static_cast<double>(img.rows) / img.cols;
            auto colSize = imgMaxSize;
            auto rowSize = colSize * aspectRatio;
            rowSize = rowSize - std::fmod(rowSize, 8);

            cv::resize(img, img, cv::Size(colSize, rowSize));
        }
    }
}

cv::Mat readGrayImg(const std::string& imgPath,
                       const int imgMaxSize)
{
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    reshapeImg(img, imgMaxSize);
    std::cout << "size before: " << img.size << std::endl;
    std::cout << "size after: " << img.size << std::endl;
    // cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    return img;
}

std::vector<FeaturePtr<double>> normalizeFeatCoords(const std::vector<FeaturePtr<>>& features,
                                                     const int imgHeight,
                                                     const int imgWidth,
                                                     const double normalizationRange)
{   
    std::vector<FeaturePtr<double>> featuresNormalized;

    // scaling to have kpts in range [-0.7, 0.7]
    auto imgCenterX = imgWidth / 2; // = 480/2 = 240
    auto imgCenterY = imgHeight / 2; // = 640/2 = 320
    // casted to double, cause floating point coords required
    auto imgScale = static_cast<double>(std::max(imgHeight, imgWidth) * normalizationRange); // 640*0.7 = 448

    for(size_t i = 0; i < features.size(); ++i)
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
    double fovRadians = fovDegrees * 3.1415 / 180.0;

    double focalLengthPx = imgDim / (2.0 * tan(fovRadians / 2.0));

    return focalLengthPx;
}

Eigen::Matrix3d getIntrinsicsMat(const double focalLengthmm,
                                 const int imgHeight,
                                 const int imgWidth,
                                 const double fovDegrees)
{
    Eigen::Matrix3d intrinsics = Eigen::Matrix3d::Identity();

    // get focal lengths:
    auto f_x = focalLengthmmToPx(focalLengthmm, imgWidth, fovDegrees);
    auto f_y = focalLengthmmToPx(focalLengthmm, imgHeight, fovDegrees);

    auto c_x = imgWidth / 2;
    auto c_y = imgHeight / 2;

    intrinsics(0,0) = f_x;
    intrinsics(0,2) = c_x;
    intrinsics(1,1) = f_y;
    intrinsics(1,2) = c_y;

    return intrinsics;
}

Eigen::Matrix3d getIntrinsicsMat(const double focalLengthPx,
                                 const int imgHeight,
                                 const int imgWidth)
{
    Eigen::Matrix3d intrinsics = Eigen::Matrix3d::Identity();

    intrinsics(0,0) = focalLengthPx;
    intrinsics(1,1) = focalLengthPx;
    intrinsics(0,2) = imgWidth / 2;
    intrinsics(1,2) = imgHeight / 2;

    return intrinsics;

}

std::vector<cv::Point2f> featuresToCvPoints(const std::vector<FeaturePtr<>>& features)
{
    std::vector<cv::Point2f> featuresCV(features.size());

    for(size_t featIdx = 0; featIdx < features.size(); ++featIdx)
    {

        cv::Point2f pt(static_cast<float>(features[featIdx]->featCoord.x),
                       static_cast<float>(features[featIdx]->featCoord.y));
        featuresCV[featIdx] = pt;

    }
    return featuresCV;
}

cv::Mat eigen3dToCVMat(const Eigen::Matrix3d& eigenMat, int dType)
{
    cv::Mat cvMat(eigenMat.rows(), eigenMat.cols(), dType);
    for(size_t row=0; row < 3; ++row)
    {
        for(size_t col=0; col < 3; ++col)
        {
            cvMat.at<float>(row,col) = eigenMat(row,col);
        }
    }
    return cvMat;
}

Eigen::Matrix3d cvMatToEigen3d(const cv::Mat& cvMat)
{
    Eigen::Matrix3d eigenMat;
    for(size_t row=0; row < 3; ++row)
    {
        for(size_t col=0; col < 3; ++col)
        {
            eigenMat(row, col) = cvMat.at<double>(row,col);
        }
    }
    return eigenMat;
}


pcl::PointCloud<pcl::PointXYZRGB>::Ptr landmarksToPclCloud(const std::vector<Eigen::Vector3d>& landmarks)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr landmark_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    for(const auto& landmark : landmarks)
    {
        pcl::PointXYZRGB pt(0,0,0);
        pt.g = 253;
        pt.x = landmark(0);
        pt.y = landmark(1);
        pt.z = landmark(2);
        landmark_cloud->points.push_back(pt);

    }
    return landmark_cloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr landmarksToPclCloud(const std::vector<Landmark>& landmarks,
                                                           int red, int green, int blue)
{
    // auto landmark_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    // it is better to use it like that, using default shared_ptr would conflict with other pcl functions
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr landmark_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    for(const auto& landmark : landmarks)
    {
        pcl::PointXYZRGB pt(0, 0, 0);
        pt.r = 253;
        pt.x = landmark.x;
        pt.y = landmark.y;
        pt.z = landmark.z;
        landmark_cloud->points.push_back(pt);
    }

    std::cout << "landmark_cloud->points[0]: " << landmark_cloud->points[0] << std::endl;
    std::cout << "landmark_cloud->points[1]: " << landmark_cloud->points[1] << std::endl;
    std::cout << "landmark_cloud->points[2]: " << landmark_cloud->points[2] << std::endl;
    std::cout << "landmark_cloud->points[3]: " << landmark_cloud->points[3] << std::endl;
    std::cout << "landmark_cloud->points[4]: " << landmark_cloud->points[4] << std::endl;
    std::cout << "landmark_cloud->points[5]: " << landmark_cloud->points[5] << std::endl;


    return landmark_cloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr vectorToPclCloud(const std::vector<Eigen::Vector3d>& vec,
                                                        int red, int green, int blue)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    for(const auto& eigenPt : vec)
    {
        pcl::PointXYZRGB pt(red, green, blue);
        pt.x = eigenPt(0);
        pt.y = eigenPt(1);
        pt.z = eigenPt(2);

        cloud->push_back(pt);
    }

    return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr cameraPosesToPclCloud(const std::unordered_map<int, Eigen::Matrix4d>& imgIdx2camPose,
                                                             int red, int green, int blue)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr camera_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    for(const auto& [imgIdx, camPose] : imgIdx2camPose)
    {
        Eigen::Vector3d translationVector = camPose.block<3, 1>(0, 3);

        pcl::PointXYZRGB pt(red, green, blue);
        pt.x = translationVector(0);
        pt.y = translationVector(1);
        pt.z = translationVector(2);
        camera_cloud->points.push_back(pt);
    }

    // std::cout << "camera_cloud->points[0]: " << camera_cloud->points[0] << std::endl;
    // std::cout << "camera_cloud->points[1]: " << camera_cloud->points[1] << std::endl;
    // std::cout << "camera_cloud->points[2]: " << camera_cloud->points[2] << std::endl;
    // std::cout << "camera_cloud->points[3]: " << camera_cloud->points[3] << std::endl;
    // std::cout << "camera_cloud->points[4]: " << camera_cloud->points[4] << std::endl;
    // std::cout << "camera_cloud->points[5]: " << camera_cloud->points[5] << std::endl;

    return camera_cloud;

}

using namespace std::chrono_literals;
void viewCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudLandmark,
               const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudCamera)
{
    // pcl::visualization::CloudViewer viewer("Cloud Viewer");
    // viewer.showCloud(cloudLandmark, "cloudLandmark");
    // viewer.showCloud(cloudCamera, "cloudCamera");


    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->addPointCloud<pcl::PointXYZRGB>(cloudLandmark, "cloud_landmark");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud_landmark");

    viewer->addPointCloud<pcl::PointXYZRGB>(cloudCamera, "cloud_camera");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "cloud_camera");

    viewer->setBackgroundColor(0.1, 0.1, 0.1);
    viewer->initCameraParameters();

    while (!viewer->wasStopped ())
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
    // pcl::visualization::CloudViewer viewer("Cloud Viewer");
    // viewer.showCloud(cloudLandmark, "cloudLandmark");
    // viewer.showCloud(cloudCamera, "cloudCamera");


    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->addPointCloud<pcl::PointXYZRGB>(cloudLandmark1, "cloud_landmark1");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud_landmark1");

    viewer->addPointCloud<pcl::PointXYZRGB>(cloudLandmark2, "cloud_landmark2");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud_landmark2");

    viewer->addPointCloud<pcl::PointXYZRGB>(cloudCamera1, "cloud_camera1");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "cloud_camera1");
    
    viewer->addPointCloud<pcl::PointXYZRGB>(cloudCamera2, "cloud_camera2");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "cloud_camera2");

    viewer->setBackgroundColor(0.1, 0.1, 0.1);
    viewer->initCameraParameters();

    while (!viewer->wasStopped ())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

    }
}

void writeInliersToVector(const cv::Mat& inliersCV,
                              std::vector<bool>& inliersVec)
    {
        // std::cout << "inliersVec.rows: " << inliersCV.rows << std::endl;
        // std::cout << "inliersVec.cols: " << inliersCV.cols << std::endl;
        // std::cout << "inliersCV.at<uchar>(matchIdx): " << static_cast<unsigned>(inliersCV.at<uchar>(0)) << std::endl;
        // std::cout << "inliersCV.at<uchar>(matchIdx): " << static_cast<unsigned>(inliersCV.at<uchar>(0)) << std::endl;
        for(size_t matchIdx = 0; matchIdx < inliersCV.rows; ++matchIdx)
        {
            inliersVec.push_back(inliersCV.at<uchar>(matchIdx));
            // std::cout << "inliersCV.at<uchar>(matchIdx): " << inliersCV.at<uchar>(matchIdx) << std::endl;
        }
    }

void saveCloud(std::vector<Landmark>& landmarks,
               std::unordered_map<int, Eigen::Matrix4d>& imgIdx2camPose,
               const std::string& cloudPath)
{
    auto posesCloud = cameraPosesToPclCloud(imgIdx2camPose);
    auto landmarksCloud =  landmarksToPclCloud(landmarks);

    *landmarksCloud += *posesCloud;

    pcl::io::savePCDFileASCII (cloudPath, *landmarksCloud);


}

}