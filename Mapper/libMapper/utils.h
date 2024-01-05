#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "FeatureDetector.h"
#include "datatypes.h"



using namespace reconstructor::Core;

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
/*
Reshapes input image in place, making sure that resulting
image sides are divisible by 8
*/
void reshapeImg(cv::Mat &img,
                 const int imgMaxSize);

/*
Reads image in rgb format
*/
cv::Mat readImg(const std::string& imgPath,
                const int imgMaxSize);

/*
Reads image in grayscale and reshapes if needed
*/
cv::Mat readGrayImg(const std::string& imgPath,
                        const int imgMaxSize);

/*
Normalizes feature coordinates, to be in [-coordRange, coordRange]
*/
std::vector<FeaturePtr<double>> normalizeFeatCoords(const std::vector<FeaturePtr<>>& features,
                                                    const int imgHeight,
                                                    const int imgWidth,
                                                    const double coordRange = 0.7);


// calculates f_x in px coordinates
double focalLengthmmToPx(const double focalLengthmm,
                         const double imgDim,
                         const double fovDegrees);

Eigen::Matrix3d getIntrinsicsMat(const double focalLengthmm,
                                 const int imgHeight,
                                 const int imgWidth,
                                 const double fovDegrees);

Eigen::Matrix3d getIntrinsicsMat(const double focalLengthPx,
                                 const int imgHeight,
                                 const int imgWidth);


std::vector<cv::Point2f> featuresToCvPoints(const std::vector<FeaturePtr<>>& features);

cv::Mat eigen3dToCVMat(const Eigen::Matrix3d& eigenMat, int dType = CV_32F);

Eigen::Matrix3d cvMatToEigen3d(const cv::Mat& cvMat);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr landmarksToPclCloud(const std::vector<Eigen::Vector3d>& landmarks);

// creates cloud from landmarks
pcl::PointCloud<pcl::PointXYZRGB>::Ptr landmarksToPclCloud(const std::vector<Landmark>& landmarks);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr cameraPosesToPclCloud(const std::unordered_map<int, Eigen::Matrix4d>& imgIdx2camPose,
                                                              int red = 253, int green = 0, int blue = 0);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr vectorToPclCloud(const std::vector<Eigen::Vector3d>& vec,
                                                        int red, int green, int blue);

// visualizes pointcloud
void viewCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudLandmark,
               const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudCamera);

void viewCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudLandmark1,
                const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudLandmark2,
               const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudCamera1, 
               const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudCamera2);

void writeInliersToVector(const cv::Mat& inliersCV,
                              std::vector<bool>& inliersVec);


void saveCloud(std::vector<Landmark>& landmarks,
               std::unordered_map<int, Eigen::Matrix4d>& imgIdx2camPose,
               const std::string& cloudPath);

}