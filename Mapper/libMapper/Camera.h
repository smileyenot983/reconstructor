#pragma once

#include "datatypes.h"
#include <Eigen/Dense>

/*
simplest model of camera:
1. project on z = 1: x = X/Z, y = Y/Z
2. apply distortion(radial): x = x + (k1 * r + k2 * r^2), r = x^2 + y^2 and same for y coord
3. get pixel coords: u = f_x * x + c_x
*/
class PinholeCamera
{
public:
    PinholeCamera() {}

    // initialization from known focal length(in px)
    PinholeCamera(int height,
                  int width,
                  double fX,
                  double fY)
        : fX(fX), fY(fY)
    {
        cX = width / 2;
        cY = height / 2;
        k1 = k2 = 0;
    }

    // initialization from unknown focal length
    PinholeCamera(int height,
                  int width)
    {
        cX = width / 2;
        cY = height / 2;
        k1 = k2 = 0;

        double diag35mm = 36.0 * 36.0 + 24.0 * 24.0;
        double diagPx = width * width + height * height;

        fX = 50.0 * sqrt(diagPx / diag35mm);
        fY = fX;
    }

    // initialization as in colmap
    PinholeCamera(int height,
                  int width,
                  double focalLengthFactor)
    {
        cX = width / 2;
        cY = height / 2;
        k1 = k2 = 0;

        fY = fX = focalLengthFactor * std::max(height, width);
    }

    /*
    projects 3d point on 2d image plane
    */
    Eigen::Vector2d project(Eigen::Vector3d coord3d)
    {
        double x = coord3d(0) / coord3d(2);
        double y = coord3d(1) / coord3d(2);

        double radius = x * x + y * y;
        double distortion = k1 * radius + k2 * radius * radius;

        x += distortion;
        y += distortion;

        double u = fX * x + cX;
        double v = fY * y + cY;

        Eigen::Vector2d coord2d(u, v);

        return coord2d;
    }

    // projects back to 3d z = 1
    Eigen::Vector3d unproject(Eigen::Vector2d coord2d)
    {
        double x = (coord2d(0) - cX) / fX;
        double y = (coord2d(1) - cY) / fY;

        double radius = x * x + y * y;
        double distortion = k1 * radius + k2 * radius * radius;

        x -= distortion;
        y -= distortion;

        Eigen::Vector3d coord3d(x, y, 1);

        return coord3d;
    }

    // returns K as cv Matrix
    cv::Mat getMatrixCV() const
    {
        cv::Mat intrinsicsCV = cv::Mat::zeros(3, 3, CV_64F);
        intrinsicsCV.at<double>(0, 0) = fX;
        intrinsicsCV.at<double>(0, 2) = cX;
        intrinsicsCV.at<double>(1, 1) = fY;
        intrinsicsCV.at<double>(1, 2) = cY;
        intrinsicsCV.at<double>(2, 2) = 1.0;

        return intrinsicsCV;
    }

    cv::Mat getDistortCV() const
    {
        cv::Mat distCoeffsCV(4, 1, CV_64F);
        distCoeffsCV.at<double>(0, 0) = k1;
        distCoeffsCV.at<double>(1, 0) = k2;
        distCoeffsCV.at<double>(2, 0) = 0.0;
        distCoeffsCV.at<double>(3, 0) = 0.0;

        return distCoeffsCV;
    }

    double fX, fY, cX, cY, k1, k2;
};