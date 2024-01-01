#pragma once

#include <vector>
#include <cmath>
#include <memory>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>


namespace reconstructor::Core
{
    template <typename coordType = int>
    struct FeatCoord
    {
        FeatCoord() {}

        FeatCoord(coordType x, coordType y)
            : x(x), y(y)
        {}
        virtual ~FeatCoord(){}

        coordType x;
        coordType y;
    };

        /*
    Adds feature confidence in addition to coords
    */
    template <typename coordType = int>
    struct FeatCoordConf : FeatCoord<coordType>
    {
        FeatCoordConf() 
            : FeatCoord<coordType>()
        {}

        FeatCoordConf(const coordType x, const coordType y, const double conf)
            : FeatCoord<coordType>(x, y), conf(conf)
        {}

        double conf = 0.01;
    };

    // TODO:
    // check how initialization using constructor is done;
    struct FeatDesc
    {
        FeatDesc() {}

        // the only way to call vector constructor using iterators
        template <typename InputIterator>
        FeatDesc(InputIterator first, InputIterator last)
            : desc(first, last)
        {}

        // calculates l2 norm of descriptor
        double norm()
        {
            double sum = 0.0;
            for(const auto& descElem : desc)
            {
                sum += descElem*descElem;
            }
            return sqrt(sum);
        }

        std::vector<float> desc;
        int type = CV_32F;
    };

    /*
    Struct for storing feature with coords and descriptor
    */
    template <typename coordType = int>
    struct Feature
    {
        Feature(FeatCoord<coordType> featCoord, FeatDesc featDesc)
            : featCoord(featCoord), featDesc(featDesc)
        {}
        Feature() {}
        virtual ~Feature() {}

        FeatCoord<coordType> featCoord;
        FeatDesc featDesc;

        // id of the landmark, corresponding to feature
        int landmarkId = -1;
    };

    template <typename coordType = int>
    using FeaturePtr = std::shared_ptr<Feature<coordType>>;


    /*
    struct for storing feature coords, descriptor and confidence
    */
    template <typename coordType = int>
    struct FeatureConf : Feature<coordType>
    {
        FeatureConf(FeatCoord<coordType> featCoord, FeatDesc featDesc, double conf)
        : Feature<coordType>(featCoord, featDesc)
        , conf(conf) 
        {}

        FeatureConf(FeatCoordConf<coordType> featCoordConf, FeatDesc featDesc)
        : Feature<coordType>(static_cast<FeatCoord<>>(featCoordConf), featDesc)
        , conf(featCoordConf.conf)
        {}

        FeatureConf(FeatCoord<coordType> featCoord, FeatDesc featDesc)
        : Feature<coordType>(featCoord, featDesc)
        {}

        FeatureConf(){}

        double conf;
    };

    template <typename coordType = int>
    using FeatureConfPtr = std::shared_ptr<FeatureConf<coordType>>;


    struct TriangulatedFeature
    {
        TriangulatedFeature(){}
        TriangulatedFeature(const int imgIdx, const int featIdx)
        : imgIdx(imgIdx)
        , featIdx(featIdx)
        {}
        int imgIdx;
        int featIdx;
    };

    /*
    Landmarks represent triangulated 2d features coordinates in 3d
    */
    struct Landmark
    {
        Landmark(){}

        Landmark(const double x,
                 const double y,
                 const double z)
        : x(x)
        , y(y)
        , z(z)
        {}
        std::vector<TriangulatedFeature> triangulatedFeatures; 
        double x;
        double y;
        double z;

        bool initialLandmark = false;
    };


    struct Match
    {
    public:
        Match(size_t idx1, size_t idx2)
            : idx1(idx1), idx2(idx2)
        {
        }
        size_t idx1;
        size_t idx2;
    };
}