cmake_minimum_required(VERSION 3.16)
project(Mapper)
set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_PREFIX_PATH "/home/ramil/SOFTWARE/libtorch")

message(STATUS ${PROJECT_SOURCE_DIR})

set(HEADERS
        libMapper/BasicFlags.h
        libMapper/utils.h
        libMapper/datatypes.h
        libMapper/FeatureDetector.h
        libMapper/FeatureSuperPoint.h
        libMapper/FeatureMatcher.h
        libMapper/FeatureMatcherSuperglue.h
        libMapper/SequentialReconstructor.h
        libMapper/ImageMatcher.h
        libMapper/GeometricFilter.h
)

set(SOURCES
        libMapper/utils.cpp        
        libMapper/FeatureDetector.cpp
        libMapper/FeatureSuperPoint.cpp 
        libMapper/FeatureMatcher.cpp 
        libMapper/FeatureMatcherSuperglue.cpp
        libMapper/SequentialReconstructor.cpp
        libMapper/ImageMatcher.cpp  
        libMapper/GeometricFilter.cpp    
)

find_package(OpenCV 4.2 REQUIRED)
find_package(Torch REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS}
                                      ${TORCH_LIBRARIES}
                                      Eigen3::Eigen)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}
                                                  ${OpenCV_INCLUDE_DIRS}
                                                  ${TORCH_INCLUDE_DIRS})


