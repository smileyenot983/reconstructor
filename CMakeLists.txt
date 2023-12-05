cmake_minimum_required(VERSION 3.16)
project(reconstructor)
set(CMAKE_CXX_STANDARD 17)
add_subdirectory(Mapper)

# change to a path where your libtorch is installed
list(APPEND CMAKE_PREFIX_PATH "/home/ramil/SOFTWARE/libtorch")

find_package(OpenCV 4.2 REQUIRED)
find_package(Torch REQUIRED)

add_executable(reconstruct src/reconstruct.cpp)
target_link_libraries(reconstruct ${OpenCV_LIBS} 
                                  ${TORCH_LIBRARIES}
                                  Mapper)
target_include_directories(reconstruct PUBLIC include
                                    ${OpenCV_INCLUDE_DIRS}
                                    ${TORCH_INCLUDE_DIRS})

