
# reconstructor

This project is aimed for 3d reconstruction(sparse up-to scale cloud + camera positions) using camera images

# reconstructor features

At the moment there is cpp a wrapper over SuperPoint(https://github.com/magicleap/SuperPointPretrainedNetwork), 
also a cpp wrapper over SuperGlue matcher(https://github.com/magicleap/SuperGluePretrainedNetwork)

# what is done at the moment

1. 2 feature extractors: SuperPoint, ORB + BaseClass for possible extension
2. 2 feature matchers: SuperGlue, Flann + BaseClass for possible extension
3. Feature filtering(based on epipolar geometry)
4. Initial img pair choice and initial triangulation
5. Separate class for end2end reconstruction
6. Img matcher(which for now assumes all images are matched)
7. 3d visualization of reconstructed cloud(up to scale) + camera positions
8. incremental reconstruction via pnp + BA

# todo:

1. image matcher(apply some image retrieval, FAISS, metric learning)
2. use openmp/tbb to make it faster
3. code refactoring + add comments
4. make colored pointcloud + save in meshlab format


# todo later:

1. use monocular depth estimating networks(for example https://github.com/isl-org/MiDaS) to obtain real scale

# how to run

1. install OpenCV 4.2+  https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html
2. install libTorch https://pytorch.org/, choose Language: C++/Java, only CPU platform tested at the moment
3. install PCL https://pointclouds.org/downloads/
4. clone this repo and change path to libtorch in CMakeLists.txt "CMAKE_PREFIX_PATH"
5. `mkdir build && cd build`
6. `cmake .. && make`
7. run `./reconstruct`


