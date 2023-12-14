
# reconstructor

This project is aimed for 3d reconstruction using camera images

# reconstructor features

At the moment there is cpp a wrapper over SuperPoint net(https://github.com/magicleap/SuperPointPretrainedNetwork), 

# what is done at the moment

1. 2 feature extractors: SuperPoint, ORB
2. 2 feature matchers: SuperGlue, Flann
3. Separate class for end2end reconstruction

# todo:

1. image matcher
2. feature filtering(epipolar geometry)
3. initialization(epipolar geometry) and triangulation
4. iterative pnp + optional BA
5. global BA


# how to run

1. install OpenCV 4.2+  https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html
2. install libTorch https://pytorch.org/, choose Language: C++/Java, only CPU platform tested at the moment
3. clone this repo and change path to libtorch in CMakeLists.txt "CMAKE_PREFIX_PATH"
4. `mkdir build && cd build`
5. `cmake .. && make`
6. run `./reconstruct`


