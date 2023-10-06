# Setting Up Environment:

In order to run the benchmark, you need
1. The AGAR benchmark image library. You can download it [from their site](https://paperswithcode.com/dataset/agar), or you can use the demo images in the zip file.
2. [OpenCV](https://github.com/opencv/opencv.git). It is easiest to just clone it into this directory with > "git clone https://github.com/opencv/opencv.git". If running the C++ benchmark, follow [these directions](https://www.geeksforgeeks.org/how-to-install-opencv-in-c-on-linux/#) to install opencv locally, for example on WSL2. It takes a while, wish there was a way to only use parts of it

Following that, you may need to recreate the Makefile for your machine, as there are some user-specific paths that I have no interest in fixing. That should be possible with ```cmake .```. For every subsequent compile run ```make```, then you can execute with ```./benchmark```. The default image is water_coins.jpg. To use another image, run ```./benchmark [image name]```. The image results will be displayed in new windows (if supported), and the execution times will be printed.
