The latest code can always be downloaded at
https://github.com/danielhauagge/CUDABOF

AUTHORS
----------------------------------------------------------------------
Daniel Hauagge <daniel.hauagge@gmail.com>


REFERENCES
----------------------------------------------------------------------
This is the implementation of the algorithm presented in

    Bayesian Multi-scale Differential Optical Flow
    E P Simoncelli, Handbook of Computer Vision and Applications,
    pages 397--422. Academic Press, Apr 1999.

variable naming follows closelly that used in the article. The
derivative kernels were extracted from

    Optimally Rotation-Equivariant Directional Derivative Kernels
    H Farid and E P Simoncelli, Int'l Conf Computer Analysis of
    Images and Patterns, pp. 207--214, Sep 1997.


REQUIREMENTS
----------------------------------------------------------------------
CUDA 1.1 (http://www.nvidia.com/object/cuda_what_is.html)
OpenCV (http://sourceforge.net/projects/opencvlibrary/)


COMPILING
----------------------------------------------------------------------
To compile this project edit the  first lines of the makefile with the
correct  paths to  CUDA resources. After that just type

make

I only tested  this on Linux and beleive that to compile on Windows a
fair amount of editing will have to be done.
