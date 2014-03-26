# Copyright (c) 2008 Daniel Cabrini Hauagge
# 
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
# 
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.


###
# Configuration
###

NVIDIA_INSTAL_DIR=/Developer/GPU\ Computing/C/
CUDA_BASE=/usr/local/cuda

###
# You shouldn't have to change down from here
###

INCLUDE:= -I.                                        \
          -I/usr/local/cuda/include                  \
	  -I$(NVIDIA_INSTAL_DIR)/common/inc/         \
          -I$(CUDA_BASE)/include

CXXFLAGS= -fno-strict-aliasing -DUNIX -O0 -ggdb $(shell pkg-config --cflags opencv) -Wall -arch i386 #x86_64

LDFLAGS= -L$(NVIDIA_INSTAL_DIR)/lib                        \
         -L$(NVIDIA_INSTAL_DIR)/common/lib                 \
         -L$(CUDA_BASE)/lib                             \
         -lcudart -framework OpenGL -framework GLUT \
         $(shell pkg-config --libs opencv) -lcutil_i386 

# -lGLEW_x86 
#-lGL -lGLU -lglut  \

#NVCCFLAGS = --compile --host-compilation 'C++' \
#            --compiler-options -fno-strict-aliasing \
#           --compiler-options  -ggdb $(INCLUDE)

#-gencode=arch=compute_10,code=\"sm_10,compute_10\"  -gencode=arch=compute_20,code=\"sm_20,compute_20\" 
#--host-compilation 'C++' 
NVCCFLAGS = --compile \
	    --compiler-options -fno-strict-aliasing $(INCLUDE) -DUNIX -O2 


#/usr/local/cuda/bin/nvcc      
#-I. -I/usr/local/cuda/include -I../../common/inc -I../../../shared//inc 


EXEC = test01                    \
       test02_spatialDerivs      \
       test03_gaussianPyramid    \
       test04_opticalFlow

all: $(EXEC)

VERBOSITY:=
ECHO_COMPILING=@echo "==== Compiling $@ ===	"


######
#
# COMPILATION OF OBJECT FILES
#
######

%.o: %.cu %.hpp CUDABOF.hpp
	$(ECHO_COMPILING)
	$(VERBOSITY)nvcc $(NVCCFLAGS) $<

%.o: %.cpp *.hpp
	$(ECHO_COMPILING)
	$(VERBOSITY)g++ -c $< $(INCLUDE) $(CXXFLAGS)


######
#
# TESTS
#
######

TEST_BUILD_RULE= $(VERBOSITY)g++ $(CXXFLAGS) -fPIC -o $@ $^ $(LDFLAGS)

test01: test01_kernel.o test01.o Utils.o MathUtils.o
	$(ECHO_COMPILING)
	$(TEST_BUILD_RULE)

test02_spatialDerivs: test02_spatialDerivs.o Utils.o MathUtils.o BayesianOpticalFlowKernels.o
	$(ECHO_COMPILING)
	$(TEST_BUILD_RULE)

test03_gaussianPyramid: test03_gaussianPyramid.o Utils.o MathUtils.o GaussianPyramidKernel.o
	$(ECHO_COMPILING)
	$(TEST_BUILD_RULE)

test04_opticalFlow: test04_opticalFlow.o Utils.o MathUtils.o OpticalFlow.o BayesianOpticalFlowKernels.o GaussianPyramidKernel.o
	$(ECHO_COMPILING)
	$(TEST_BUILD_RULE)

######
#
# TESTS
#
######
clean:
	rm -f *.o $(EXEC)
