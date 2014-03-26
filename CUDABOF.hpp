// Copyright (c) 2008 Daniel Cabrini Hauagge
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.


#ifndef __CUDABOF_HPP__
#define  __CUDABOF_HPP__

// OpenCV
#ifndef __CUDACC__
#include <opencv/cv.h>
#include <opencv/highgui.h>
#endif

// STD
#include <iomanip>
#include <iostream>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cutil.h>
//#include <cutil_inline.h>
//#include <cutil_gl_inline.h>
//#include <cutil_gl_error.h>
//#include <cuda_gl_interop.h>

// Macros
#define CB_PRINT_LINE std::cerr << __FILE__ << ":" << __LINE__ << std::endl
#define CB_PRINT_VAR(var) std::cerr << __FILE__ << ":" << __LINE__ << " " << #var << " = " << var << std::endl

#define CB_TILE_W  16
#define CB_TILE_H  16

#define CB_ESC 1048603

#endif
