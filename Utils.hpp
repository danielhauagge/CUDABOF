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


#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <CUDABOF.hpp>

// Stream operator
std::ostream &
operator<<(std::ostream &stream, const cudaDeviceProp &devProp);

std::ostream &
operator<<(std::ostream &stream, const CvSize &size);

// CvImage depth
template<typename T> inline int getCvMatDepth();

template<> inline int getCvMatDepth<int>()            { return IPL_DEPTH_32S; }
template<> inline int getCvMatDepth<char>()           { return IPL_DEPTH_8S;  }
template<> inline int getCvMatDepth<unsigned char>()  { return IPL_DEPTH_8U;  }
template<> inline int getCvMatDepth<float>()          { return CV_32FC1;}//IPL_DEPTH_32F; }
template<> inline int getCvMatDepth<double>()         { return IPL_DEPTH_64F; }

// More operators
bool operator==(const CvSize &a, const CvSize &b);

// Color
CvScalar hsv2rgb(float h, float s, float v);

void
drawBayesianOpticalFlow(cv::Mat &displayImage,
                        const cv::Mat &dx, const cv::Mat &dy,
                        const cv::Mat &Lambda_xx, const cv::Mat &Lambda_yy, const cv::Mat &Lambda_xy,
                        cv::Size gridSize,
                        float lineMulFactor, float ellipseMulFactor);
#endif
