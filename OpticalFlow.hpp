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


#ifndef __OPTICAL_FLOW_HPP__
#define __OPTICAL_FLOW_HPP__

#include <CUDABOF.hpp>
#include <Utils.hpp>
#include <LinearMemory.hpp>
#include <Array.hpp>
#include <MemoryUtils.hpp>
#include <Pyramid.hpp>
#include <GaussianPyramidKernel.hpp>
#include <BayesianOpticalFlowKernels.hpp>

namespace CUDABOF 
{
  
  class OpticalFlow
  {
  public:
    OpticalFlow();
    OpticalFlow(const cv::Mat &prevImage, const cv::Mat &currImage,
		int nLevels, int windowSide,
		float lambdaZero, float lambdaOne, float lambdaTwo, float lambdaPrior);
    
    void init(const cv::Mat &prevImage, const cv::Mat &currImage,
	      int nLevels, int windowSide,
	      float lambdaZero, float lambdaOne, float lambdaTwo, float lambdaPrior);
    
    void operator()(const cv::Mat &nextImage);

    void operator()(const cv::Mat &nextImage, 
		    cv::Mat &mu_x, cv::Mat &mu_y,
		    cv::Mat &Lambda_xx, cv::Mat &Lambda_yy, cv::Mat &Lambda_xy);

    void setLambdaZero(float lambdaZero);
    void setLambdaOne(float lambdaOne);
    void setLambdaTwo(float lambdaTwo);
    void setLambdaPrior(float lambdaPrior);
    
    float getLambdaZero() const;
    float getLambdaOne() const;
    float getLambdaTwo() const;
    float getLambdaPrior() const;
    int getNLevels() const;

  protected:

    float _lambdaZero, _lambdaOne, _lambdaTwo, _lambdaPrior;
    Pyramid< Array<float> > _gaussPyr[3];
    //Pyramid< Array<float> > _dxPyr[3];
    Pyramid< Array<float> > _Mxx, _Myy, _Mxy;
    Pyramid< Array<float> > _bx, _by;
    Pyramid< Array<float> > _Idt, _Idy, _Idx;
    Pyramid< Array<float> > _Lambdaxx, _Lambdayy, _Lambdaxy;
    Pyramid< Array<float> > _mux, _muy;
    
    Array<float> _normalizingFactor;

    static const int N_AUX = 5;
    Pyramid< LinearMemory<float> > _aux[N_AUX];
    
    int _nLevels, _windowSide;

    int _nextIndex, _currIndex, _prevIndex;
    void _incrementIndex();
    
    void _computeFlow(const cv::Mat &nextImage);
    void _computeBaseLevelFlow();
    void _computeFlowForLevel(int level);
  };
  
} // namespace CUDABOF

#endif
