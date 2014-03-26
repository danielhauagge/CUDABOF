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


#ifndef __BAYESIAN_OPTICAL_FLOW_KERNELS_HPP__
#define __BAYESIAN_OPTICAL_FLOW_KERNELS_HPP__

#include <CUDABOF.hpp>
#include <MathUtils.hpp>

extern "C"
void
spatialDeriv(cudaArray *Image,
             float *Idx, float *Idy,
             size_t pitch, unsigned int width, unsigned int height);

extern "C"
void
bofBaseLevel1(cudaArray *prevImage, cudaArray *currImage, cudaArray *nextImage,
              float *Idx, float *Idy, float *Idt, float *normalizingFactor,
              size_t pitch, unsigned int width, unsigned int height,
              float lambdaOne, float lambdaTwo);

extern "C"
void
bofBaseLevel2(cudaArray *Idx, cudaArray *Idy, cudaArray *Idt, cudaArray *normalizingFactor,
              float *Mxx, float *Myy, float *Mxy,
              float *bx, float *by,
              size_t pitch, unsigned int width, unsigned int height);

extern "C"
void
bofBaseLevel3(cudaArray *Mxx, cudaArray *Myy, cudaArray *Mxy,
              cudaArray *bx, cudaArray *by,
              float *Lambdaxx, float *Lambdayy, float *Lambdaxy,
              float *mux, float *muy,
              size_t pitch, unsigned int width, unsigned int height,
              float lambdaPrior, unsigned int windowSide);

extern "C"
void
bofRefineEstimate(cudaArray *prevImage, cudaArray *currImage, cudaArray *nextImage,
                  cudaArray *LambdaxxCoarse, cudaArray *LambdayyCoarse, cudaArray *LambdaxyCoarse,
                  cudaArray *muxCoarse, cudaArray *muyCoarse,
                  float *Lambdaxx, float *Lambdayy, float *Lambdaxy,
                  float *mux, float *muy,
                  size_t pitch, unsigned int width, unsigned int height,
                  float lambdaZero, float lambdaOne, float lambdaTwo);

#endif
