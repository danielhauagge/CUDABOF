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


#include <GaussianPyramidKernel.hpp>

static texture<float, 2> _levelTexture;

#define KERNEL_SIZE 5
#define HALF_KERNEL 2
#define NORM_FACTOR 0.00390625 // 1.0/(16.0^2)

//
// Gaussian 5 x 5 kernel = [1, 4, 6, 4, 1]/16
//

__global__
void
gaussianPyramidDownsampleKernel(float *downLevel,
				size_t downLevelPitch,
				unsigned int downWidth, unsigned int downHeight)
{
   // calculate normalized texture coordinates
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  //if(x < downWidth and y < downHeight) {
    float buf[KERNEL_SIZE];

    float u0 = (2 * x) - HALF_KERNEL;
    float v0 = (2 * y) - HALF_KERNEL;

    for(int i = 0; i < KERNEL_SIZE; i++) {
      buf[i] =
	    (tex2D(_levelTexture, u0    , v0 + i) + tex2D(_levelTexture, u0 + 4, v0 + i)) +
	4 * (tex2D(_levelTexture, u0 + 1, v0 + i) + tex2D(_levelTexture, u0 + 3, v0 + i)) +
	6 *  tex2D(_levelTexture, u0 + 2, v0 + i);
    }

    downLevel[y * downLevelPitch + x] = (buf[0] + buf[4] + 4*(buf[1] + buf[3]) + 6 * buf[2]) * NORM_FACTOR;
    //}
}

void
gaussianPyramidDownsample(cudaArray *level,
			  float *downLevel,
			  size_t downLevelPitch,
			  unsigned int downWidth, unsigned int downHeight)
{
  dim3 grid(iDivUp(downWidth, CB_TILE_W), iDivUp(downHeight, CB_TILE_H));
  dim3 threads(CB_TILE_W, CB_TILE_H);

  // Bind the array to the texture
  _levelTexture.normalized = false;
  _levelTexture.filterMode = cudaFilterModePoint;
  _levelTexture.addressMode[0] = cudaAddressModeClamp;
  _levelTexture.addressMode[1] = cudaAddressModeClamp;
  CUDA_SAFE_CALL( cudaBindTextureToArray(_levelTexture, level) );
  gaussianPyramidDownsampleKernel<<< grid , threads >>>(downLevel, downLevelPitch/sizeof(float),
							downWidth, downHeight);
  CUDA_SAFE_CALL( cudaUnbindTexture(_levelTexture));

  CUT_CHECK_ERROR( __PRETTY_FUNCTION__ );
  CUDA_SAFE_CALL( cudaThreadSynchronize());
}
