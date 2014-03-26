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


#include <test01_kernel.hpp>

static texture<float, 2, cudaReadModeElementType> _tex;

__global__
void
addOneKernel(float *out, int width, int height, unsigned int outPitch)
{
  // calculate normalized texture coordinates
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x < width and y < height) {
    float u = x;
    float v = y;

    out[y * outPitch + x] = tex2D(_tex, u, v) + 1;
  }
}

void
runAddOneKernel(float *out, int width, int height, unsigned int outPitch,
		cudaArray *array)
{
  dim3 grid(iDivUp(width, CB_TILE_W), iDivUp(height, CB_TILE_H));//CB_BLOCK_W, CB_BLOCK_H);
  dim3 threads(CB_TILE_W, CB_TILE_H);

  _tex.normalized = 0; // access the texture with image coords

  // Bind the array to the texture'
  CUDA_SAFE_CALL( cudaBindTextureToArray( _tex, array ) );
  CUT_CHECK_ERROR( __PRETTY_FUNCTION__ );
  addOneKernel<<< grid , threads >>>(out, width, height, outPitch/sizeof(float));
  CUDA_SAFE_CALL( cudaUnbindTexture(_tex));

  CUT_CHECK_ERROR( __PRETTY_FUNCTION__ );
  CUDA_SAFE_CALL( cudaThreadSynchronize());
}
