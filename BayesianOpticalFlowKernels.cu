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


#include <BayesianOpticalFlowKernels.hpp>

// Global params
static texture<float, 2> _prevImageTexture;
static texture<float, 2> _currImageTexture;
static texture<float, 2> _nextImageTexture;
static texture<float, 2> _IdxTexture;  // spatial derivative dx
static texture<float, 2> _IdyTexture;  // spatial derivative dy
static texture<float, 2> _IdtTexture;  // temporal derivative
static texture<float, 2> _nfTexture;   // normalizing factor

static texture<float, 2> _MxxTexture;
static texture<float, 2> _MxyTexture;
static texture<float, 2> _MyyTexture;

static texture<float, 2> _bxTexture;
static texture<float, 2> _byTexture;

static texture<float, 2> _muxCoarseTexture;      // mux, note that mux != Mxx
static texture<float, 2> _muyCoarseTexture;
static texture<float, 2> _LambdaxxCoarseTexture;
static texture<float, 2> _LambdayyCoarseTexture;
static texture<float, 2> _LambdaxyCoarseTexture;


// Kernels for derivation
// TODO put reference here
#define KERNEL_SIZE      3
#define HALF_KERNEL_SIZE 1
__device__ __constant__ float smoothKernel[3] = {0.223755f, 0.552490f, 0.223755f};
__device__ __constant__ float diffKernel[3]  = {-0.453014f, 0.0f, 0.453014f};

// For debug only
#if __DEVICE_EMULATION__
#define PRINT_FLOAT_VAR(var) printf("l%03d: " #var " = %e\n", __LINE__, var)
#else
#define PRINT_FLOAT_VAR(var)
#endif

__device__
void
spatialDeriv(float2 addr, float2 &deriv,
	     unsigned int width, unsigned int height)
{
  float accumX[KERNEL_SIZE];
  float accumY[KERNEL_SIZE];

  float u0 = addr.x - HALF_KERNEL_SIZE;
  float v0 = addr.y - HALF_KERNEL_SIZE;

  for(int i = 0; i < KERNEL_SIZE; i++) {
    accumX[i] = 0;
    accumY[i] = 0;
    for(int j = 0; j < KERNEL_SIZE; j++) {
      accumX[i] += tex2D(_currImageTexture, u0 + i, v0 + j) * smoothKernel[j];
      accumY[i] += tex2D(_currImageTexture, u0 + i, v0 + j) * diffKernel[j];
    }
  }

  deriv.x = 0.0;
  deriv.y = 0.0;
  for(int i = 0; i < KERNEL_SIZE; i++) {
    deriv.x += diffKernel[i] * accumX[i];
    deriv.y += smoothKernel[i] * accumY[i];
  }
}

/*******************************************************************************/
__global__
void
spatialDerivKernel(float *Idx, float *Idy,
		   size_t pitch,
		   unsigned int width, unsigned int height)
{
  float2 deriv, addr;

  // calculate normalized texture coordinates
  addr.x = blockIdx.x * blockDim.x + threadIdx.x;
  addr.y = blockIdx.y * blockDim.y + threadIdx.y;

  //  if(addr.x < width and addr.y < height) {
    spatialDeriv(addr, deriv, width, height);

    unsigned int offset = addr.y * pitch + addr.x;
    Idx[offset] = deriv.x;
    Idy[offset] = deriv.y;
    //  }
}

void
spatialDeriv(cudaArray *Image,
	     float *Idx, float *Idy,
	     size_t pitch,
	     unsigned int width, unsigned int height)
{
  dim3 grid(iDivUp(width, CB_TILE_W), iDivUp(height, CB_TILE_H));
  dim3 threads(CB_TILE_W, CB_TILE_H);

  // Bind the array to the texture'
  _currImageTexture.normalized = false;
  _currImageTexture.filterMode = cudaFilterModePoint;
  _currImageTexture.addressMode[0] = cudaAddressModeClamp;
  _currImageTexture.addressMode[1] = cudaAddressModeClamp;

  CUDA_SAFE_CALL( cudaBindTextureToArray(_currImageTexture, Image) );
  spatialDerivKernel<<< grid , threads >>>(Idx, Idy,
					   pitch/sizeof(float), width, height);
  CUDA_SAFE_CALL( cudaUnbindTexture(_currImageTexture));

  CUT_CHECK_ERROR( __PRETTY_FUNCTION__ );
  CUDA_SAFE_CALL( cudaThreadSynchronize());
}


/*******************************************************************************/
__global__
void
bofBaseLevelKernel1(float *Idx, float *Idy, float *Idt, float *normalizingFactor,
		    size_t pitch, unsigned int width, unsigned int height,
		    float lambdaOne, float lambdaTwo)
{
  float2 deriv, addr;

  // thread addr
  addr.x = blockIdx.x * blockDim.x + threadIdx.x;
  addr.y = blockIdx.y * blockDim.y + threadIdx.y;

  //  if(addr.x < width and addr.y < height) {
    // Spatial derivs
    spatialDeriv(addr, deriv, width, height);

    unsigned int offset = addr.y * pitch + addr.x;
    Idx[offset] = deriv.x;
    Idy[offset] = deriv.y;

    // Temporal derivs
    Idt[offset] = diffKernel[0] * tex2D(_prevImageTexture, addr.x, addr.y)
      + diffKernel[1] * tex2D(_currImageTexture, addr.x, addr.y)
      + diffKernel[2] * tex2D(_nextImageTexture, addr.x, addr.y);

    // Normalizing factor
    normalizingFactor[offset] = lambdaOne * (deriv.x * deriv.x + deriv.y * deriv.y) + lambdaTwo;
    //  }
}

void
bofBaseLevel1(cudaArray *prevImage, cudaArray *currImage, cudaArray *nextImage,
	      float *Idx, float *Idy, float *Idt, float *normalizingFactor,
	      size_t pitch, unsigned int width, unsigned int height,
	      float lambdaOne, float lambdaTwo)
{
  dim3 grid(iDivUp(width, CB_TILE_W), iDivUp(height, CB_TILE_H));
  dim3 threads(CB_TILE_W, CB_TILE_H);

  // Setup texture parameters
  _prevImageTexture.normalized = false;
  _prevImageTexture.filterMode = cudaFilterModePoint;
  _prevImageTexture.addressMode[0] = cudaAddressModeClamp;
  _prevImageTexture.addressMode[1] = cudaAddressModeClamp;

  _currImageTexture.normalized = _prevImageTexture.normalized;
  _currImageTexture.filterMode = _prevImageTexture.filterMode;
  _currImageTexture.addressMode[0] = _prevImageTexture.addressMode[0];
  _currImageTexture.addressMode[1] = _prevImageTexture.addressMode[1];

  _nextImageTexture.normalized = _prevImageTexture.normalized;
  _nextImageTexture.filterMode = _prevImageTexture.filterMode;
  _nextImageTexture.addressMode[0] = _prevImageTexture.addressMode[0];
  _nextImageTexture.addressMode[1] = _prevImageTexture.addressMode[1];


  // Bind the array to the texture
  CUDA_SAFE_CALL( cudaBindTextureToArray(_prevImageTexture, prevImage) );
  CUDA_SAFE_CALL( cudaBindTextureToArray(_currImageTexture, currImage) );
  CUDA_SAFE_CALL( cudaBindTextureToArray(_nextImageTexture, nextImage) );


  // Run the kernel
  bofBaseLevelKernel1<<< grid , threads >>>(Idx, Idy, Idt, normalizingFactor,
					    pitch/sizeof(float), width, height,
					    lambdaOne, lambdaTwo);

  // Clean up
  CUDA_SAFE_CALL( cudaUnbindTexture(_prevImageTexture));
  CUDA_SAFE_CALL( cudaUnbindTexture(_currImageTexture));
  CUDA_SAFE_CALL( cudaUnbindTexture(_nextImageTexture));

  CUT_CHECK_ERROR( __PRETTY_FUNCTION__ );
  CUDA_SAFE_CALL( cudaThreadSynchronize());
}



/*******************************************************************************/
__global__
void
bofBaseLevelKernel2(float *Mxx, float *Myy, float *Mxy,
		    float *bx, float *by,
		    size_t pitch, unsigned int width, unsigned int height)
{
  // thread addr
  float2 addr;
  addr.x = blockIdx.x * blockDim.x + threadIdx.x;
  addr.y = blockIdx.y * blockDim.y + threadIdx.y;

  //if(addr.x < width and addr.y < height) {
    float Idx = tex2D(_IdxTexture, addr.x, addr.y);
    float Idy = tex2D(_IdyTexture, addr.x, addr.y);
    float Idt = tex2D(_IdtTexture, addr.x, addr.y);
    float nf  = tex2D(_nfTexture , addr.x, addr.y);

    unsigned int offset = addr.y * pitch + addr.x;
    Mxx[offset] = Idx * Idx/nf;
    Mxy[offset] = Idx * Idy/nf;
    Myy[offset] = Idy * Idy/nf;
    bx[offset]  = Idx * Idt/nf;
    by[offset]  = Idy * Idt/nf;
    //}
}

void
bofBaseLevel2(cudaArray *Idx, cudaArray *Idy, cudaArray *Idt, cudaArray *normalizingFactor,
	      float *Mxx, float *Myy, float *Mxy,
	      float *bx, float *by,
	      size_t pitch, unsigned int width, unsigned int height)
{
  dim3 grid(iDivUp(width, CB_TILE_W), iDivUp(height, CB_TILE_H));
  dim3 threads(CB_TILE_W, CB_TILE_H);

  // Setup texture parameters
  _IdxTexture.normalized = false;
  _IdxTexture.filterMode = cudaFilterModePoint;
  _IdxTexture.addressMode[0] = cudaAddressModeClamp;
  _IdxTexture.addressMode[1] = cudaAddressModeClamp;

  _IdyTexture.normalized     = _IdxTexture.normalized;
  _IdyTexture.filterMode     = _IdxTexture.filterMode;
  _IdyTexture.addressMode[0] = _IdxTexture.addressMode[0];
  _IdyTexture.addressMode[1] = _IdxTexture.addressMode[1];

  _IdtTexture.normalized     = _IdxTexture.normalized;
  _IdtTexture.filterMode     = _IdxTexture.filterMode;
  _IdtTexture.addressMode[0] = _IdxTexture.addressMode[0];
  _IdtTexture.addressMode[1] = _IdxTexture.addressMode[1];

  _nfTexture.normalized      = _IdxTexture.normalized;
  _nfTexture.filterMode      = _IdxTexture.filterMode;
  _nfTexture.addressMode[0]  = _IdxTexture.addressMode[0];
  _nfTexture.addressMode[1]  = _IdxTexture.addressMode[1];

  // Bind the array to the texture
  CUDA_SAFE_CALL( cudaBindTextureToArray(_IdxTexture, Idx) );
  CUDA_SAFE_CALL( cudaBindTextureToArray(_IdyTexture, Idy) );
  CUDA_SAFE_CALL( cudaBindTextureToArray(_IdtTexture, Idt) );
  CUDA_SAFE_CALL( cudaBindTextureToArray(_nfTexture, normalizingFactor) );

  // Run the kernel
  bofBaseLevelKernel2<<< grid , threads >>>(Mxx, Myy, Mxy,
					    bx, by,
					    pitch/sizeof(float), width, height);

  // Clean up
  CUDA_SAFE_CALL( cudaUnbindTexture(_IdxTexture));
  CUDA_SAFE_CALL( cudaUnbindTexture(_IdyTexture));
  CUDA_SAFE_CALL( cudaUnbindTexture(_IdtTexture));
  CUDA_SAFE_CALL( cudaUnbindTexture(_nfTexture));

  CUT_CHECK_ERROR( __PRETTY_FUNCTION__ );
  CUDA_SAFE_CALL( cudaThreadSynchronize());
}


/*******************************************************************************/
__global__
void
bofBaseLevelKernel3(float *Lambdaxx, float *Lambdayy, float *Lambdaxy,
		    float *mux, float *muy,
		    size_t pitch, unsigned int width, unsigned int height,
		    float invLambdaPrior, unsigned int windowSide)
{
  // thread addr
  float2 addr;
  addr.x = blockIdx.x * blockDim.x + threadIdx.x;
  addr.y = blockIdx.y * blockDim.y + threadIdx.y;

  //if(addr.x < width and addr.y < height) {
    unsigned int offset = addr.y * pitch + addr.x;

    unsigned int u0 = addr.x - windowSide/2;
    unsigned int v0 = addr.y - windowSide/2;

    float Mxx = 0, Myy = 0, Mxy = 0;
    float bx = 0, by = 0;
    unsigned int u = u0, v;
    for(int i = 0; i < windowSide; i++, u++) {
      v = v0;
      for(int j = 0; j < windowSide; j++, v++) {
	Mxx += tex2D(_MxxTexture, u, v);
	Myy += tex2D(_MyyTexture, u, v);
	Mxy += tex2D(_MxyTexture, u, v);
	bx  += tex2D(_bxTexture,  u, v);
	by  += tex2D(_byTexture,  u, v);
      }
    }

    int nPoints = windowSide * windowSide;
    Mxx /= nPoints;
    Mxy /= nPoints;
    Myy /= nPoints;
    bx  /= nPoints;
    by  /= nPoints;

    ////
    // Compute Lambda
    ////
    Mxx += invLambdaPrior;
    Myy += invLambdaPrior;
    float detM = Mxx * Myy - Mxy * Mxy;

    float Lxx =  Myy / detM;
    float Lyy =  Mxx / detM;
    float Lxy = -Mxy / detM;

    Lambdaxx[offset] = Lxx;
    Lambdayy[offset] = Lyy;
    Lambdaxy[offset] = Lxy;

    ////
    // Compute mu
    ////
    mux[offset] =  -(Lxx * bx + Lxy * by);
    muy[offset] =  -(Lxy * bx + Lyy * by);
    //}
}

void
bofBaseLevel3(cudaArray *Mxx, cudaArray *Myy, cudaArray *Mxy,
	      cudaArray *bx, cudaArray *by,
	      float *Lambdaxx, float *Lambdayy, float *Lambdaxy,
	      float *mux, float *muy,
	      size_t pitch, unsigned int width, unsigned int height,
	      float lambdaPrior, unsigned int windowSide)
{
  dim3 grid(iDivUp(width, CB_TILE_W), iDivUp(height, CB_TILE_H));
  dim3 threads(CB_TILE_W, CB_TILE_H);

  // Setup texture parameters
  _MxxTexture.normalized = false;
  _MxxTexture.filterMode = cudaFilterModePoint;
  _MxxTexture.addressMode[0] = cudaAddressModeClamp;
  _MxxTexture.addressMode[1] = cudaAddressModeClamp;

  _MyyTexture.normalized     = _MxxTexture.normalized;
  _MyyTexture.filterMode     = _MxxTexture.filterMode;
  _MyyTexture.addressMode[0] = _MxxTexture.addressMode[0];
  _MyyTexture.addressMode[1] = _MxxTexture.addressMode[1];

  _MxyTexture.normalized     = _MxxTexture.normalized;
  _MxyTexture.filterMode     = _MxxTexture.filterMode;
  _MxyTexture.addressMode[0] = _MxxTexture.addressMode[0];
  _MxyTexture.addressMode[1] = _MxxTexture.addressMode[1];

  _bxTexture.normalized     = _MxxTexture.normalized;
  _bxTexture.filterMode     = _MxxTexture.filterMode;
  _bxTexture.addressMode[0] = _MxxTexture.addressMode[0];
  _bxTexture.addressMode[1] = _MxxTexture.addressMode[1];

  _byTexture.normalized     = _MxxTexture.normalized;
  _byTexture.filterMode     = _MxxTexture.filterMode;
  _byTexture.addressMode[0] = _MxxTexture.addressMode[0];
  _byTexture.addressMode[1] = _MxxTexture.addressMode[1];

  // Bind the array to the texture
  CUDA_SAFE_CALL( cudaBindTextureToArray(_MxxTexture, Mxx) );
  CUDA_SAFE_CALL( cudaBindTextureToArray(_MyyTexture, Myy) );
  CUDA_SAFE_CALL( cudaBindTextureToArray(_MxyTexture, Mxy) );
  CUDA_SAFE_CALL( cudaBindTextureToArray(_bxTexture, bx) );
  CUDA_SAFE_CALL( cudaBindTextureToArray(_byTexture, by) );

  // Run the kernel
  bofBaseLevelKernel3<<< grid , threads >>>(Lambdaxx, Lambdayy, Lambdaxy,
					    mux, muy,
					    pitch/sizeof(float), width, height,
					    1.0/lambdaPrior, windowSide);

  // Clean up
  CUDA_SAFE_CALL( cudaUnbindTexture(_MxxTexture));
  CUDA_SAFE_CALL( cudaUnbindTexture(_MxyTexture));
  CUDA_SAFE_CALL( cudaUnbindTexture(_MyyTexture));
  CUDA_SAFE_CALL( cudaUnbindTexture(_bxTexture));
  CUDA_SAFE_CALL( cudaUnbindTexture(_byTexture));

  CUT_CHECK_ERROR( __PRETTY_FUNCTION__ );
  CUDA_SAFE_CALL( cudaThreadSynchronize());
}


/*******************************************************************************/
__global__
void
bofRefineEstimateKernel(float *Lambdaxx, float *Lambdayy, float *Lambdaxy,
			float *mux, float *muy,
			size_t pitch, unsigned int width, unsigned int height,
			float lambdaZero, float lambdaOne, float lambdaTwo)
{
  // Thread address
  float2 deriv, addr;
  addr.x = blockIdx.x * blockDim.x + threadIdx.x;
  addr.y = blockIdx.y * blockDim.y + threadIdx.y;

  //if(addr.x < width and addr.y < height) {
    unsigned int offset = addr.y * pitch + addr.x;
    float2 addrCoarse = make_float2(addr.x/2.0, addr.y/2.0);

    float2 mu;
    mu.x = 2.0 * tex2D(_muxCoarseTexture, addrCoarse.x, addrCoarse.y);
    mu.y = 2.0 * tex2D(_muyCoarseTexture, addrCoarse.x, addrCoarse.y);

    // Gradients
    spatialDeriv(addr, deriv, width, height);
    float dt = diffKernel[0] * tex2D(_prevImageTexture, addr.x - mu.x, addr.y - mu.y)
      + diffKernel[1] * tex2D(_currImageTexture, addr.x, addr.y)
      + diffKernel[2] * tex2D(_nextImageTexture, addr.x + mu.x, addr.y + mu.y);

    // Normalizing factor
    float nf = lambdaOne * (deriv.x * deriv.x + deriv.y * deriv.y) + lambdaTwo;

    // Lambda prime
    float LambdaPrimexx = 4.0 * tex2D(_LambdaxxCoarseTexture, addrCoarse.x, addrCoarse.y) + lambdaZero;
    float LambdaPrimeyy = 4.0 * tex2D(_LambdayyCoarseTexture, addrCoarse.x, addrCoarse.y) + lambdaZero;
    float LambdaPrimexy = 4.0 * tex2D(_LambdaxyCoarseTexture, addrCoarse.x, addrCoarse.y);

    // Invert lambda prime
    float det              =  LambdaPrimexx * LambdaPrimeyy - LambdaPrimexy * LambdaPrimexy;
    float InvLambdaPrimexx =  LambdaPrimeyy / det;
    float InvLambdaPrimeyy =  LambdaPrimexx / det;
    float InvLambdaPrimexy = -LambdaPrimexy / det;

    // Lambda
    float _InvLambdaxx = InvLambdaPrimexx + deriv.x * deriv.x/nf;
    float _InvLambdayy = InvLambdaPrimeyy + deriv.y * deriv.y/nf;
    float _InvLambdaxy = InvLambdaPrimexy + deriv.x * deriv.y/nf;

    // Invert Lambda
    det = _InvLambdaxx * _InvLambdayy - _InvLambdaxy * _InvLambdaxy;
    float _Lambdaxx =   _InvLambdayy / det;
    float _Lambdayy =   _InvLambdaxx / det;
    float _Lambdaxy = - _InvLambdaxy / det;

    Lambdaxx[offset] = _Lambdaxx;
    Lambdayy[offset] = _Lambdayy;
    Lambdaxy[offset] = _Lambdaxy;

    // b prime
    float2 bprime = make_float2(-deriv.x * dt / nf, -deriv.y * dt / nf);

    // mu
    mu.x += _Lambdaxx * bprime.x + _Lambdaxy * bprime.y;
    mu.y += _Lambdaxy * bprime.x + _Lambdayy * bprime.y;

    mux[offset] = mu.x;
    muy[offset] = mu.y;
    //}
}

void
bofRefineEstimate(cudaArray *prevImage, cudaArray *currImage, cudaArray *nextImage,
		  cudaArray *LambdaxxCoarse, cudaArray *LambdayyCoarse, cudaArray *LambdaxyCoarse,
		  cudaArray *muxCoarse, cudaArray *muyCoarse,
		  float *Lambdaxx, float *Lambdayy, float *Lambdaxy,
		  float *mux, float *muy,
		  size_t pitch, unsigned int width, unsigned int height,
		  float lambdaZero, float lambdaOne, float lambdaTwo)
{
  dim3 grid(iDivUp(width, CB_TILE_W), iDivUp(height, CB_TILE_H));
  dim3 threads(CB_TILE_W, CB_TILE_H);

  // Setup texture parameters
  _muxCoarseTexture.normalized = false;
  _muxCoarseTexture.filterMode = cudaFilterModeLinear;
  _muxCoarseTexture.addressMode[0] = cudaAddressModeClamp;
  _muxCoarseTexture.addressMode[1] = cudaAddressModeClamp;

  _muyCoarseTexture.normalized = _muxCoarseTexture.normalized;
  _muyCoarseTexture.filterMode = _muxCoarseTexture.filterMode;
  _muyCoarseTexture.addressMode[0] = _muxCoarseTexture.addressMode[0];
  _muyCoarseTexture.addressMode[1] = _muxCoarseTexture.addressMode[1];

  _LambdaxxCoarseTexture.normalized = _muxCoarseTexture.normalized;
  _LambdaxxCoarseTexture.filterMode = _muxCoarseTexture.filterMode;
  _LambdaxxCoarseTexture.addressMode[0] = _muxCoarseTexture.addressMode[0];
  _LambdaxxCoarseTexture.addressMode[1] = _muxCoarseTexture.addressMode[1];

  _LambdayyCoarseTexture.normalized = _muxCoarseTexture.normalized;
  _LambdayyCoarseTexture.filterMode = _muxCoarseTexture.filterMode;
  _LambdayyCoarseTexture.addressMode[0] = _muxCoarseTexture.addressMode[0];
  _LambdayyCoarseTexture.addressMode[1] = _muxCoarseTexture.addressMode[1];

  _LambdaxyCoarseTexture.normalized = _muxCoarseTexture.normalized;
  _LambdaxyCoarseTexture.filterMode = _muxCoarseTexture.filterMode;
  _LambdaxyCoarseTexture.addressMode[0] = _muxCoarseTexture.addressMode[0];
  _LambdaxyCoarseTexture.addressMode[1] = _muxCoarseTexture.addressMode[1];

  // prev, curr, and next images
  _prevImageTexture.normalized = _muxCoarseTexture.normalized;
  _prevImageTexture.filterMode = _muxCoarseTexture.filterMode;
  _prevImageTexture.addressMode[0] = _muxCoarseTexture.addressMode[0];
  _prevImageTexture.addressMode[1] = _muxCoarseTexture.addressMode[1];

  _currImageTexture.normalized = _muxCoarseTexture.normalized;
  _currImageTexture.filterMode = _muxCoarseTexture.filterMode;
  _currImageTexture.addressMode[0] = _muxCoarseTexture.addressMode[0];
  _currImageTexture.addressMode[1] = _muxCoarseTexture.addressMode[1];

  _nextImageTexture.normalized = _muxCoarseTexture.normalized;
  _nextImageTexture.filterMode = _muxCoarseTexture.filterMode;
  _nextImageTexture.addressMode[0] = _muxCoarseTexture.addressMode[0];
  _nextImageTexture.addressMode[1] = _muxCoarseTexture.addressMode[1];

   // Bind the array to the texture
  CUDA_SAFE_CALL( cudaBindTextureToArray(_muxCoarseTexture, muxCoarse) );
  CUDA_SAFE_CALL( cudaBindTextureToArray(_muyCoarseTexture, muyCoarse) );
  CUDA_SAFE_CALL( cudaBindTextureToArray(_LambdaxxCoarseTexture, LambdaxxCoarse) );
  CUDA_SAFE_CALL( cudaBindTextureToArray(_LambdayyCoarseTexture, LambdayyCoarse) );
  CUDA_SAFE_CALL( cudaBindTextureToArray(_LambdaxyCoarseTexture, LambdaxyCoarse) );

  CUDA_SAFE_CALL( cudaBindTextureToArray(_prevImageTexture, prevImage) );
  CUDA_SAFE_CALL( cudaBindTextureToArray(_currImageTexture, currImage) );
  CUDA_SAFE_CALL( cudaBindTextureToArray(_nextImageTexture, nextImage) );

  // Run the kernel
  bofRefineEstimateKernel<<< grid , threads >>>(Lambdaxx, Lambdayy, Lambdaxy,
						mux, muy,
						pitch/sizeof(float), width, height,
						lambdaZero, lambdaOne, lambdaTwo);

  // Clean up
  CUDA_SAFE_CALL( cudaUnbindTexture(_muxCoarseTexture));
  CUDA_SAFE_CALL( cudaUnbindTexture(_muyCoarseTexture));
  CUDA_SAFE_CALL( cudaUnbindTexture(_LambdaxxCoarseTexture));
  CUDA_SAFE_CALL( cudaUnbindTexture(_LambdayyCoarseTexture));
  CUDA_SAFE_CALL( cudaUnbindTexture(_LambdaxyCoarseTexture));

  CUT_CHECK_ERROR( __PRETTY_FUNCTION__ );
  CUDA_SAFE_CALL( cudaThreadSynchronize());
}
