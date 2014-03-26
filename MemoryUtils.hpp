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


#ifndef __MEMORY_UTILS_HPP__
#define __MEMORY_UTILS_HPP__

#include <CUDABOF.hpp>
#include <Utils.hpp>
#include <Array.hpp>
#include <LinearMemory.hpp>

namespace CUDABOF
{

template<typename T>
void memoryCopy(Array<T> &dst, const LinearMemory<T> &src);

template<typename T>
void memoryCopy(Array<T> &dst, const cv::Mat &src);

template<typename T>
void memoryCopy(LinearMemory<T> &dst, const cv::Mat &src);

template<typename T>
void memoryCopy(cv::Mat &dst, const Array<T> &src);

template<typename T>
void memoryCopy(cv::Mat &dst, const LinearMemory<T> &src);

/************************
 * Implementation
 ************************/

template<typename T>
void memoryCopy(Array<T> &dst, const LinearMemory<T> &src)
{
    assert(dst.getSize() == src.getSize());
    assert(dst.getArray() != NULL);
    assert(src.getPtr() != NULL);

    CUDA_SAFE_CALL( cudaMemcpy2DToArray(dst.getArray(), 0, 0,
                                        src.getPtr(), src.getPitch(),
                                        src.getSize().width * sizeof(T), src.getSize().height,
                                        cudaMemcpyDeviceToDevice) );
}

template<typename T>
void memoryCopy(Array<T> &dst, const cv::Mat &src)
{
    assert(src.size() == dst.getSize());
    assert(src.depth() == getCvMatDepth<T>());
    assert(src.channels() == 1);

    CUDA_SAFE_CALL( cudaMemcpy2DToArray(dst.getArray(), 0, 0,
                                        src.data, src.step, // FIXME step can vary (thats why it is a vector)
                                        dst.getSize().width * sizeof(T), dst.getSize().height,
                                        cudaMemcpyHostToDevice) );
}

template<typename T>
void memoryCopy(LinearMemory<T> &dst, const cv::Mat &src)
{
    assert(dst.getPtr() != NULL);
    assert(src.size() == dst.getSize());
    assert(src.depth() == getCvMatDepth<T>());
    assert(src.channels() == 1);

    CUDA_SAFE_CALL( cudaMemcpy2D(dst.getPtr(), dst.getPitch(),
                                 src.data, src.step, // FIXME step can vary (thats why it is a vector)
                                 dst.getSize().width * sizeof(T), dst.getSize().height,
                                 cudaMemcpyHostToDevice) );
}

template<typename T>
void memoryCopy(cv::Mat &dst, const Array<T> &src)
{
    assert(dst.data != NULL);
    assert(src.getArray() != NULL);
    assert(dst.size() == src.getSize());
    assert(dst.depth() == getCvMatDepth<T>());
    assert(dst.channels() == 1);

    CUDA_SAFE_CALL( cudaMemcpy2DFromArray(dst.data, dst.step, // FIXME step can vary (thats why it is a vector)
                                          src.getArray(), 0, 0,
                                          src.getSize().width * sizeof(T), src.getSize().height,
                                          cudaMemcpyDeviceToHost) );
}

template<typename T>
void memoryCopy(cv::Mat &dst, const LinearMemory<T> &src)
{
    assert(dst.data != NULL);
    assert(src.getPtr() != NULL);
    assert(dst.size() == src.getSize());
    assert(dst.depth() == getCvMatDepth<T>());
    assert(dst.channels() == 1);

    CUDA_SAFE_CALL( cudaMemcpy2D(dst.data, dst.step, // FIXME step can vary (thats why it is a vector)
                                 src.getPtr(), src.getPitch(),
                                 src.getSize().width * sizeof(T), src.getSize().height,
                                 cudaMemcpyDeviceToHost) );
    CUT_CHECK_ERROR(__PRETTY_FUNCTION__);// " failed");
    CUDA_SAFE_CALL( cudaThreadSynchronize());
}

}
#endif
