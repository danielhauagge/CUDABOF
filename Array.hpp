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


#ifndef __ARRAY_HPP__
#define __ARRAY_HPP__

#include <CUDABOF.hpp>
#include <Utils.hpp>
#include <LinearMemory.hpp>

namespace CUDABOF
{
template<typename T>
class Array
{
public:

    Array();
    Array(cv::Size size);
    Array(const Array &otherArray);

    Array &operator=(const Array &otherArray);

    ~Array();

    void init(cv::Size size);
    void dealloc();

    // Getters
    cudaArray *getArray() { return _array; }
    const cudaArray *getArray() const { return _array; }
    cv::Size getSize() const { return _size; }


protected:
    cv::Size _size;
    cudaArray *_array;
    int *_refCount;
};


template<typename T>
Array<T>::Array(): _array(NULL), _refCount(NULL)
{
}

template<typename T>
Array<T>::Array(cv::Size size): _array(NULL), _refCount(NULL)
{
    init(size);
}

template<typename T>
Array<T>::Array(const Array<T> &otherArray):
    _array(NULL), _refCount(NULL)
{
    *this = otherArray;
}

template<typename T>
Array<T>::~Array()
{
    dealloc();
}

template<typename T>
void
Array<T>::init(cv::Size size)
{
    dealloc();

    _size = size;

    _refCount = new int;
    *_refCount = 1;

    // Allocate array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    CUDA_SAFE_CALL( cudaMallocArray( &_array, &channelDesc, size.width, size.height));
    assert(_array != NULL);
}

template<typename T>
void
Array<T>::dealloc()
{
    if(_array != NULL) {
        (*_refCount)--;
        if(*_refCount == 0) {
            CUDA_SAFE_CALL( cudaFreeArray(_array));
            _array = NULL;
            _size = cvSize(0, 0);
            delete _refCount;
            _refCount = NULL;
        }
    }
}

template<typename T>
Array<T> &
Array<T>::operator=(const Array<T> &otherArray)
{
    if(otherArray._array != _array) {

        dealloc();

        _array = otherArray._array;
        _refCount = otherArray._refCount;
        if(_refCount) (*_refCount)++;
    }
    return *this;
}
}

#endif
