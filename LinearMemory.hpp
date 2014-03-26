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


#ifndef __LINEAR_MEMORY_HPP__
#define __LINEAR_MEMORY_HPP__

#include <CUDABOF.hpp>
#include <Utils.hpp>

namespace CUDABOF
{
template<typename T>
class LinearMemory
{
public:
    LinearMemory();
    LinearMemory(CvSize size);
    LinearMemory(const LinearMemory &otherLM);
    ~LinearMemory();

    void init(CvSize size);
    void dealloc();

    LinearMemory &operator=(const LinearMemory &otherLM);

    // Getter methods
    T *getPtr() { return _ptr; }
    const T *getPtr() const { return _ptr; }
    size_t getPitch() const { return _pitch; }
    CvSize getSize() const { return _size; }

protected:
    CvSize _size;
    T *_ptr;
    int *_refCount;
    size_t _pitch; // pitch in bytes
};


template<typename T>
LinearMemory<T>::LinearMemory(): _ptr(NULL), _refCount(NULL)
{
}

template<typename T>
LinearMemory<T>::LinearMemory(CvSize size): _ptr(NULL), _refCount(NULL)
{
    init(size);
}

template<typename T>
LinearMemory<T>::LinearMemory(const LinearMemory<T> &otherLM):
    _ptr(NULL), _refCount(NULL)
{
    *this = otherLM;
}

template<typename T>
LinearMemory<T>::~LinearMemory()
{
    dealloc();
}

template<typename T>
void
LinearMemory<T>::init(CvSize size)
{
    dealloc();

    assert(size.width > 0 and size.height > 0);
    _size = size;

    _refCount = new int;
    *_refCount = 1;

    int widthInBytes = size.width * sizeof(T);
    assert(_ptr == NULL);
    CUDA_SAFE_CALL( cudaMallocPitch((void **)&_ptr, &_pitch,
                                    widthInBytes, size.height) );

    assert(_pitch % sizeof(T) == 0);
    assert(_ptr != NULL);
}

template<typename T>
void
LinearMemory<T>::dealloc()
{
    if(_ptr != NULL) {
        (*_refCount)--;
        if(*_refCount == 0) {
            CUDA_SAFE_CALL( cudaFree(_ptr) );
            _ptr = NULL;

            delete _refCount;
            _refCount = NULL;
        }
    }
}

template<typename T>
LinearMemory<T> &
LinearMemory<T>::operator=(const LinearMemory<T> &otherLM)
{
    if(otherLM._ptr != _ptr) {
        dealloc();

        _ptr = otherLM._ptr;
        _refCount = otherLM._refCount;

        if(_refCount) (*_refCount)++;
    }
    return *this;
}
}

#endif
