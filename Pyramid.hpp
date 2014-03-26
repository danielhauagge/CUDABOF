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


#ifndef __PYRAMID_HPP__
#define __PYRAMID_HPP__

#include <CUDABOF.hpp>

#include <vector>

namespace CUDABOF
{

template<typename T>
class Pyramid
{
public:
    Pyramid();
    Pyramid(cv::Size baseSize, int nLevels);

    void init(cv::Size baseSize, int nLevels);

    T &operator[](int i);
    const T &operator[](int i) const;

    int getNLevels() const { return _nLevels; }
    cv::Size getBaseSize() const { return _baseSize; }

protected:
    std::vector<T> _pyr;
    cv::Size _baseSize;
    int _nLevels;
};

template<typename T>
Pyramid<T>::Pyramid() {}

template<typename T>
Pyramid<T>::Pyramid(cv::Size baseSize, int nLevels)
{
    init(baseSize, nLevels);
}

template<typename T>
void
Pyramid<T>::init(cv::Size baseSize, int nLevels)
{
    assert(nLevels > 0);

    _baseSize = baseSize;
    _nLevels  = nLevels;

    _pyr.resize(nLevels);
    for(int i = 0; i < nLevels; i++) {
        assert(baseSize.width > 0 and baseSize.height > 0);

        _pyr[i].init(baseSize);
        baseSize.width  = (baseSize.width + 1) / 2;
        baseSize.height = (baseSize.height + 1) / 2;
    }
}

template<typename T>
T &
Pyramid<T>::operator[](int i)
{
    return _pyr[i];
}

template<typename T>
const T &
Pyramid<T>::operator[](int i) const
{
    return _pyr[i];
}

} // namespace CUDABOF
#endif
