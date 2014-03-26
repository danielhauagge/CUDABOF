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


#include <OpticalFlow.hpp>

namespace CUDABOF
{
/*******
 * Helper function
 *******/
void
fillGaussianPyramid(Pyramid<Array<float> > &gaussPyr,
                    Pyramid< LinearMemory<float> > &aux)
{
    int nLevels = gaussPyr.getNLevels();

    for(int i = 1; i < nLevels; i++) {
        gaussianPyramidDownsample(gaussPyr[i - 1].getArray(),
                                  aux[i].getPtr(),
                                  aux[i].getPitch(),
                                  aux[i].getSize().width, aux[i].getSize().height);
        memoryCopy(gaussPyr[i], aux[i]);
    }
}

OpticalFlow::OpticalFlow()
{
}


OpticalFlow::OpticalFlow(const cv::Mat &prevImage, const cv::Mat &currImage,
                         int nLevels, int windowSide,
                         float lambdaZero, float lambdaOne, float lambdaTwo, float lambdaPrior)
{
    init(prevImage, currImage, nLevels, windowSide, lambdaZero, lambdaOne, lambdaTwo, lambdaPrior);
}

void
OpticalFlow::init(const cv::Mat &prevImage, const cv::Mat &currImage,
                  int nLevels, int windowSide,
                  float lambdaZero, float lambdaOne, float lambdaTwo, float lambdaPrior)
{
    assert(prevImage.size() == currImage.size());
    assert(prevImage.depth() == CV_32FC1);
    assert(currImage.depth() == CV_32FC1);

    _nLevels = nLevels;
    _windowSide = windowSide;
    _lambdaZero = lambdaZero;
    _lambdaOne = lambdaOne;
    _lambdaTwo = lambdaTwo;
    _lambdaPrior = lambdaPrior;

    _nextIndex = 2;
    _currIndex = 1;
    _prevIndex = 0;

    // Allocate all pyramids
    CvSize size = prevImage.size();
    for(int i = 0; i < 3; i++) {
        _gaussPyr[i].init(size, nLevels);
    }
    _Mxx.init(size, nLevels);
    _Myy.init(size, nLevels);
    _Mxy.init(size, nLevels);
    _bx.init(size, nLevels);
    _by.init(size, nLevels);
    _Idx.init(size, nLevels);
    _Idy.init(size, nLevels);
    _Idt.init(size, nLevels);
    _Lambdaxx.init(size, nLevels);
    _Lambdayy.init(size, nLevels);
    _Lambdaxy.init(size, nLevels);
    _mux.init(size, nLevels);
    _muy.init(size, nLevels);
    _normalizingFactor.init(_gaussPyr[0][nLevels - 1].getSize());

    for(int i = 0; i < N_AUX; i++) _aux[i].init(size, nLevels);

    // Initialize gaussian pyramids
    memoryCopy(_gaussPyr[_currIndex][0], prevImage);
    fillGaussianPyramid(_gaussPyr[_currIndex], _aux[0]);

    memoryCopy(_gaussPyr[_nextIndex][0], currImage);
    fillGaussianPyramid(_gaussPyr[_nextIndex], _aux[0]);

}

void
OpticalFlow::_incrementIndex()
{
    _prevIndex = _currIndex;
    _currIndex = _nextIndex;
    _nextIndex = (_nextIndex + 1) % 3;
}

void
OpticalFlow::operator()(const cv::Mat &nextImage)
{
    _computeFlow(nextImage);
}

void
OpticalFlow::operator()(const cv::Mat &nextImage,
                        cv::Mat &mu_x, cv::Mat &mu_y,
                        cv::Mat &Lambda_xx, cv::Mat &Lambda_yy, cv::Mat &Lambda_xy)
{
    _computeFlow(nextImage);

    memoryCopy(Lambda_xx, _Lambdaxx[0]);
    memoryCopy(Lambda_yy, _Lambdayy[0]);
    memoryCopy(Lambda_xy, _Lambdaxy[0]);

    memoryCopy(mu_x, _mux[0]);
    memoryCopy(mu_y, _muy[0]);
}

void
OpticalFlow::_computeFlow(const cv::Mat &nextImage)
{
    _incrementIndex();

    assert(nextImage.depth() == CV_32FC1);//IPL_DEPTH_32F);
    assert(nextImage.channels() == 1);

    memoryCopy(_gaussPyr[_nextIndex][0], nextImage);
    fillGaussianPyramid(_gaussPyr[_nextIndex], _aux[0]);

    /* *
     * Compute flow for base level
     * */
    int level = _nLevels - 1;
    size_t pitch = _aux[0][level].getPitch();
    CvSize size  = _aux[0][level].getSize();

    // Base level 1 kernel
    bofBaseLevel1(_gaussPyr[_prevIndex][level].getArray(),
                  _gaussPyr[_currIndex][level].getArray(),
                  _gaussPyr[_nextIndex][level].getArray(),
                  _aux[0][level].getPtr() /*Idx*/, _aux[1][level].getPtr()/*Idy*/,
                  _aux[2][level].getPtr() /*Idt*/, _aux[3][level].getPtr() /*normalizingFactor*/,
                  pitch, size.width, size.height,
                  _lambdaOne, _lambdaTwo);

    memoryCopy(_Idx[level], _aux[0][level]);
    memoryCopy(_Idy[level], _aux[1][level]);
    memoryCopy(_Idt[level], _aux[2][level]);
    memoryCopy(_normalizingFactor, _aux[3][level]);

    // Base level 2 kernel
    bofBaseLevel2(_Idx[level].getArray(), _Idy[level].getArray(), _Idt[level].getArray(),
                  _normalizingFactor.getArray(),
                  _aux[0][level].getPtr() /*Mxx*/, _aux[1][level].getPtr() /*Myy*/,
                  _aux[2][level].getPtr() /*Mxy*/,
                  _aux[3][level].getPtr() /*bx*/, _aux[4][level].getPtr() /*by*/,
                  pitch, size.width, size.height);

    memoryCopy(_Mxx[level], _aux[0][level]);
    memoryCopy(_Myy[level], _aux[1][level]);
    memoryCopy(_Mxy[level], _aux[2][level]);
    memoryCopy(_bx[level], _aux[3][level]);
    memoryCopy(_by[level], _aux[4][level]);


    // Base level 3 kernel
    bofBaseLevel3(_Mxx[level].getArray(), _Myy[level].getArray(), _Mxy[level].getArray(),
                  _bx[level].getArray(), _by[level].getArray(),
                  _aux[0][level].getPtr() /*Lambdaxx*/, _aux[1][level].getPtr() /*Lambdayy*/, _aux[2][level].getPtr() /*Lambdaxy*/,
                  _aux[3][level].getPtr() /*mux*/, _aux[4][level].getPtr() /*muy*/,
                  pitch, size.width, size.height,
                  _lambdaPrior, _windowSide);

    memoryCopy(_Lambdaxx[level], _aux[0][level]);
    memoryCopy(_Lambdayy[level], _aux[1][level]);
    memoryCopy(_Lambdaxy[level], _aux[2][level]);
    memoryCopy(_mux[level], _aux[3][level]);
    memoryCopy(_muy[level], _aux[4][level]);


    /* *
     * Refine flow for other levels
     * */
    for(level--; level >= 0; level--) {
        pitch = _aux[0][level].getPitch();
        size  = _aux[0][level].getSize();

        bofRefineEstimate(_gaussPyr[_prevIndex][level].getArray(),
                          _gaussPyr[_currIndex][level].getArray(),
                          _gaussPyr[_nextIndex][level].getArray(),
                          _Lambdaxx[level + 1].getArray(), _Lambdayy[level + 1].getArray(), _Lambdaxy[level + 1].getArray(),
                          _mux[level + 1].getArray(), _muy[level + 1].getArray(),
                          _aux[0][level].getPtr() /*Lambdaxx*/, _aux[1][level].getPtr() /*Lambdayy*/, _aux[2][level].getPtr() /*Lambdaxy*/,
                          _aux[3][level].getPtr() /*mux*/, _aux[4][level].getPtr() /*muy*/,
                          pitch, size.width, size.height,
                          _lambdaZero, _lambdaOne, _lambdaTwo);

        memoryCopy(_Lambdaxx[level], _aux[0][level]);
        memoryCopy(_Lambdayy[level], _aux[1][level]);
        memoryCopy(_Lambdaxy[level], _aux[2][level]);
        memoryCopy(_mux[level], _aux[3][level]);
        memoryCopy(_muy[level], _aux[4][level]);
    }
}

void
OpticalFlow::setLambdaZero(float lambdaZero)
{
    _lambdaZero = lambdaZero;
}

void
OpticalFlow::setLambdaOne(float lambdaOne)
{
    _lambdaOne = lambdaOne;
}

void
OpticalFlow::setLambdaTwo(float lambdaTwo)
{
    _lambdaTwo = lambdaTwo;
}

void
OpticalFlow::setLambdaPrior(float lambdaPrior)
{
    _lambdaPrior = lambdaPrior;
}

float
OpticalFlow::getLambdaZero() const
{
    return _lambdaZero;
}

float
OpticalFlow::getLambdaOne() const
{
    return _lambdaOne;
}

float
OpticalFlow::getLambdaTwo() const
{
    return _lambdaTwo;
}

float
OpticalFlow::getLambdaPrior() const
{
    return _lambdaPrior;
}

int
OpticalFlow::getNLevels() const
{
    return _nLevels;
}
} // namespace
