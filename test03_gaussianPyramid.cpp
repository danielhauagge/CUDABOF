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


#include <CUDABOF.hpp>
#include <LinearMemory.hpp>
#include <Array.hpp>
#include <Pyramid.hpp>
#include <MemoryUtils.hpp>
#include <GaussianPyramidKernel.hpp>

using namespace CUDABOF;

int
main(int argc, char **argv)
{
    if(argc != 2) {
        std::cout << "Usage:\n\t" << argv[0] << " <image>" << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat image = cvLoadImage(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat floatImage(image.size(), CV_32FC1); //IPL_DEPTH_32F, 1);
    //cvConvertScale(image, floatImage, 1/255.0);
    image.convertTo(floatImage, CV_32FC1, 1 / 255.0);

    int nLevels = 5;
    Pyramid< Array<float> > gaussianPyramid(image.size(), nLevels);
    Pyramid< LinearMemory<float> > auxPyramid(image.size(), nLevels);

    memoryCopy(gaussianPyramid[0], floatImage);

    for(int i = 1; i < nLevels; i++) {

        gaussianPyramidDownsample(gaussianPyramid[i - 1].getArray(),
                                  auxPyramid[i].getPtr(),
                                  auxPyramid[i].getPitch(),
                                  auxPyramid[i].getSize().width, auxPyramid[i].getSize().height);
        memoryCopy(gaussianPyramid[i], auxPyramid[i]);
    }


    // Show results
    cv::namedWindow("image", 0);
    for(int i = 0; i < nLevels; i++) {
        cv::Mat image(gaussianPyramid[i].getSize(), CV_32FC1);//IPL_DEPTH_32F, 1);
        memoryCopy(image, gaussianPyramid[i]);

        cv::imshow("image", image);
        cvWaitKey(0);
    }

    return EXIT_SUCCESS;
}
