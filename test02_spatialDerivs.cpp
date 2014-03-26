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
#include <MemoryUtils.hpp>
#include <Array.hpp>
#include <BayesianOpticalFlowKernels.hpp>

#include <iomanip>

using namespace CUDABOF;

int
main(int argc, char **argv)
{
  if(argc != 2) {
    std::cout << "Usage:\n\t" << argv[0] << " <image>" << std::endl;
    return EXIT_SUCCESS;
  }

  cv::Mat aux = cvLoadImage(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat image(aux.size(), CV_32FC1);//IPL_DEPTH_32F, 1);
  aux.convertTo(image, CV_32FC1, 1/255.0);
  cv::Mat imageDx(image.size(), CV_32FC1);//IPL_DEPTH_32F, 1);
  cv::Mat imageDy(image.size(), CV_32FC1);//IPL_DEPTH_32F, 1);
  
  // Buffers for GPGPU
  Array<float> arrayImage(image.size());
  LinearMemory<float>  lMemDx(image.size()), lMemDy(image.size());

  // Transfer image to GPU
  memoryCopy(arrayImage, image);
  assert(lMemDy.getPitch() == lMemDx.getPitch());
  
  // Compute derivative
  spatialDeriv(arrayImage.getArray(),
	       lMemDx.getPtr(), lMemDy.getPtr(), 
	       lMemDx.getPitch(),
	       image.size().width, image.size().height);

  // Bring back results
  memoryCopy(imageDx, lMemDx);
  memoryCopy(imageDy, lMemDy);

  // debug
  for(int i = 10; i < 20; i++) {
    for(int j = 10; j < 20; j++) {
      std::cout << std::setw(4) << imageDx.at<float>(i,j) << "   ";
    }
    std::cout << "\n";
  }

  cv::normalize(imageDx, imageDx, 1, 0, CV_MINMAX);
  cv::normalize(imageDy, imageDy, 1, 0, CV_MINMAX);
  
  // Show results
  cv::namedWindow("image");
  cv::namedWindow("dx");
  cv::namedWindow("dy");

  cv::imshow("image", image);
  cv::imshow("dx", imageDx);
  cv::imshow("dy", imageDy);

  cv::waitKey(0);
}
