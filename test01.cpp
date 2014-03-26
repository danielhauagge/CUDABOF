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
#include <MemoryUtils.hpp>

#include <iomanip>

#include <test01_kernel.hpp>

using namespace CUDABOF;

#if 0
int
main(int argc, const char **)
{
  cv::Size size(1024, 768);
  
  cv::Mat image(size, CV_32FC1);
  for(int i = 0; i < 10; i++) image.at<float>(0,i) = i + 10;

  Array<float> arr(size);
  LinearMemory<float> lMem(size);

  CB_PRINT_LINE;
  memoryCopy(arr, image);

  runAddOneKernel(lMem.getPtr(), size.width, size.height, lMem.getPitch(), arr.getArray());

  memoryCopy(arr, lMem);

  for(int i = 0; i < 10; i++) CB_PRINT_VAR(image.at<float>(0,i));

  memoryCopy(image, arr);
  for(int i = 0; i < 10; i++) CB_PRINT_VAR(image.at<float>(0,i));


  memoryCopy(arr, image);
  
  return EXIT_SUCCESS;
}
#endif

void
printElems(const cv::Mat &img)
{
  for(int i = 0; i < 10; i++) {
    std::cout << std::setw(3) << img.at<float>(1, i) << "   ";
  }
  std::cout << "\n";
}

int
main(int argc, const char **)
{
  cv::Size size(1024, 768);
  
  cv::Mat image(size, CV_32FC1);
  cv::Mat ret_image(size, CV_32FC1);
  for(int i = 0; i < size.height; i++) 
    for(int j = 0; j < size.width; j++)
      image.at<float>(i,j) = i * size.width + j;

  Array<float> arr(size);
  LinearMemory<float> lMem(size);

  memoryCopy(arr, image);
  runAddOneKernel(lMem.getPtr(), size.width, size.height, lMem.getPitch(), arr.getArray());
  memoryCopy(arr, lMem);

  printElems(image);
  memoryCopy(ret_image, arr);
  printElems(ret_image);
  
  // Check the results
  bool passed = true;
  for(int i = 0; i < size.height and passed; i++) {
    for(int j = 0; j < size.width and passed; j++) {
      passed = (ret_image.at<float>(i,j) - image.at<float>(i,j)) == 1;
    }
  }

  if(passed) {
    std::cout << "Test PASSED" << std::endl;
    return EXIT_SUCCESS;
  } else { 
    std::cout << "Test FAILED" << std::endl;
    return EXIT_FAILURE;
  }
}
