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
#include <OpticalFlow.hpp>

//#include <Array.hpp> // debug


#include <sstream>
#include <iomanip>

#include <cutil.h>

using namespace CUDABOF;

void
drawText(cv::Mat &image, cv::Point position, int lineNo, const char *text)
{
  //CvFont font;
  //cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1, 1);
  
  position.y += 15 * lineNo;
  
  cv::putText(image, text, position, CV_FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
}

template<typename T>
void
drawLabelValuePair(cv::Mat &image, cv::Point position, int lineNo, const char *label, T value)
{
  std::stringstream text;
  text << std::setw(15) << std::right << label << " = " << value;

  drawText(image, position, lineNo, text.str().c_str());
}

int
main(int argc, char **argv)
{
  OpticalFlow optFlow;

  cv::Mat mu_x, mu_y;
  cv::Mat Lambda_xx, Lambda_xy, Lambda_yy;

  // Timing variables
  double gpuTime, gpuAccumTime = 0.0;
  double totalWeight = 0.0, fps = 0;;
  unsigned int timer;
  CUT_SAFE_CALL( cutCreateTimer(&timer) );

  /* *
   * Optical flow algorithm parameters
   * */
  float lambdaZero  = 0.15;
  float lambdaOne   = 0.5;
  float lambdaTwo   = 0.5;
  float lambdaPrior = 0.1;
  int windowSide    = 3;


  /* *
   * Process command line arguments
   * */
  if((argc == 2) and ((strcmp(argv[1], "-h") == 0) or (strcmp(argv[1], "--help") == 0))) {
    std::cerr << "Usage:\n\t" << argv[0] << " <movie.avi>" << std::endl;
    return EXIT_FAILURE;
  }

  /* *
   * Load movie, or capture from cam
   * */
  cv::VideoCapture capture;
  if(argc == 2) {
    capture.open(argv[1]);
    if(not capture.isOpened()) {
      std::cerr << "Could not open file " << argv[1] << std::endl;
      return EXIT_FAILURE;
    }
  } else {
    capture.open(0); // Any camera is good
    CB_PRINT_VAR(capture.isOpened());
    if(not capture.isOpened()) {
      std::cerr << "Could not capture from camera." << std::endl;
      return EXIT_FAILURE;
    }
  }


  cv::Mat frame;
  cv::Size frameSize;

  cv::Mat frames[3], auxBW;
  int prevIdx = 0, currIdx = 1, nextIdx = 2;

  capture >> frame; //framePtr = cvQueryFrame(capture);
  
  frameSize = frame.size();// cvGetSize(framePtr);

  
  // garantee that the smallest image side is bigger than 32 pixels
  int maxLevels = int( log(MIN(frameSize.width, frameSize.height))/log(2.0) - 4.0 );
  int nLevels   = maxLevels;

  auxBW.create(frameSize, CV_8UC1);
  for(int i = 0; i < 3; i++)
    frames[i].create(frameSize, CV_32FC1);

  mu_x.create(frameSize, CV_32FC1);//IPL_DEPTH_32F, 1);
  mu_y.create(frameSize, CV_32FC1);//, IPL_DEPTH_32F, 1);
  Lambda_xx.create(frameSize, CV_32FC1);//, IPL_DEPTH_32F, 1);
  Lambda_yy.create(frameSize, CV_32FC1);//, IPL_DEPTH_32F, 1);
  Lambda_xy.create(frameSize, CV_32FC1);//, IPL_DEPTH_32F, 1);

  cv::cvtColor(frame, auxBW, CV_BGR2GRAY);  
  auxBW.convertTo(frames[prevIdx], CV_32FC1, 1/255.0);

  capture >> frame;
  cv::cvtColor(frame, auxBW, CV_BGR2GRAY);  
  auxBW.convertTo(frames[currIdx], CV_32FC1, 1/255.0);

  optFlow.init(frames[prevIdx], frames[currIdx],
	       nLevels, windowSide,
	       lambdaZero, lambdaOne, lambdaTwo, lambdaPrior);

  /* *
   * Simple GUI
   * */
  CvPoint textCorner = cvPoint(20, 20);
  
  int pressedKey = 0;
  const char *displayWindowName = "Optical Flow Demo";
  const char *controlWindowName = "Optical Flow Controls";

  CvSize infoImageSize = cvSize(300, 200);

  cvNamedWindow(displayWindowName, 0);
  cvResizeWindow(displayWindowName, frameSize.width, frameSize.height);
  cvMoveWindow(displayWindowName, 0, 0);

  cvNamedWindow(controlWindowName, 0);
  cvMoveWindow(controlWindowName, 0, 0);
  cvMoveWindow(controlWindowName, frameSize.width + 30, infoImageSize.height + 60);


  cv::Mat displayImage(frameSize, CV_8UC3);//IPL_DEPTH_8U, 3);
  cv::Mat infoImage(infoImageSize, CV_8UC3);//, IPL_DEPTH_8U, 3);

  int lineMulFactor = 1, ellipseMulFactor = 0, gridSpacing = 10; // Optical flow drawing parameters
  int lambdaZero_x_1000  = int(1000.0 * lambdaZero);
  int lambdaOne_x_1000   = int(1000.0 * lambdaOne);
  int lambdaTwo_x_1000   = int(1000.0 * lambdaTwo);
  int lambdaPrior_x_1000 = int(1000.0 * lambdaPrior); 


  cvCreateTrackbar("line mul factor", controlWindowName, &lineMulFactor, 10000, NULL);
  cvCreateTrackbar("ellipse mul factor", controlWindowName, &ellipseMulFactor, 80, NULL);
  cvCreateTrackbar("grid spacing", controlWindowName, &gridSpacing, 100, NULL);
  cvCreateTrackbar("lambdaZero x 1000", controlWindowName, &lambdaZero_x_1000, 2000, NULL);
  cvCreateTrackbar("lambdaOne x 1000", controlWindowName, &lambdaOne_x_1000, 2000, NULL);
  cvCreateTrackbar("lambdaTwo x 1000", controlWindowName, &lambdaTwo_x_1000, 2000, NULL);
  cvCreateTrackbar("lambdaPrior x 1000", controlWindowName, &lambdaPrior_x_1000, 2000, NULL);
  cvCreateTrackbar("n levels", controlWindowName, &nLevels, maxLevels, NULL);

  std::cout << "Press ESC to exit." << std::endl;

  while((pressedKey = cvWaitKey(10)) != 'q') { //CB_ESC) {

    gridSpacing = MAX(gridSpacing, 1);
    nLevels = MAX(nLevels, 1);

    lambdaZero  = float(lambdaZero_x_1000)  / 1000.0;
    lambdaOne   = float(lambdaOne_x_1000)   / 1000.0;
    lambdaTwo   = float(lambdaTwo_x_1000)   / 1000.0;
    lambdaPrior = float(lambdaPrior_x_1000) / 1000.0;

    optFlow.setLambdaZero(lambdaZero);
    optFlow.setLambdaOne(lambdaOne);
    optFlow.setLambdaTwo(lambdaTwo);
    optFlow.setLambdaPrior(lambdaPrior);
    
    capture >> frame;
    if(frame.empty()) break;
    cv::cvtColor(frame, auxBW, CV_BGR2GRAY);  
    auxBW.convertTo(frames[nextIdx], CV_32FC1, 1/255.0);

        
    /* *
     * Compute optical flow
     * */
    if(optFlow.getNLevels() != nLevels) {
      optFlow.init(frames[prevIdx], frames[currIdx],
		   nLevels, windowSide,
		   lambdaZero, lambdaOne, lambdaTwo, lambdaPrior);    
    }
    
    CUT_SAFE_CALL( cutResetTimer(timer) );
    CUT_SAFE_CALL( cutStartTimer(timer) );
    
    optFlow(frames[nextIdx], mu_x, mu_y, Lambda_xx, Lambda_yy, Lambda_xy);

    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    CUT_SAFE_CALL( cutStopTimer(timer) );
    gpuTime = cutGetTimerValue(timer);
    gpuAccumTime = gpuTime + 0.9 * gpuAccumTime;
    totalWeight  = 1.0 + 0.9 * totalWeight;
    fps = totalWeight / (gpuAccumTime / 1000.0) /*gpuAccumTime is in ms*/; 

    fps = 1000 / gpuTime;

    /* *
     * Display results
     * */
    frame.convertTo(displayImage, CV_8UC3);
    drawBayesianOpticalFlow(displayImage, 
			    mu_x, mu_y,
			    Lambda_xx, Lambda_yy, Lambda_xy,
			    cvSize(gridSpacing, gridSpacing),
			    lineMulFactor, ellipseMulFactor);
    
    int lineNo = 0;    
    infoImage.setTo(cv::Scalar(0,0,0));
    drawText(infoImage, textCorner, lineNo++, "Press ESC to exit");
    lineNo++;
    drawLabelValuePair(infoImage, textCorner, lineNo++, "lambda zero", lambdaZero);
    drawLabelValuePair(infoImage, textCorner, lineNo++, "lambda one", lambdaOne);
    drawLabelValuePair(infoImage, textCorner, lineNo++, "lambda two", lambdaTwo);
    drawLabelValuePair(infoImage, textCorner, lineNo++, "lambda prior", lambdaPrior);
    drawLabelValuePair(infoImage, textCorner, lineNo++, "FPS", fps); 
    drawLabelValuePair(infoImage, textCorner, lineNo++, "n Levels", nLevels); 

    cv::imshow(displayWindowName, displayImage);
    cv::imshow(controlWindowName, infoImage);


    prevIdx = currIdx;
    currIdx = nextIdx;
    nextIdx = (nextIdx + 1)%3;
  }
  
  return EXIT_SUCCESS;
}
