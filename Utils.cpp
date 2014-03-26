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


#include <Utils.hpp>

std::ostream &
operator<<(std::ostream &stream, const cudaDeviceProp &devProp)
{
    using namespace std;
    int w = 40;

    stream
            << "\n"
            << setw(w) << "name: " << devProp.name << "\n"
            << setw(w) << "total global memory: " << (devProp.totalGlobalMem >> 20) << " MB\n"
            << setw(w) << "shared memory per block: " << devProp.sharedMemPerBlock << " B\n"
            << setw(w) << "registers per block: " << devProp.regsPerBlock << "\n"
            << setw(w) << "warp size: " << devProp.warpSize << "\n"
            << setw(w) << "mem pitch: " << devProp.memPitch << "\n"
            << setw(w) << "max threads per block: " << devProp.maxThreadsPerBlock << "\n"
            << setw(w) << "max grid size: " << devProp.maxGridSize[0] << "x" << devProp.maxGridSize[1] << "x" << devProp.maxGridSize[2] << "\n"
            << setw(w) << "total const memory: " << devProp.totalConstMem << " B\n"
            << setw(w) << "major: " << devProp.major << "\n"
            << setw(w) << "minor: " << devProp.minor << "\n"
            << setw(w) << "clock rate: " << devProp.clockRate << " KHz\n"
            << setw(w) << "texture alignment: " << devProp.textureAlignment << "\n"
            << endl;

    return stream;
}

bool
operator==(const CvSize &a, const CvSize &b)
{
    return a.width == b.width and a.height == b.height;
}

bool
operator!=(const CvSize &a, const CvSize &b)
{
    return a.width != b.width or a.height != b.height;
}

std::ostream &
operator<<(std::ostream &stream, const CvSize &size)
{
    stream << size.width << "x" << size.height;
    return stream;
}

CvScalar
hsv2rgb(float h, float s, float v)
{
    h = MAX(h, 0);
    h = fmodf(h, 360.0);

    s = MAX(0, MIN(s, 1.0));
    v = MAX(0, MIN(v, 1.0));

    int hi = int(h / 60) % 6;
    float f = h / 60.0 - hi;
    float p = v * (1 - s);
    float q = v * (1 - f * s);
    float t = v * (1 - (1 - f) * s);

    v *= 255;
    q *= 255;
    p *= 255;
    t *= 255;

    switch(hi) {
    case 0:
        return CV_RGB(v, t, p);
    case 1:
        return CV_RGB(q, v, p);
    case 2:
        return CV_RGB(p, v, t);
    case 3:
        return CV_RGB(p, q, v);
    case 4:
        return CV_RGB(t, p, v);
    default:
        return CV_RGB(v, p, q);
    }
}

void
drawBayesianOpticalFlow(cv::Mat &displayImage,
                        const cv::Mat &dx, const cv::Mat &dy,
                        const cv::Mat &Lambda_xx, const cv::Mat &Lambda_yy, const cv::Mat &Lambda_xy,
                        cv::Size gridSize,
                        float lineMulFactor, float ellipseMulFactor)
{
    cv::Size size = displayImage.size();

    cv::Scalar color = CV_RGB(255, 0, 0);

    if(lineMulFactor > 0) {
        srand(0);

        for(int i = gridSize.height; i < (size.height - gridSize.height); i += gridSize.height) {
            for(int j = gridSize.width; j < (size.width - gridSize.width); j += gridSize.width) {
                cv::Point p1(j, i);
                cv::Point p2;

                float x = dx.at<float>(i, j);
                float y = dy.at<float>(i, j);

                x *= lineMulFactor;
                y *= lineMulFactor;

                p2.x = p1.x + (int)x;
                p2.y = p1.y + (int)y;

                color = hsv2rgb(rand() % 360, 1, 1);

                cv::line(displayImage, p1, p2, color, 1);
            }
        }
    }

    if(ellipseMulFactor > 0) {
        srand(0);

        cv::Mat covar(2, 2, CV_32F);//, eigenvals(2, 1, CV_32F), eigenvecs(2, 2, CV_32F);

        cv::Size axes;
        float xx, yy, xy;
        float angle;
        float eval1, eval2;
        CvPoint p1;
        for(int i = gridSize.height; i < (size.height - gridSize.height); i += gridSize.height) {
            for(int j = gridSize.width; j < (size.width - gridSize.width); j += gridSize.width) {
                p1 = cvPoint(j, i);

                xx = Lambda_xx.at<float>(i, j); //  (CV_IMAGE_ELEM((const IplImage*)Lambda_xx, float, i, j));
                yy = Lambda_yy.at<float>(i, j); //(CV_IMAGE_ELEM((const IplImage*)Lambda_yy, float, i, j));
                xy = Lambda_xy.at<float>(i, j); //(CV_IMAGE_ELEM((const IplImage*)Lambda_xy, float, i, j));

                covar.at<float>(0, 0) = xx; //CV_MAT_ELEM(*(const CvMat*)covar, float, 0, 0) = xx;
                covar.at<float>(1, 1) = yy; //CV_MAT_ELEM(*(const CvMat*)covar, float, 1, 1) = yy;
                covar.at<float>(1, 0) = covar.at<float>(0, 1) = xy; //CV_MAT_ELEM(*(const CvMat*)covar, float, 1, 0) = CV_MAT_ELEM(*(const CvMat*)covar, float, 0, 1) = xy;

#if 0
                cv::SVD(covar, eigenvals, eigenvecs, 0, CV_SVD_MODIFY_A);

                color = hsv2rgb(rand() % 360, 1, 1);
#else
                cv::SVD svd(covar,  cv::SVD::MODIFY_A);
                cv::Mat &eigenvals = svd.w;
                cv::Mat &eigenvecs = svd.u;
#endif
                eval1 = ellipseMulFactor * 0.01 * eigenvals.at<float>(0, 0); //CV_MAT_ELEM(*(const CvMat*)eigenvals, float, 0, 0);
                eval2 = ellipseMulFactor * 0.01 * eigenvals.at<float>(1, 0); //CV_MAT_ELEM(*(const CvMat*)eigenvals, float, 1, 0);

                // singular value = sqrt(eigenvalue)
                eval1 = eval1 * eval1;
                eval2 = eval2 * eval2;

                axes = cv::Size(MAX(int(eval1), 1), MAX(int(eval2), 1));

                //angle = acos(CV_MAT_ELEM(*(const CvMat*)eigenvecs, float, 0, 0)) * 180 / M_PI;
                angle = acos(eigenvecs.at<float>(0, 0)) * 180 / M_PI;

                cv::ellipse(displayImage, p1, axes, angle, 0, 360, color);
            }
        }

    }
}
