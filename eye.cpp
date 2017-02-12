#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>

using namespace cv;
using namespace std;

//eyeball radius range
#define MINRAD 10
#define MAXRAD 25
//dark pixel value threshold
#define DARKPIXELVALUE 10
//voting result threshold
#define RESULTTHRESHOLD 3.0

Point eyepoint_detect
    (
    Mat& imeye,
    bool& is_locate_correct
    )
{
    Mat blurim;
    //blur noise
    GaussianBlur(imeye, blurim, Size(9,9), 0);
    //calc gradient, curvature
    float kernel[3] = {-0.5f, 0.0f, 0.5f};
    Mat filterX(1, 3, CV_32F, kernel);
    Mat filterY(3, 1, CV_32F, kernel);
    Mat Lx, Ly;
    filter2D(blurim, Lx, CV_32F, filterX, Point (-1, -1), 0, BORDER_REPLICATE);
    filter2D(blurim, Ly, CV_32F, filterY, Point (-1, -1), 0, BORDER_REPLICATE);
    Mat Lxx, Lxy, Lyy, Lyx;
    filter2D(Lx, Lxx, CV_32F, filterX, Point (-1, -1), 0, BORDER_REPLICATE);
    filter2D(Lx, Lxy, CV_32F, filterY, Point (-1, -1), 0, BORDER_REPLICATE);
    filter2D(Ly, Lyx, CV_32F, filterX, Point (-1, -1), 0, BORDER_REPLICATE);
    filter2D(Ly, Lyy, CV_32F, filterY, Point (-1, -1), 0, BORDER_REPLICATE);
    Mat Lx2, Ly2;
    pow(Ly, 2, Ly2);
    pow(Lx, 2, Lx2);
    Mat Lvv = Ly2.mul(Lxx) - 2 * Lx.mul(Lxy).mul(Ly) + Lx2.mul(Lyy);
    Mat Lw = Lx2 + Ly2;
    Mat Lw15;
    pow(Lw, 1.5, Lw15);
    Mat k;
    divide(-Lvv, Lw15, k);
    Mat LwoverLvv;
    divide(Lw, Lvv, LwoverLvv);
    //calc displacement
    Mat Dx, Dy;
    Dx = -Lx.mul(LwoverLvv);
    Dy = -Ly.mul(LwoverLvv);
    Mat Dx2, Dy2, magnitude_displacement;
    pow(Dx, 2, Dx2);
    pow(Dy, 2, Dy2);
    cv::sqrt(Dx2+Dy2, magnitude_displacement);
    //calc curvedness as voting weight
    Mat Lxx2, Lxy2, Lyy2, curvednesstemp, curvedness;
    pow(Lxx, 2, Lxx2);
    pow(Lxy, 2, Lxy2);
    pow(Lyy, 2, Lyy2);
    cv::sqrt(Lxx2 + 2 * Lxy2 + Lyy2 , curvednesstemp);
    curvedness = cv::abs(curvednesstemp);
    //voting
    const int srcheight = blurim.rows;
    const int srcwidth = blurim.cols;
    Mat center_map(srcheight, srcwidth, CV_32F, 0.0f);
    for(int y = 0; y < srcheight; y++)
        {
        for(int x = 0; x < srcwidth; x++)
            {
            if(Dx.at<float>(y, x) == 0.0f && Dy.at<float>(y, x) == 0.0f)
                {
                continue;
                }
            int dstx = x + (int)Dx.at<float>(y, x);
            int dsty = y + (int)Dy.at<float>(y, x);
            int radius = (int)magnitude_displacement.at<float>(y, x);
            if(dstx > 0 && dsty > 0 && dstx < srcwidth && dsty < srcheight)
                {
                //consider gradients from white to black only
                if(k.at<float>(y, x) < 0)
                    {
                    //limit eyeball radius
                    if(radius >= MINRAD && radius <= MAXRAD)
                        {
                        //consider eyeball contour only
                        if(imeye.at<float>(y, x) < DARKPIXELVALUE)
                            {
                            center_map.at<float>(dsty, dstx) += curvedness.at<float>(y, x);
                            }
                        }
                    }
                }
            }
        }
    //smooth the result
    Mat gauss_blured_center_map(srcheight, srcwidth, CV_32F, 0.0f);
    GaussianBlur(center_map, gauss_blured_center_map, Size(5,5), 0);
    //get location
    double maxv = 0, minv = 0;
    Point minlocation, maxlocation;
    minMaxLoc(gauss_blured_center_map, &minv, &maxv, &minlocation, &maxlocation);
    if(maxv > RESULTTHRESHOLD)
        {
        is_locate_correct = true;
        }
    else
        {
        is_locate_correct = false;
        }
    return maxlocation;
}

int main
    (
    void
    )
{
    Mat im;
    Mat gray;
    Mat eqgray;
    VideoCapture cap;
    cap.open(0);
    if(!cap.isOpened())
        {
            cout << "Failed to open cam" << endl;
            exit(EXIT_FAILURE);
        }
    vector<Rect> face_results;
    CascadeClassifier opencv_faceclassifier;
    opencv_faceclassifier.load("haarcascade_frontalface_alt2.xml");
    //main loop
    while(1)
        {
        cap >> im;
        cvtColor(im, gray, CV_BGR2GRAY);
        equalizeHist(gray, eqgray);
        opencv_faceclassifier.detectMultiScale(eqgray, face_results, 1.2, 2, CV_HAAR_FIND_BIGGEST_OBJECT, Size(100, 100));
        if(face_results.size() > 0)
            {
            //pick the biggest face
            Rect face = face_results[0];
            int dist1 = face.width >> 3;
            int dist2 = face.width >> 2;
            //set eye regions
            Rect lefteyeroi = Rect(face.x + dist1, face.y + dist2, dist1 + dist2, dist2);
            Rect righteyeroi = Rect(face.x +  2 * dist1 + dist2, face.y + dist2, dist1 + dist2, dist2);
            Mat lefteyeim = eqgray(lefteyeroi);
            Mat righteyeim = eqgray(righteyeroi);
            bool leftfound = false;
            bool rightfound = false;
            //detect
            Point lefteyepos = eyepoint_detect(lefteyeim, leftfound);
            Point righteyepos = eyepoint_detect(righteyeim, rightfound);
            //draw result
            if(leftfound)
                {
                circle(im, lefteyepos + lefteyeroi.tl(), 4, Scalar(0, 255, 0), -1);
                }
            if(rightfound)
                {
                circle(im, righteyepos + righteyeroi.tl(), 4, Scalar(0, 255, 0), -1);
                }
            }
        imshow("eye", im);
        //press esc to end
        if(waitKey(10) == 27)
            {
            break;
            }
        }
    return 0;
}
